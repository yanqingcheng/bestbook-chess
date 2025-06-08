# opening_book.py – backward-induction chess opening book
# ---------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path

import chess  # python-chess for FEN handling and move application
from chess import pgn  # PGN export
import datetime

import hashlib
import time
import requests
from urllib.parse import quote_plus

import threading

# ---------------------------   
#  Final counter for each depth
# ---------------------------   
from collections import Counter
import atexit

NODE_COUNT   = 0               # total nodes ever instantiated
DEPTH_HIST   = Counter()       # depth → count
REPORT_EVERY = 1               # print every N new nodes

def _bump(depth: int) -> None:
    """Call this once per new node."""
    global NODE_COUNT
    NODE_COUNT  += 1
    DEPTH_HIST[depth] += 1

    if (NODE_COUNT > 1) and (NODE_COUNT % REPORT_EVERY == 0):
        live = " ".join(f"{d}:{DEPTH_HIST[d]}" for d in sorted(DEPTH_HIST))
        print(f"\rnodes {NODE_COUNT:,} | {live}", end="\r", flush=True)

@atexit.register
def _report_totals() -> None:
    """Print a summary when the program finishes or is Ctrl-C’d."""
    if NODE_COUNT == 0:   # nothing ran
        return
    print("\n\n=== Opening-book summary ===")
    print(f"total nodes: {NODE_COUNT:,}")
    for d in sorted(DEPTH_HIST):
        print(f" depth {d}: {DEPTH_HIST[d]:,}")
    print("============================\n")

def _reset_counters() -> None:
    """Zero the node & depth counters for a fresh search."""
    global NODE_COUNT, DEPTH_HIST
    NODE_COUNT = 0
    DEPTH_HIST.clear()

# ---------------------------
# Core data structures
# ---------------------------

@dataclass
class Node:
    """A position in the opening tree.

    Attributes:
        fen: FEN string describing the board layout (may repeat across paths)
        turn_white: True if it’s White to move
        depth: Ply distance from the root
        parent: Reference to parent Node, or None at root
        children: OrderedDict mapping UCI move string -> (probability, child Node)
        value: Backward-induction value (+1 win, 0 draw, -1 loss)
        best_move: Chosen move UCI for White, or None for Black nodes
        games: Number of games reaching this position (for future pruning)
        reach_prob: Probability of reaching this position under the book policy
    """
    fen: str
    turn_white: bool
    depth: int
    parent: Node | None = None
    children: OrderedDict[str, tuple[float, Node]] = field(default_factory=OrderedDict)
    value: float = 0.0
    best_move: str | None = None
    games: int = 0
    reach_prob: float = 1.0

    def __repr__(self) -> str:  # pragma: no cover
        mv = f" best={self.best_move}" if self.best_move else ""
        return f"<Node depth={self.depth} value={self.value:.3f}{mv}>"


# ---------------------------
# Configuration helpers
# ---------------------------

DEFAULT_CONFIG = {
    "max_depth": 12,                # Ply depth (6 full moves)
    "min_reach_probability": 0.001, # Min probability to explore a branch
    "min_games": 5,                 # Min games to consider a branch
    "book_side": "white",           # Side to play the book moves (white or black)
}

def load_config(path: str | Path | None = None) -> dict[str, float | int]:
    """Load a JSON config or fall back to defaults."""
    if path is None:
        return DEFAULT_CONFIG.copy()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)
    with path.open() as fp:
        user_cfg = json.load(fp)
    cfg = DEFAULT_CONFIG.copy()
    cfg.update(user_cfg)
    return cfg

# ---------------------------
# Lichess Explorer helpers
# ---------------------------

CACHE_DIR = Path(".cache")
CACHE_DIR.mkdir(exist_ok=True)
LICHESS_EXPLORER_URL = "https://explorer.lichess.ovh/masters"

def _cache_path(key: str) -> Path:
    """Return a filesystem-safe path for this cache key."""
    digest = hashlib.md5(key.encode()).hexdigest()  # short and opaque
    return CACHE_DIR / f"{digest}.json"

# ---------------------------
# Robust JSON fetch with rate-limit + cache
# ---------------------------

_MIN_INTERVAL = 1.5          # seconds between hits to explorer.lichess.ovh
_MAX_RETRIES  = 3            # on 429
_BACKOFF_BASE = 2.0          # exponential factor

_last_hit      = 0.0         # monotonic timestamp of last *successful* call
_lock          = threading.Lock()

def _get_json(url: str, *, ttl: int = 86_400) -> dict:
    """
    Fetch URL with on-disk cache **and** polite rate-limit.

    • Waits _MIN_INTERVAL secs between live requests.
    • Retries on 429 (Too Many Requests) up to _MAX_RETRIES times, with
      exponential back-off.
    • Falls back to stale cache (if any) when all retries fail.
    """
    global _last_hit
    path = _cache_path(url)

    # 1) Fresh-cache fast path
    if path.exists():
        age = time.time() - path.stat().st_mtime
        if age < ttl:
            return json.loads(path.read_text())

    # 2) Live fetch with global rate-limit
    for attempt in range(_MAX_RETRIES):
        with _lock:
            wait = _MIN_INTERVAL - (time.monotonic() - _last_hit)
            if wait > 0:
                time.sleep(wait)

        try:
            resp = requests.get(url, timeout=10)
            if resp.status_code == 429:
                raise requests.HTTPError("429 Too Many Requests")

            resp.raise_for_status()
            data = resp.json()

            # cache and record timestamp
            path.write_text(json.dumps(data))
            with _lock:
                _last_hit = time.monotonic()
            return data

        except requests.HTTPError as http_exc:
            if "429" in str(http_exc):
                # exponential back-off
                backoff = _BACKOFF_BASE ** attempt
                time.sleep(backoff)
                continue  # retry
            raise  # other HTTP errors propagate

        except Exception as exc:
            # network error—try stale cache?
            if path.exists():
                return json.loads(path.read_text())
            raise RuntimeError(f"API call failed and no cache: {exc}") from exc

    # 3) Exhausted retries ➜ use stale cache or fail
    if path.exists():
        return json.loads(path.read_text())
    raise RuntimeError("Exceeded rate-limit retries and no cached data.")

# ---------------------------
# Fetching raw move stats from Lichess Masters explorer
# ---------------------------

def fetch_moves(fen: str) -> list[dict[str, int | str]]:
    """Return raw move stats from the Lichess Masters explorer."""

    url  = f"{LICHESS_EXPLORER_URL}?fen={quote_plus(fen)}&moves=50"
    data = _get_json(url)
    return data.get("moves", [])

# ---------------------------
# Tree construction - one ply expansion
# ---------------------------

def expand(node: Node, cfg: dict) -> None:
    """Grow one ply beneath node, pruning on reach_prob."""
    if node.depth >= cfg["max_depth"]:
        return
    
    raw_moves = fetch_moves(node.fen)
    # total games at this node
    total_games = sum(m["white"] + m["draws"] + m["black"] for m in raw_moves)
    if total_games == 0:
        return

    for m in raw_moves:
        count = m["white"] + m["draws"] + m["black"]
        prob  = count / total_games

        # compute reach probability under book policy
        if is_our_move(node, cfg):
            # We will play the book move—keep same reach
            child_reach = node.reach_prob
        else:
            # Opponent replies stochastically
            child_reach = node.reach_prob * prob

        # prune unlikely-to-be-reached lines
        if child_reach < cfg["min_reach_probability"] or count < cfg["min_games"]:
            continue

        # compute child FEN by pushing the UCI move
        board = chess.Board(node.fen)
        move  = chess.Move.from_uci(m["uci"])
        board.push(move)
        child_fen = board.fen()

        # attach child
        child = Node(
            fen        = child_fen,
            turn_white = not node.turn_white,
            depth      = node.depth + 1,
            parent     = node,
            games      = count,
            reach_prob = child_reach,
        )
        child._wdl = (m["white"], m["draws"], m["black"])
        node.children[m["uci"]] = (prob, child)

        # bump counters for this new node
        _bump(child.depth)

# ---------------------------
# Leaf node scoring
# ---------------------------

def score_terminal(node: Node) -> float:
    """
    +1  if the side-to-move wins
    −1  if the side-to-move loses
     0  on draw
    """
    if hasattr(node, "_wdl"):
        # we have W/D/L stats saved from expand()
        wins_white, draws, wins_black = node._wdl
    else:
        fen = node.fen
        url = f"{LICHESS_EXPLORER_URL}?fen={quote_plus(fen)}&moves=0"
        stats = _get_json(url)

        wins_white = stats.get("white", 0)
        wins_black = stats.get("black", 0)
        draws      = stats.get("draws", 0)
    
    total      = wins_white + wins_black + draws
    if total == 0:
        return 0.0  # no data

    return 0.0 if total == 0 else (wins_white - wins_black) / total

# ---------------------------
# Evaluation via backward induction
# ---------------------------

def evaluate(node: Node, cfg: dict) -> float:
    # If this is a brand‐new root search, clear out last run’s counters
    if node.depth == 0 and node.parent is None:
        _reset_counters()
        _bump(0)   # count the root itself

    # 1) Expand one ply if not already done and within depth limit
    if not node.children and node.depth < cfg["max_depth"]:
        expand(node, cfg)

    # 2) If leaf, compute its terminal score
    if not node.children:
        node.value = score_terminal(node)
        return node.value
    
    # 3) Otherwise back up values:
    if is_our_move(node, cfg):
        # We choose the child with highest value
        if cfg["book_side"] == "white":
            best_move, (_, best_child) = max(
                node.children.items(),
                key=lambda kv: evaluate(kv[1][1], cfg) 
            )
        else:  # black book → minimise
            best_move, (_, best_child) = min(
                node.children.items(),
                key=lambda kv: evaluate(kv[1][1], cfg)
            )
        node.best_move = best_move
        node.value     = best_child.value
    else:
        # Opponent is stochastic: expectation over all children
        exp_val = 0.0
        for prob, child in node.children.values():
            exp_val += prob * evaluate(child, cfg)
        node.value = exp_val

    return node.value

# ---------------------------   
# Utility helpers
# ---------------------------
    
def is_our_move(node: Node, cfg: dict) -> bool:
    """Return True when the book side gets to choose."""
    return (node.turn_white and cfg["book_side"] == "white") or \
           (not node.turn_white and cfg["book_side"] == "black")

def choose_child(children, cfg):
    """Return best (move, child) chosen by our side."""
    if cfg["book_side"] == "white":
        return max(children.items(), key=lambda kv: kv[1][1].value)
    else:  # black book → minimise
        return min(children.items(), key=lambda kv: kv[1][1].value)

# ---------------------------   
# Output utilities
# ---------------------------

def tree_to_pgn(root: Node, cfg: dict) -> str:
    """
    Build a PGN where:
      • at our moves: only the chosen move is kept;
      • at opponent moves: every child is a variation;
    Recurses depth-first through the built tree.
    """
    game = chess.pgn.Game()
    if root.fen != chess.STARTING_FEN:          # include FEN header for sub-trees
        game.headers["FEN"] = root.fen
    board = chess.Board(root.fen)

    def walk(node: Node, pgn_node: chess.pgn.ChildNode) -> None:
        if not node.children:
            return

        if is_our_move(node, cfg):
            mv = node.best_move
            _, child = node.children[mv]
            board.push(chess.Move.from_uci(mv))
            next_pgn = pgn_node.add_variation(board.peek())
            walk(child, next_pgn)
            board.pop()
        else:
            for mv, (_, child) in node.children.items():
                board.push(chess.Move.from_uci(mv))
                var = pgn_node.add_variation(board.peek())
                walk(child, var)
                board.pop()

    walk(root, game)
    exporter = chess.pgn.StringExporter(headers=True, variations=True, comments=False)
    return game.accept(exporter).strip()

def write_repertoire(root: Node, cfg: dict, *, preview: int = 500) -> None:
    """
    • Converts the tree to PGN.
    • Writes it to a timestamped file in cwd.
    • Prints the first `preview` chars (truncated with … if longer).
    """
    pgn_text = tree_to_pgn(root, cfg)
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(f"repertoire_{cfg['book_side']}_d{cfg['max_depth']}_{ts}.pgn")
    path.write_text(pgn_text + "\n", encoding="utf-8")

    clip = (pgn_text[:preview] + " …") if len(pgn_text) > preview else pgn_text
    print("\n--- Repertoire PGN (truncated) ---")
    print(clip)
    print(f"\n(full PGN written to {path.resolve()})\n")

def extract_our_lines(root: Node, cfg) -> list[str]:
    """Trace the best-response UCI moves for the book side from the root."""
    moves: list[str] = []
    node = root
    while True:
        # If it's *our* turn and we've chosen a move, record it
        if is_our_move(node, cfg) and node.best_move:
            move = node.best_move
            moves.append(move)
            _, node = node.children[move]
        # If it's the opponent's turn and the branch is deterministic
        elif node.best_move:
            _, node = node.children[node.best_move]
        else:
            break
    return moves

# ---------------------------
# CLI entry point
# ---------------------------

if __name__ == "__main__":
    import argparse, logging

    parser = argparse.ArgumentParser(description="Build best-response opening tree.")
    parser.add_argument("--config", type=Path, help="Path to JSON config file")
    parser.add_argument("--max-depth", default=DEFAULT_CONFIG["max_depth"], type=int, help="Max ply depth to expand")
    parser.add_argument("--min-reach-probability", default=DEFAULT_CONFIG["min_reach_probability"], type=float, help="Min reach probability to explore a branch")
    parser.add_argument("--min-games", default=DEFAULT_CONFIG["min_games"], type=int, help="Min games to consider a branch")
    parser.add_argument("--book-side", choices=["white", "black"], default=DEFAULT_CONFIG["book_side"], help="Side to play the book moves (default: white)")    
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg  = load_config(args.config)
    cfg["max_depth"] = args.max_depth
    cfg["min_reach_probability"] = args.min_reach_probability
    cfg["min_games"] = args.min_games
    cfg["book_side"] = args.book_side

    root = Node(fen=chess.STARTING_FEN, turn_white=True, depth=0)
    _bump(0) # bump root node count

    print()
    logging.info("Building %s book…", cfg["book_side"])
    evaluate(root, cfg)
    write_repertoire(root, cfg)
