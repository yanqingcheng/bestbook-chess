# opening_book.py – backward-induction chess opening book
# ---------------------------------------------------------------

from __future__ import annotations

import json
from dataclasses import dataclass, field
from collections import OrderedDict
from pathlib import Path

import chess  # python-chess for FEN handling and move application

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
# Lichess API stub
# ---------------------------

LICHESS_EXPLORER_URL = "https://explorer.lichess.ovh/masters"

def fetch_moves(fen: str) -> list[dict[str, int | str]]:
    """Query Lichess Explorer API for moves from a FEN.

    Returns a list of dicts with keys: san, uci, white, draws, black, averageRating.
    """
    # TODO: implement HTTP call + simple on-disk cache
    raise NotImplementedError

# ---------------------------
# Tree construction
# ---------------------------

def expand(node: Node, cfg: dict) -> None:
    """Grow one ply beneath `node`, pruning on reach_prob."""
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
        if node.turn_white:
            # White will play the book move—keep same reach
            child_reach = node.reach_prob
        else:
            # Black replies stochastically
            child_reach = node.reach_prob * prob

        # prune unlikely-to-be-reached lines
        if child_reach < cfg["min_reach_probability"] or count < cfg["min_games"]:
            continue

        # compute child FEN by pushing the UCI move
        board = chess.Board(node.fen if node.fen != "startpos" else None)
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
        node.children[m["uci"]] = (prob, child)

# ---------------------------
# Evaluation via backward induction
# ---------------------------

def score_terminal(node: Node) -> float:
    """Compute terminal expectation: +1*P(win) + 0*P(draw) -1*P(loss)."""
    # TODO: fetch and compute outcome stats for node.fen
    raise NotImplementedError

def evaluate(node: Nodbest_prob, cfg: dict) -> float:
    # 1) Expand one ply if not already done and within depth limit
    if not node.children and node.depth < cfg["max_depth"]:
        expand(node, cfg)

    # 2) If leaf, compute its terminal score
    if not node.children:
        node.value = score_terminal(node)
        return node.value

    # 3) Otherwise back up values:
    if node.turn_white:
        # White chooses the child with highest value
        best_move, (best_prob, best_child) = max(
            node.children.items(),
            key=lambda item: evaluate(item[1][1], cfg)
        )
        node.best_move = best_move
        node.value     = best_child.value
    else:
        # Black is stochastic: expectation over all children
        exp_val = 0.0
        for prob, child in node.children.values():
            exp_val += prob * evaluate(child, cfg)
        node.value = exp_val

    return node.value

# ---------------------------
# Output utilities
# ---------------------------

def extract_white_lines(root: Node) -> list[str]:
    """Trace the best-response UCI moves for White from the root."""
    line: list[str] = []
    node = root
    while node.best_move:
        move = node.best_move
        line.append(move)
        _, child = node.children[move]
        node = child
    return line

# ---------------------------
# CLI entry point
# ---------------------------

if __name__ == "__main__":
    import argparse, logging

    parser = argparse.ArgumentParser(description="Build best-response opening tree.")
    parser.add_argument("--config", type=Path, help="Path to JSON config file")
    parser.add_argument("--start-fen", default="startpos")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    cfg  = load_config(args.config)
    root = Node(fen=args.start_fen, turn_white=True, depth=0)

    logging.info("Building tree…")
    evaluate(root, cfg)
    logging.info("Main line: %s", extract_white_lines(root))
