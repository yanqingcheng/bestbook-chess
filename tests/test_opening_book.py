# Unit tests for opening_book.py
# ----------------------------------------------------------------------
import sys, os
# Ensure project root is on PYTHONPATH for test imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pytest
import chess

from opening_book import (
    Node,
    evaluate,
    load_config,
    tree_to_pgn,
    is_our_move,
)

# ----------------------------------------------------------------------
# Dummy API implementations for deterministic testing with real moves
# ----------------------------------------------------------------------

def dummy_fetch_moves(fen: str) -> list[dict]:
    """
    3-ply tree with explicit game-counts.

       startpos  200 g
       │
       ├─ 1. e4            100 g   (58 W  5 D 37 B)
       │   ├─ … c5          60 g   (54 W  3 D 3 B)
       │   │    └─ 2.Nf3    60 g   (54 W  3 D 3 B)   leaf
       │   └─ … e5          40 g   ( 4 W  2 D 34 B)
       │        └─ 2.Nf3    40 g   ( 4 W  2 D 34 B)  leaf
       │
       └─ 1. d4            100 g   (34 W 10 D 56 B)
           ├─ … d5          60 g   ( 6 W  6 D 48 B)
           │    └─ 2.c4     60 g   ( 6 W  6 D 48 B)  leaf
           └─ … Nf6         40 g   (28 W  4 D  8 B)
                └─ 2.c4     40 g   (28 W  4 D  8 B)  leaf
    """
    def after(ucis):
        b = chess.Board()
        for u in ucis:
            b.push_uci(u)
        return b.fen()

    # First, all FEN handles
    start = chess.STARTING_FEN
    e4, d4 = after(["e2e4"]), after(["d2d4"])
    e4c5, e4e5 = after(["e2e4", "c7c5"]), after(["e2e4", "e7e5"])
    d4d5, d4Nf6 = after(["d2d4", "d7d5"]), after(["d2d4", "g8f6"])

    return {
        start: [
            dict(uci="e2e4", white=58, draws=5,  black=37),
            dict(uci="d2d4", white=34, draws=10, black=56),
        ],
        e4: [
            dict(uci="c7c5", white=54, draws=3, black=3),
            dict(uci="e7e5", white=4,  draws=2, black=34),
        ],
        d4: [
            dict(uci="d7d5", white=6,  draws=6, black=48),
            dict(uci="g8f6", white=28, draws=4, black=8),
        ],
        # leaf ply (only one reply each)
        e4c5: [dict(uci="g1f3", white=54, draws=3, black=3)],
        e4e5: [dict(uci="g1f3", white=4,  draws=2, black=34)],
        d4d5: [dict(uci="c2c4", white=6,  draws=6, black=48)],
        d4Nf6: [dict(uci="c2c4", white=28, draws=4, black=8)],
    }.get(fen, [])


def dummy_score_terminal(node: Node) -> float:
    """
    Return (P_white – P_black) / total  ⇒  + = good for White.
    Leaf expectations:
        e4 c5 Nf3 :  (54-3)/60  = +0.85
        e4 e5 Nf3 :   (4-34)/40 = –0.75
        d4 d5 c4  :   (6-48)/60 = –0.70
        d4 Nf6 c4 :  (28-8)/40  = +0.50
    """
    def after(ucis):
        b = chess.Board()
        for u in ucis:
            b.push_uci(u)
        return b.fen()

    table = {
        after(["e2e4", "c7c5", "g1f3"]): (54, 3, 3),
        after(["e2e4", "e7e5", "g1f3"]): (4, 2, 34),
        after(["d2d4", "d7d5", "c2c4"]): (6, 6, 48),
        after(["d2d4", "g8f6", "c2c4"]): (28, 4, 8),
    }
    w, d, l = table.get(node.fen, (0, 0, 0))
    total = w + d + l
    return (w - l) / total if total else 0.0

# ----------------------------------------------------------------------
# Fixture: patch the real API calls
# ----------------------------------------------------------------------

@pytest.fixture(autouse=True)
def patch_api(monkeypatch):
    import opening_book
    monkeypatch.setattr(opening_book, 'fetch_moves', dummy_fetch_moves)
    monkeypatch.setattr(opening_book, 'score_terminal', dummy_score_terminal)
    yield

# ----------------------------------------------------------------------
# Tests
# ----------------------------------------------------------------------

def base_cfg(depth=2, side="white"):
    return dict(
        max_depth=depth,
        min_reach_probability=0.0,
        min_games=0,
        book_side=side,
    )

def test_load_config_defaults(tmp_path):
    """Defaults and JSON merge behavior."""
    cfg = load_config(None)
    assert cfg['max_depth'] == 12
    data = {'max_depth': 3}
    p = tmp_path / 'cfg.json'
    p.write_text(json.dumps(data))
    cfg2 = load_config(p)
    assert cfg2['max_depth'] == 3


def test_white_root_prefers_e4():
    """Given our skewed leaf values, White must choose 1.e4."""
    cfg_white = base_cfg(depth=3, side="white")
    root = Node(fen=chess.STARTING_FEN, turn_white=True, depth=0)
    evaluate(root, cfg_white)
    assert root.best_move == "e2e4"

def test_black_book_prefers_e5_and_d5():
    cfg_b = base_cfg(depth=3, side="black")
    root = Node(fen=chess.STARTING_FEN, turn_white=True, depth=0)
    evaluate(root, cfg_b)

    e4_reply = root.children["e2e4"][1].best_move
    d4_reply = root.children["d2d4"][1].best_move
    assert e4_reply == "e7e5"
    assert d4_reply == "d7d5"

def test_is_our_move_logic():
    n_white = Node(fen=chess.STARTING_FEN, turn_white=True, depth=0)
    n_black = Node(fen=chess.STARTING_FEN, turn_white=False, depth=0)
    assert is_our_move(n_white, base_cfg(side="white"))
    assert not is_our_move(n_white, base_cfg(side="black"))
    assert is_our_move(n_black, base_cfg(side="black"))

def test_tree_to_pgn_white():
    root = Node(fen=chess.STARTING_FEN, turn_white=True, depth=0)
    evaluate(root, base_cfg(side="white"))
    pgn = tree_to_pgn(root, base_cfg(side="white"))
    # Our repertoire move (e4) must appear as the first token.
    assert "1. e4" in pgn

def test_tree_to_pgn_black_depth3():
    cfg_black = base_cfg(depth=3, side="black")
    root = Node(fen=chess.STARTING_FEN, turn_white=True, depth=0) 
    evaluate(root, cfg_black)
    pgn = tree_to_pgn(root, cfg_black)

    # 1.e4 branch shows both replies for Black and the sole White counter
    assert "1. e4" in pgn and "1... e5" in pgn and "1... c5" in pgn

    # 1.d4 branch variations present
    assert "1. d4" in pgn and "1... d5" in pgn and "1... Nf6" in pgn

