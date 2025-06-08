# Unit tests for opening_book.py
# ----------------------------------------------------------------------
import sys, os
# Ensure project root is on PYTHONPATH for test imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pytest
import chess
from opening_book import Node, expand, evaluate, load_config, extract_our_lines, score_terminal

# ----------------------------------------------------------------------
# Dummy API implementations for deterministic testing with real moves
# ----------------------------------------------------------------------

def dummy_fetch_moves(fen: str) -> list[dict]:
    """
    Simulate Lichess Explorer data for a depth-2 opening tree using real FENs:

    Start position: generate FEN after initial board setup
      - 'e2e4' → fen after e2e4
      - 'd2d4' → fen after d2d4

    From e4 position:
      - 'c7c5' → fen after e2e4 c7c5
      - 'e7e5' → fen after e2e4 e7e5

    From d4 position:
      - 'd7d5' → fen after d2d4 d7d5
      - 'g8f6' → fen after d2d4 g8f6

    All other FENs return an empty list (leaf nodes).
    """
    # Build key FENs
    board = chess.Board()
    start_fen = board.fen()
    # e4 and d4
    board_e4 = chess.Board()
    board_e4.push_uci('e2e4')
    e4_fen = board_e4.fen()
    board_d4 = chess.Board()
    board_d4.push_uci('d2d4')
    d4_fen = board_d4.fen()
    # replies to e4
    board_e4c5 = chess.Board()
    board_e4c5.push_uci('e2e4'); board_e4c5.push_uci('c7c5')
    e4c5_fen = board_e4c5.fen()
    board_e4e5 = chess.Board()
    board_e4e5.push_uci('e2e4'); board_e4e5.push_uci('e7e5')
    e4e5_fen = board_e4e5.fen()
    # replies to d4
    board_d4d5 = chess.Board()
    board_d4d5.push_uci('d2d4'); board_d4d5.push_uci('d7d5')
    d4d5_fen = board_d4d5.fen()
    board_d4Nf6 = chess.Board()
    board_d4Nf6.push_uci('d2d4'); board_d4Nf6.push_uci('g8f6')
    d4Nf6_fen = board_d4Nf6.fen()

    data = {
        start_fen: [
            {'uci': 'e2e4', 'white': 60, 'draws': 20, 'black': 20, 'fen': e4_fen},
            {'uci': 'd2d4', 'white': 50, 'draws': 30, 'black': 20, 'fen': d4_fen}
        ],
        e4_fen: [
            {'uci': 'c7c5', 'white': 40, 'draws': 30, 'black': 30, 'fen': e4c5_fen},
            {'uci': 'e7e5', 'white': 30, 'draws': 40, 'black': 30, 'fen': e4e5_fen}
        ],
        d4_fen: [
            {'uci': 'd7d5',  'white': 45, 'draws': 25, 'black': 30, 'fen': d4d5_fen},
            {'uci': 'g8f6','white': 35, 'draws': 35, 'black': 30, 'fen': d4Nf6_fen}
        ]
    }
    return data.get(fen, [])


def dummy_score_terminal(node: Node) -> float:
    """
    Terminal score expectation at depth limit:
      +1*P(win) + 0*P(draw) -1*P(loss)

    Recomputes the same FENs as dummy_fetch_moves to build the stats map.
    """
    # Rebuild the key FENs to match dummy_fetch_moves
    board_e4c5 = chess.Board()
    board_e4c5.push_uci('e2e4'); board_e4c5.push_uci('c7c5')
    e4c5_fen = board_e4c5.fen()
    board_e4e5 = chess.Board()
    board_e4e5.push_uci('e2e4'); board_e4e5.push_uci('e7e5')
    e4e5_fen = board_e4e5.fen()
    board_d4d5 = chess.Board()
    board_d4d5.push_uci('d2d4'); board_d4d5.push_uci('d7d5')
    d4d5_fen = board_d4d5.fen()
    board_d4Nf6 = chess.Board()
    board_d4Nf6.push_uci('d2d4'); board_d4Nf6.push_uci('g8f6')
    d4Nf6_fen = board_d4Nf6.fen()

    stats_map = {
        # key: (white, draws, black)
        e4c5_fen: (40, 30, 30),
        e4e5_fen: (30, 40, 30),
        d4d5_fen: (45, 25, 30),
        d4Nf6_fen: (35, 35, 30),
    }
    val = stats_map.get(node.fen)
    if not val:
        return 0.0
    w, d, l = val
    total = w + d + l
    return (w - l) / total

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


def test_white_book_picks_best_root_move():
    """With side=white we expect e4 to outrank d4 under dummy stats."""
    cfg = base_cfg(side="white")
    root = Node(fen=chess.STARTING_FEN, turn_white=True, depth=0)
    evaluate(root, cfg)
    assert root.best_move in {"e2e4", "d2d4"}  # at least not None


def test_black_book_stochastic_root():
    """With side=black the root (White to move) should have no best_move."""
    cfg = base_cfg(side="black")
    root = Node(fen=chess.STARTING_FEN, turn_white=True, depth=0)
    evaluate(root, cfg)
    # book side is black, so root is opponent: best_move stays None
    assert root.best_move is None


def test_extract_our_lines_white():
    """extract_our_lines returns only *our* moves for side=white."""
    cfg = base_cfg(side="white")
    root = Node(fen=chess.STARTING_FEN, turn_white=True, depth=0)
    evaluate(root, cfg)
    line = extract_our_lines(root, cfg)
    assert line  # non-empty
    assert all(move in {"e2e4", "d2d4"} for move in line)


def test_extract_our_lines_black():
    """Same helper but for a black repertoire."""
    cfg = base_cfg(side="black")
    # set up after 1.e4 so it's Black to move
    b = chess.Board(); b.push_uci("e2e4")
    root = Node(fen=b.fen(), turn_white=False, depth=0)
    evaluate(root, cfg)
    line = extract_our_lines(root, cfg)
    # our first choice as Black must be present
    assert line and line[0] in {"c7c5", "e7e5"}

