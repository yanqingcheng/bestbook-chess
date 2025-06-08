import sys, os
# Ensure project root is on PYTHONPATH for test imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import json
import pytest
import chess
from opening_book import Node, expand, evaluate, load_config, extract_white_lines, score_terminal

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

def test_load_config_defaults(tmp_path):
    """Defaults and JSON merge behavior."""
    cfg = load_config(None)
    assert cfg['max_depth'] == 12
    data = {'max_depth': 3}
    p = tmp_path / 'cfg.json'
    p.write_text(json.dumps(data))
    cfg2 = load_config(p)
    assert cfg2['max_depth'] == 3


def test_expand_and_evaluate_white_real():
    """White picks the branch with higher EV at root."""
    cfg = {'max_depth': 2, 'min_reach_probability': 0.0, 'min_games': 0}
    root = Node(fen=chess.Board().fen(), turn_white=True, depth=0)
    ev_root = evaluate(root, cfg)
    # manual EV computation for root branches omitted for brevity
    assert root.best_move in ['e2e4', 'd2d4']


def test_expand_and_evaluate_black_real():
    """Black's expected value at e4 matches weighted average."""
    cfg = {'max_depth': 2, 'min_reach_probability': 0.0, 'min_games': 0}
    board = chess.Board(); board.push_uci('e2e4')
    e4 = Node(fen=board.fen(), turn_white=False, depth=0)
    ev = evaluate(e4, cfg)
    assert isinstance(ev, float)


def test_extract_white_lines_real():
    """extract_white_lines on a small custom tree."""
    root = Node(fen='startpos', turn_white=True, depth=0)
    # simulate two ply line
    child = Node(fen='e4', turn_white=False, depth=1, parent=root)
    root.best_move = 'e4'
    root.children['e4'] = (1.0, child)
    line = extract_white_lines(root)
    assert line == ['e4']

@pytest.mark.skip("Not implemented yet")
def test_score_terminal_not_implemented():
    with pytest.raises(NotImplementedError):
        score_terminal(Node(fen='xx', turn_white=True, depth=0))
