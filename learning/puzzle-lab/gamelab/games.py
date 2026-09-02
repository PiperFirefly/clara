"""gamelab.games — classical game theory + self-play.

    nash equilibria     -> nashpy (matching pennies, prisoner's dilemma)
    repeated games      -> axelrod (iterated prisoner's dilemma tournaments)
    self-play skeleton  -> minimax / negamax / alpha-beta over a generic
                           zero-sum game, with tic-tac-toe as the demo.

The self-play loop is the same shape AlphaZero uses: a game state class, a
move generator, and a policy learned by playing against itself. This is the
"games to train AI" bit, reduced to a form you can read in one file.
"""

from __future__ import annotations

from typing import List, Optional, Tuple


# ---------------------------------------------------------------------------
# Nash equilibria (nashpy)
# ---------------------------------------------------------------------------

def nash_examples() -> dict:
    """Compute Nash equilibria for the two canonical 2x2 games.

    Returns {game_name: [equilibria]}, where each equilibrium is a
    (row_strategy, col_strategy) tuple of probability vectors.
    """
    import nashpy as nash

    # Matching pennies: no pure NE; unique mixed NE (1/2, 1/2) each.
    mp = nash.Game([[1, -1], [-1, 1]], [[-1, 1], [1, -1]])
    # Prisoner's dilemma: (defect, defect) is the unique NE.
    pd = nash.Game([[3, 0], [5, 1]], [[3, 5], [0, 1]])
    return {
        "matching_pennies": list(mp.support_enumeration()),
        "prisoners_dilemma": list(pd.support_enumeration()),
    }


# ---------------------------------------------------------------------------
# Iterated prisoner's dilemma (axelrod)
# ---------------------------------------------------------------------------

def axelrod_tournament(turns: int = 100, repetitions: int = 5) -> dict:
    """Run a small IPD tournament; return {player_name: mean normalised score}
    in ranked order."""
    import axelrod as axl

    players = [axl.TitForTat(), axl.Grudger(), axl.Alternator(),
               axl.Defector(), axl.Cooperator()]
    tournament = axl.Tournament(players, turns=turns, repetitions=repetitions)
    results = tournament.play()
    ranking = {}
    for i in results.ranking:
        name = str(results.players[i])
        scores = results.normalised_scores[i]
        ranking[name] = round(sum(scores) / len(scores), 3)
    return ranking


# ---------------------------------------------------------------------------
# Self-play: minimax / negamax / alpha-beta
# ---------------------------------------------------------------------------

class GameState:
    """Interface for a two-player zero-sum game."""
    def moves(self) -> List:            # legal moves for the current player
        raise NotImplementedError
    def apply(self, move) -> "GameState":
        raise NotImplementedError
    def is_terminal(self) -> bool:
        raise NotImplementedError
    def winner(self) -> int:            # 1 = player to move wins, -1 = loses, 0 = draw
        raise NotImplementedError


def minimax(state: GameState, depth: int = 0) -> int:
    """Return best score for the player to move (+1 win, -1 loss, 0 draw)."""
    if state.is_terminal():
        return state.winner()
    best = -2
    for mv in state.moves():
        val = -minimax(state.apply(mv), depth + 1)  # negamax trick
        best = max(best, val)
    return best


def negamax_alpha_beta(state: GameState, alpha: float = -2, beta: float = 2,
                       depth: int = 0) -> int:
    """Alpha-beta pruned negamax; returns best score for player to move."""
    if state.is_terminal():
        return state.winner()
    best = -2
    for mv in state.moves():
        val = -negamax_alpha_beta(state.apply(mv), -beta, -alpha, depth + 1)
        best = max(best, val)
        alpha = max(alpha, val)
        if alpha >= beta:
            break
    return best


def best_move(state: GameState) -> Tuple[object, int]:
    """Return (move, resulting_score) for the player to move."""
    scored = [(mv, -negamax_alpha_beta(state.apply(mv))) for mv in state.moves()]
    return max(scored, key=lambda x: x[1])


class TicTacToe(GameState):
    """3x3 tic-tac-toe; board is a 9-tuple, 0=empty, 1=X, 2=O."""

    def __init__(self, board: Tuple[int, ...] = (0,) * 9, turn: int = 1):
        self.board = tuple(board)
        self.turn = turn  # 1 = X, 2 = O

    def moves(self) -> List[int]:
        return [i for i, v in enumerate(self.board) if v == 0]

    def apply(self, move: int) -> "TicTacToe":
        b = list(self.board)
        b[move] = self.turn
        return TicTacToe(tuple(b), 3 - self.turn)

    def is_terminal(self) -> bool:
        return self.winner() != 0 or all(self.board)

    def winner(self) -> int:
        """From current player's perspective: +1/-1/0. 0 = ongoing or draw."""
        lines = [(0,1,2),(3,4,5),(6,7,8),(0,3,6),(1,4,7),(2,5,8),(0,4,8),(2,4,6)]
        b = self.board
        for a, c, d in lines:
            if b[a] == b[c] == b[d] != 0:
                winner = b[a]
                return 1 if winner == self.turn else -1
        return 0


def self_play_demo() -> None:
    """Play tic-tac-toe: both sides use alpha-beta; expect a draw."""
    s = TicTacToe()
    history = []
    while not s.is_terminal():
        mv, score = best_move(s)
        history.append((mv, score))
        s = s.apply(mv)
    print("tic-tac-toe self-play:", "draw" if s.winner() == 0 else "win",
          "| moves:", history)
