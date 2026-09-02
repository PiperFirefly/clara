"""gamelab.rl — tabular Q-learning self-play (the literal "games train AI" loop).

An agent learns tic-tac-toe by playing against ITSELF: one Q-table shared by
both sides, each move relabeled so the current player sees themselves as '1'.
Reward +1 win / -1 loss / 0 draw. This is the same loop as AlphaZero, reduced
to a form you can read in one file (no policy/value networks — just a dict).

After ~50k self-play games it plays near-optimally: it takes wins, blocks
threats, and draws the perfect opponent.
"""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Tuple

from .games import TicTacToe


def _flip(board: Tuple[int, ...], me: int) -> Tuple[int, ...]:
    """Relabel so `me`'s marks are 1 and the opponent's are 2 (my perspective)."""
    return tuple(1 if x == me else (2 if x != 0 else 0) for x in board)


def _win_from_board(board: Tuple[int, ...], me: int) -> int:
    """+1 if `me` has a line, -1 if opponent does, else 0."""
    opp = 3 - me
    lines = [(0, 1, 2), (3, 4, 5), (6, 7, 8), (0, 3, 6), (1, 4, 7),
             (2, 5, 8), (0, 4, 8), (2, 4, 6)]
    for a, b, c in lines:
        if board[a] == board[b] == board[c] != 0:
            return 1 if board[a] == me else -1
    return 0


class QAgent:
    def __init__(self, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 0.1):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q = defaultdict(lambda: defaultdict(float))  # state -> {action: value}

    def best(self, board: Tuple[int, ...], moves) -> int:
        vals = {m: self.q[board][m] for m in moves}
        return max(moves, key=lambda m: vals[m])

    def choose(self, board: Tuple[int, ...], moves) -> int:
        if random.random() < self.epsilon:
            return random.choice(moves)
        return self.best(board, moves)

    def train(self, games: int = 50000, decay: float = 0.9999) -> None:
        for g in range(games):
            s = TicTacToe()
            history = []
            me = 1
            while not s.is_terminal():
                board = _flip(s.board, me)  # my marks = 1
                moves = s.moves()
                a = self.choose(board, moves)
                history.append((board, a))
                s = s.apply(a)
                me = 3 - me
            # reward from each stored state's player's perspective
            r = _win_from_board(s.board, 1)  # X's final result
            # last stored player's perspective reward = r if that player == X else -r
            for i in range(len(history) - 1, -1, -1):
                board, a = history[i]
                # player who made this move: X if i even, O if i odd
                perspective = r if i % 2 == 0 else -r
                # Q-update for this player (who saw themselves as '1')
                self.q[board][a] += self.alpha * (
                    perspective - self.q[board][a])
            self.epsilon = max(0.01, self.epsilon * decay)

    def play_move(self, board: Tuple[int, ...], me: int) -> int:
        """Greedy move for mark `me` on a raw board."""
        b = _flip(board, me)
        s = TicTacToe(board)
        return self.best(b, s.moves())
