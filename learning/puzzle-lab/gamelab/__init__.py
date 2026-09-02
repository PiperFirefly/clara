"""gamelab — game theory: surreal numbers/nimbers + classical GT + search + RL."""
from .surreal import (
    Game, ZERO, ONE, NEG_ONE, from_int, from_dyadic,
    mex, nimber, nim_sum, nim_winning_move, subtraction_game,
)
from .games import (
    minimax, negamax_alpha_beta, best_move, TicTacToe,
    nash_examples, axelrod_tournament, self_play_demo,
)
from .mcts import mcts_best_move, Node
from .rl import QAgent

__all__ = [
    "Game", "ZERO", "ONE", "NEG_ONE", "from_int", "from_dyadic",
    "mex", "nimber", "nim_sum", "nim_winning_move", "subtraction_game",
    "minimax", "negamax_alpha_beta", "best_move", "TicTacToe",
    "nash_examples", "axelrod_tournament", "self_play_demo",
    "mcts_best_move", "Node", "QAgent",
]
