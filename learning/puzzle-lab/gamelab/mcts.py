"""gamelab.mcts — Monte Carlo Tree Search (the AlphaGo/AlphaZero search).

Selection (UCB1) -> expansion -> random rollouts -> backprop. Stronger than
alpha-beta for large/stochastic games because it doesn't need a full depth
search — it samples promising lines. Works on any GameState with
moves()/apply()/is_terminal()/winner().
"""

from __future__ import annotations

import math
import random
from typing import Optional

_UCB_C = 1.414  # sqrt(2), the standard exploration constant


class Node:
    __slots__ = ("state", "parent", "move", "wins", "visits", "children", "untried")

    def __init__(self, state, parent=None, move=None):
        self.state = state
        self.parent = parent
        self.move = move
        self.wins = 0.0
        self.visits = 0
        self.children = []
        self.untried = list(state.moves())

    def ucb1(self) -> float:
        if self.visits == 0:
            return float("inf")
        # negate: child.wins is from the child's (opponent's) perspective,
        # so from the parent's view the exploit value is -wins/visits.
        exploit = -self.wins / self.visits
        explore = _UCB_C * math.sqrt(math.log(self.parent.visits) / self.visits)
        return exploit + explore


def _rollout(state) -> int:
    """Random play to a terminal; return +1/-1/0 from the START player's view."""
    s = state
    depth = 0
    while not s.is_terminal():
        s = s.apply(random.choice(s.moves()))
        depth += 1
    w = s.winner()  # relative to s.turn (the player to move at s)
    return -w if depth % 2 == 1 else w


def _backprop(node: Node, result: int) -> None:
    while node is not None:
        node.visits += 1
        node.wins += result
        result = -result  # parent is the opponent's turn
        node = node.parent


def mcts_best_move(state, iterations: int = 2000, c: float = _UCB_C,
                   rng: Optional[random.Random] = None) -> object:
    """Return the move with the most visits after `iterations` of MCTS."""
    global _UCB_C
    _UCB_C = c
    rng = rng or random.Random()
    root = Node(state)
    for _ in range(iterations):
        node = root
        # select
        while not node.state.is_terminal() and not node.untried and node.children:
            node = max(node.children, key=Node.ucb1)
        # expand
        if not node.state.is_terminal() and node.untried:
            mv = node.untried.pop(rng.randrange(len(node.untried)))
            child = Node(node.state.apply(mv), node, mv)
            node.children.append(child)
            node = child
        # rollout + backprop
        _backprop(node, _rollout(node.state))
    return max(root.children, key=lambda n: n.visits).move
