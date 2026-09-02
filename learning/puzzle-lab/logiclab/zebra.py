"""logiclab.zebra — Einstein's Zebra puzzle, as a reusable Z3 logic-grid template.

The canonical 5-house riddle (5 colors, nationalities, drinks, smokes, pets;
each category has exactly one of each). Solves it and reports who drinks water
and who owns the zebra. Written as a template: swap the domains + clues to
encode any similar logic-grid puzzle.
"""

from __future__ import annotations

import z3


def zebra():
    """Solve Einstein's riddle; return (water_drinker, zebra_owner, mapping)."""
    n = 5
    # domains
    colors = ["red", "green", "ivory", "yellow", "blue"]
    nations = ["englishman", "spaniard", "ukrainian", "norwegian", "japanese"]
    drinks = ["coffee", "tea", "milk", "orange_juice", "water"]
    smokes = ["old_gold", "kools", "chesterfields", "lucky_strike", "parliaments"]
    pets = ["dog", "snails", "fox", "horse", "zebra"]

    def var(name, domain):
        return {d: z3.Int(f"{name}_{d}") for d in domain}

    color, nation = var("color", colors), var("nation", nations)
    drink, smoke, pet = var("drink", drinks), var("smoke", smokes), var("pet", pets)

    s = z3.Solver()

    # each house (1..n) assigned exactly one value per category (distinct positions)
    def distinct(assign):
        s.add(z3.Distinct(*assign.values()))
        for d, v in assign.items():
            s.add(v >= 1, v <= n)

    for a in (color, nation, drink, smoke, pet):
        distinct(a)

    def same(*entries):  # these sit in the same house
        pairs = list(entries)
        for i in range(len(pairs) - 1):
            s.add(pairs[i] == pairs[i + 1])

    def adjacent(a, b):
        s.add(z3.Or(a == b + 1, a == b - 1))

    # clues
    same(nation["englishman"], color["red"])
    same(nation["spaniard"], pet["dog"])
    same(drink["coffee"], color["green"])
    same(nation["ukrainian"], drink["tea"])
    s.add(color["green"] == color["ivory"] + 1)
    same(smoke["old_gold"], pet["snails"])
    same(smoke["kools"], color["yellow"])
    s.add(drink["milk"] == 3)
    s.add(nation["norwegian"] == 1)
    adjacent(smoke["chesterfields"], pet["fox"])
    adjacent(smoke["kools"], pet["horse"])
    same(smoke["lucky_strike"], drink["orange_juice"])
    same(nation["japanese"], smoke["parliaments"])
    adjacent(nation["norwegian"], color["blue"])

    assert s.check() == z3.sat, "zebra puzzle unsat (check the clues)"
    m = s.model()

    def pos(assign, val):
        return m.evaluate(assign[val]).as_long()

    water_house = pos(drink, "water")
    zebra_house = pos(pet, "zebra")
    water_drinker = next(nat for nat in nations if pos(nation, nat) == water_house)
    zebra_owner = next(nat for nat in nations if pos(nation, nat) == zebra_house)
    return water_drinker, zebra_owner
