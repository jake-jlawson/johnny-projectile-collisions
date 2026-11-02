"""Rules of the lane game, without opening a window."""

import random

from game import CASES, ScenarioDeck, avoids_obstacle, correct_action, resolve_round


def test_truth_table_is_xnor():
    assert [(car, obstacle, correct_action(car, obstacle)) for car, obstacle in CASES] == [
        (0, 0, 1),
        (0, 1, 0),
        (1, 0, 0),
        (1, 1, 1),
    ]


def test_correct_action_avoids_the_obstacle_and_the_wrong_action_does_not():
    for car, obstacle in CASES:
        expected = correct_action(car, obstacle)
        assert avoids_obstacle(car, obstacle, expected)
        assert not avoids_obstacle(car, obstacle, 1 - expected)


def test_resolve_round_records_only_human_play():
    human = resolve_round(0, 0, 1, "human")
    linear = resolve_round(0, 0, 1, "linear")
    relu = resolve_round(1, 0, 1, "relu")

    assert human.success
    assert human.record_for_training
    assert human.final_lane == 1

    assert linear.success
    assert not linear.record_for_training

    assert not relu.success
    assert relu.correct_action == 0
    assert not relu.record_for_training


def test_shuffled_cycles_cover_every_situation_equally():
    deck = ScenarioDeck(random.Random(5))
    draws = [deck.next_scenario() for _ in range(40)]
    for case in CASES:
        assert draws.count(case) == 10
    for start in range(0, 40, 4):
        assert sorted(draws[start : start + 4]) == sorted(CASES)
