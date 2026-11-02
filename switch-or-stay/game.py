"""Lane rules for the Switch or Stay game.

Each round is one row of an XNOR truth table:

    car lane 0 or 1, obstacle lane 0 or 1.
    Correct action: switch (1) when the lanes match, stay (0) when they differ.

A classifier that can only cut the plane with one straight line cannot get all
four rows right. That is the point of the lesson.
"""

from __future__ import annotations

import random
from dataclasses import dataclass


# The four input combinations, in a stable order used by the tables and plots.
CASES: tuple[tuple[int, int], ...] = ((0, 0), (0, 1), (1, 0), (1, 1))

RULE_TEXT = "Switch if the obstacle is in your lane. Stay if it is in the other lane."

SEPARATION_TEXT = (
    "The four situations are an XNOR pattern. The two 'switch' points sit on "
    "opposite corners of the square, and the two 'stay' points sit on the other "
    "diagonal. No single straight line can put both switch points on one side "
    "and both stay points on the other."
)


def lane_name(lane: int) -> str:
    return "left" if int(lane) == 0 else "right"


def action_name(action: int) -> str:
    return "stay" if int(action) == 0 else "switch"


def correct_action(car_lane: int, obstacle_lane: int) -> int:
    """Return 1 (switch) when the car and the obstacle share a lane."""
    return 1 if int(car_lane) == int(obstacle_lane) else 0


def lane_after_action(car_lane: int, action: int) -> int:
    """Lane the car occupies once the chosen action has been carried out."""
    car_lane = int(car_lane)
    action = int(action)
    if action == 0:
        return car_lane
    if action == 1:
        return 1 - car_lane
    raise ValueError("action must be 0 (stay) or 1 (switch)")


def avoids_obstacle(car_lane: int, obstacle_lane: int, action: int) -> bool:
    """True when the car is in the other lane from the obstacle after acting."""
    return lane_after_action(car_lane, action) != int(obstacle_lane)


def records_as_demonstration(mode: str) -> bool:
    """Only human play may become training data.

    AI rounds are evaluation. Mixing them into the CSV would teach the next
    model to copy an earlier model, and the student would no longer be looking
    at their own demonstrations.
    """
    return mode == "human"


@dataclass(frozen=True)
class ResolvedRound:
    car_lane: int
    obstacle_lane: int
    player_action: int
    correct_action: int
    success: bool
    final_lane: int
    record_for_training: bool


def resolve_round(car_lane: int, obstacle_lane: int, action: int, mode: str) -> ResolvedRound:
    """Apply one decision and say whether it is allowed into the training file."""
    action = int(action)
    if action not in (0, 1):
        raise ValueError("action must be 0 (stay) or 1 (switch)")
    car_lane = int(car_lane)
    obstacle_lane = int(obstacle_lane)
    if car_lane not in (0, 1) or obstacle_lane not in (0, 1):
        raise ValueError("lanes must be 0 or 1")
    expected = correct_action(car_lane, obstacle_lane)
    return ResolvedRound(
        car_lane=car_lane,
        obstacle_lane=obstacle_lane,
        player_action=action,
        correct_action=expected,
        success=action == expected,
        final_lane=lane_after_action(car_lane, action),
        record_for_training=records_as_demonstration(mode),
    )


class ScenarioDeck:
    """Deal the four situations in shuffled cycles.

    Every block of four rounds contains each input combination once, so a short
    game still covers the whole truth table. The order inside each block is random.
    """

    def __init__(self, rng: random.Random | None = None):
        self._rng = rng or random.Random()
        self._queue: list[tuple[int, int]] = []

    def next_scenario(self) -> tuple[int, int]:
        if not self._queue:
            cycle = list(CASES)
            self._rng.shuffle(cycle)
            self._queue = cycle
        return self._queue.pop()
