"""Append-only CSV log of human demonstrations.

The file is the memory of the lesson. Each human keypress is written and flushed
before the next round starts, and a new session loads the rows already on disk
instead of replacing them.
"""

from __future__ import annotations

import csv
import os
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

from game import CASES


DEFAULT_DATA_PATH = Path(__file__).resolve().parent / "data" / "gameplay.csv"

COLUMNS = (
    "session_id",
    "round_id",
    "timestamp",
    "car_lane",
    "obstacle_lane",
    "player_action",
    "correct_action",
    "success",
    "reaction_time_ms",
)


@dataclass(frozen=True)
class RoundRecord:
    session_id: str
    round_id: str
    timestamp: str
    car_lane: int
    obstacle_lane: int
    player_action: int
    correct_action: int
    success: int
    reaction_time_ms: int

    def as_row(self) -> dict[str, str]:
        raw = asdict(self)
        return {key: str(raw[key]) for key in COLUMNS}


def _as_binary(name: str, value: int) -> int:
    number = int(value)
    if number not in (0, 1):
        raise ValueError(f"{name} must be 0 or 1")
    return number


class GameplayLogger:
    """Load `data/gameplay.csv` on creation and append each human round at once."""

    def __init__(self, path: Path | str = DEFAULT_DATA_PATH):
        self.path = Path(path)
        self.records: list[RoundRecord] = self._load()

    def _load(self) -> list[RoundRecord]:
        if not self.path.exists() or self.path.stat().st_size == 0:
            return []
        loaded: list[RoundRecord] = []
        with self.path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                if not row or not row.get("round_id"):
                    continue
                loaded.append(
                    RoundRecord(
                        session_id=row["session_id"],
                        round_id=row["round_id"],
                        timestamp=row["timestamp"],
                        car_lane=_as_binary("car_lane", row["car_lane"]),
                        obstacle_lane=_as_binary("obstacle_lane", row["obstacle_lane"]),
                        player_action=_as_binary("player_action", row["player_action"]),
                        correct_action=_as_binary("correct_action", row["correct_action"]),
                        success=_as_binary("success", row["success"]),
                        reaction_time_ms=int(row["reaction_time_ms"]),
                    )
                )
        return loaded

    def log_human_demonstration(
        self,
        session_id: str,
        car_lane: int,
        obstacle_lane: int,
        player_action: int,
        correct_action: int,
        success: bool,
        reaction_time_ms: int,
    ) -> RoundRecord:
        """Append one human decision and flush it to disk immediately.

        `player_action` is the key the person actually pressed. It is stored even
        when it does not match `correct_action`. Callers must not use this method
        for a network's move; AI evaluation stays in memory only.
        """
        record = RoundRecord(
            session_id=str(session_id),
            round_id=str(uuid.uuid4()),
            timestamp=datetime.now().isoformat(timespec="seconds"),
            car_lane=_as_binary("car_lane", car_lane),
            obstacle_lane=_as_binary("obstacle_lane", obstacle_lane),
            player_action=_as_binary("player_action", player_action),
            correct_action=_as_binary("correct_action", correct_action),
            success=_as_binary("success", int(bool(success))),
            reaction_time_ms=max(0, int(reaction_time_ms)),
        )
        self.path.parent.mkdir(parents=True, exist_ok=True)
        new_file = not self.path.exists() or self.path.stat().st_size == 0
        with self.path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS)
            if new_file:
                writer.writeheader()
            writer.writerow(record.as_row())
            handle.flush()
            os.fsync(handle.fileno())
        self.records.append(record)
        return record


def combination_counts(records: list[RoundRecord]) -> list[dict[str, int]]:
    """Counts for each of the four input pairs, including pairs with no rows yet."""
    totals = {(car, obstacle): {"total": 0, "success": 0} for car, obstacle in CASES}
    for record in records:
        bucket = totals[(record.car_lane, record.obstacle_lane)]
        bucket["total"] += 1
        bucket["success"] += int(record.success)
    return [
        {
            "car_lane": car,
            "obstacle_lane": obstacle,
            "total": totals[(car, obstacle)]["total"],
            "success": totals[(car, obstacle)]["success"],
        }
        for car, obstacle in CASES
    ]


def missing_successful_cases(records: list[RoundRecord]) -> list[tuple[int, int]]:
    """Input pairs that do not yet have a successful human demonstration."""
    return [
        (item["car_lane"], item["obstacle_lane"])
        for item in combination_counts(records)
        if item["success"] == 0
    ]


def reaction_summary(records: list[RoundRecord]) -> dict[str, float] | None:
    """Mean and median reaction time, in milliseconds, via NumPy."""
    if not records:
        return None
    times = np.array([record.reaction_time_ms for record in records], dtype=np.float64)
    return {
        "mean": float(times.mean()),
        "median": float(np.median(times)),
        "min": float(times.min()),
        "max": float(times.max()),
    }


def dataset_from_records(
    records: list[RoundRecord],
    successes_only: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the training matrices from logged human rounds.

    Features are exactly `[car_lane, obstacle_lane]`. The label is the action the
    human pressed, not the answer key. `correct_action` and `success` stay out of
    the feature vector: including them would hand the rule to the network.
    """
    if successes_only:
        chosen = [record for record in records if record.success]
    else:
        chosen = list(records)
    if not chosen:
        return (
            np.zeros((0, 2), dtype=np.float32),
            np.zeros((0,), dtype=np.float32),
        )
    features = np.array(
        [[record.car_lane, record.obstacle_lane] for record in chosen],
        dtype=np.float32,
    )
    labels = np.array([record.player_action for record in chosen], dtype=np.float32)
    return features, labels
