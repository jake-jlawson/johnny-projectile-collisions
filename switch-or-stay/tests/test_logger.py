"""CSV demonstrations: append, reload, and build training arrays."""

from data_logger import GameplayLogger, combination_counts, dataset_from_records


def _log(logger, car, obstacle, action, correct, success, reaction=100, session="session"):
    return logger.log_human_demonstration(
        session_id=session,
        car_lane=car,
        obstacle_lane=obstacle,
        player_action=action,
        correct_action=correct,
        success=success,
        reaction_time_ms=reaction,
    )


def test_rows_are_written_immediately_and_reloaded(tmp_path):
    path = tmp_path / "data" / "gameplay.csv"
    logger = GameplayLogger(path)
    _log(logger, 0, 0, 1, 1, True, reaction=80, session="first")

    on_disk = path.read_text(encoding="utf-8")
    assert on_disk.splitlines()[0].split(",") == [
        "session_id",
        "round_id",
        "timestamp",
        "car_lane",
        "obstacle_lane",
        "player_action",
        "correct_action",
        "success",
        "reaction_time_ms",
    ]
    assert ",0,0,1,1,1,80" in on_disk

    reopened = GameplayLogger(path)
    assert len(reopened.records) == 1
    assert reopened.records[0].session_id == "first"
    _log(reopened, 1, 0, 0, 0, True, reaction=90, session="second")

    third = GameplayLogger(path)
    assert [record.session_id for record in third.records] == ["first", "second"]
    assert path.read_text(encoding="utf-8").count("session_id") == 1


def test_a_miss_reloads_as_a_miss_and_keeps_the_human_action(tmp_path):
    path = tmp_path / "gameplay.csv"
    _log(GameplayLogger(path), 1, 1, 0, 1, False, reaction=15)
    record = GameplayLogger(path).records[0]
    assert record.success == 0
    assert record.player_action == 0
    assert record.correct_action == 1


def test_training_arrays_use_lanes_and_the_pressed_action(tmp_path):
    logger = GameplayLogger(tmp_path / "gameplay.csv")
    _log(logger, 0, 0, 0, 1, False)  # pressed stay, the rule said switch
    _log(logger, 0, 1, 0, 0, True)

    features, labels = dataset_from_records(logger.records, successes_only=True)
    assert features.shape == (1, 2)
    assert features[0].tolist() == [0.0, 1.0]
    assert labels.tolist() == [0.0]

    features, labels = dataset_from_records(logger.records, successes_only=False)
    assert features.tolist() == [[0.0, 0.0], [0.0, 1.0]]
    assert labels.tolist() == [0.0, 0.0]

    counts = combination_counts(logger.records)
    by_case = {(item["car_lane"], item["obstacle_lane"]): item for item in counts}
    assert by_case[(0, 0)]["success"] == 0
    assert by_case[(0, 1)]["success"] == 1
    assert by_case[(1, 0)]["total"] == 0
    assert by_case[(1, 1)]["total"] == 0
