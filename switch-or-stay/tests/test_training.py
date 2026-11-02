"""Training reads demonstrations and does not invent the XNOR answers."""

import numpy as np
import torch

from data_logger import GameplayLogger, dataset_from_records
from game import CASES, correct_action
from models import ARCHITECTURES, LinearNet
from training import (
    fit,
    missing_combinations,
    predict_cases,
    train_architecture,
)


# A straight cut can score at most 3 of these 4 rows. Anything nonlinear in the
# lesson is expected to score all 4 from a complete, correct demonstration set.
EXPECTS_PERFECT = {
    "linear": False,
    "deep_linear": False,
    "relu": True,
    "deep_relu": True,
    "sigmoid": True,
    "tanh": True,
}


def _perfect_demonstrations(tmp_path):
    logger = GameplayLogger(tmp_path / "gameplay.csv")
    for car, obstacle in CASES:
        action = correct_action(car, obstacle)
        logger.log_human_demonstration(
            session_id="lesson",
            car_lane=car,
            obstacle_lane=obstacle,
            player_action=action,
            correct_action=action,
            success=True,
            reaction_time_ms=250,
        )
    reloaded = GameplayLogger(tmp_path / "gameplay.csv")
    features, labels = dataset_from_records(reloaded.records, successes_only=True)
    return features, labels


def test_predictions_come_from_the_network_not_the_answer_key():
    network = LinearNet()
    with torch.no_grad():
        for parameter in network.parameters():
            parameter.zero_()
    predictions = predict_cases(network)
    assert [round(item.probability, 6) for item in predictions] == [0.5, 0.5, 0.5, 0.5]
    assert [item.predicted_action for item in predictions] == [1, 1, 1, 1]
    assert [item.correct_action for item in predictions] == [1, 0, 0, 1]


def test_fit_rejects_extra_input_columns():
    features = np.ones((4, 3), dtype=np.float32)
    labels = np.zeros(4, dtype=np.float32)
    try:
        fit(LinearNet(), features, labels, epochs=1)
    except ValueError as error:
        assert "car_lane" in str(error)
    else:
        raise AssertionError("extra columns should be rejected")


def test_missing_combinations_are_reported():
    features = np.array([[0, 0], [0, 1], [1, 1]], dtype=np.float32)
    assert missing_combinations(features) == [(1, 0)]


def test_each_architecture_on_a_complete_correct_dataset(tmp_path):
    features, labels = _perfect_demonstrations(tmp_path)
    assert features.tolist() == [[0, 0], [0, 1], [1, 0], [1, 1]]
    assert labels.tolist() == [1, 0, 0, 1]

    for architecture in ARCHITECTURES:
        _network, history = train_architecture(architecture, features, labels)
        perfect = history.cases_correct == 4 and history.training_accuracy == 1
        assert perfect is EXPECTS_PERFECT[architecture.key], architecture.key
        assert history.n_samples == 4
        assert history.missing_combinations == []
        if architecture.key == "relu":
            assert [item.predicted_action for item in history.predictions] == [1, 0, 0, 1]
        if architecture.linear_family:
            assert [item.predicted_action for item in history.predictions] != [1, 0, 0, 1]
            assert history.training_accuracy < 1
            assert history.uncommitted()
