"""Train a network to imitate the human demonstrations.

The optimiser never sees a hand-written XNOR label. It sees rows from the CSV:
inputs [car_lane, obstacle_lane], target player_action.

Binary cross-entropy with logits is the loss. PyTorch applies the sigmoid inside
the loss, which is numerically stable. We still draw a sigmoid ourselves when we
want a probability. Putting a sigmoid in the model and then using this loss would
squash the score twice, so the model modules return raw scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from torch import nn

from game import CASES, correct_action
from models import Architecture


SEED = 1
EPOCHS = 800
LEARNING_RATE = 0.1


@dataclass
class CasePrediction:
    car_lane: int
    obstacle_lane: int
    probability: float
    predicted_action: int
    correct_action: int

    @property
    def matches_rule(self) -> bool:
        return self.predicted_action == self.correct_action


@dataclass
class TrainHistory:
    losses: list[float] = field(default_factory=list)
    accuracies: list[float] = field(default_factory=list)
    training_loss: float = 0.0
    training_accuracy: float = 0.0
    predictions: list[CasePrediction] = field(default_factory=list)
    missing_combinations: list[tuple[int, int]] = field(default_factory=list)
    conflicting_inputs: list[tuple[int, int]] = field(default_factory=list)
    n_samples: int = 0

    @property
    def cases_correct(self) -> int:
        return sum(1 for prediction in self.predictions if prediction.matches_rule)

    def uncommitted(self) -> bool:
        """True when every corner is still about 50%: the network did not pick a side."""
        return bool(self.predictions) and all(
            abs(prediction.probability - 0.5) < 0.05 for prediction in self.predictions
        )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def missing_combinations(features: np.ndarray) -> list[tuple[int, int]]:
    if len(features) == 0:
        return list(CASES)
    present = {(int(row[0]), int(row[1])) for row in features}
    return [case for case in CASES if case not in present]


def conflicting_inputs(features: np.ndarray, labels: np.ndarray) -> list[tuple[int, int]]:
    """Input pairs whose demonstrations do not all share one action."""
    seen: dict[tuple[int, int], int] = {}
    conflicts: list[tuple[int, int]] = []
    for row, label in zip(features, labels):
        key = (int(row[0]), int(row[1]))
        action = int(label)
        if key in seen and seen[key] != action and key not in conflicts:
            conflicts.append(key)
        seen[key] = action
    return conflicts


def _check_training_arrays(features: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32).reshape(-1)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError(
            "Neural network inputs must be exactly [car_lane, obstacle_lane]. "
            "Do not add correct_action or success as inputs."
        )
    if features.shape[0] != labels.shape[0]:
        raise ValueError("Each demonstration needs one label.")
    if features.shape[0] == 0:
        raise ValueError("There are no demonstrations to train on.")
    if not np.isin(labels, (0.0, 1.0)).all():
        raise ValueError("Labels must be the recorded actions, 0 (stay) or 1 (switch).")
    return features, labels


def switch_probabilities(model: nn.Module, features: np.ndarray) -> np.ndarray:
    """Probability of switching for each row. Does not rewrite the model's answer."""
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[1] != 2:
        raise ValueError("Features must be exactly [car_lane, obstacle_lane].")
    model.cpu()
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(features)
        logits = model(inputs).view(-1)
        return torch.sigmoid(logits).cpu().numpy()


def switch_probability(model: nn.Module, car_lane: int, obstacle_lane: int) -> float:
    features = np.array([[car_lane, obstacle_lane]], dtype=np.float32)
    return float(switch_probabilities(model, features)[0])


def action_from_probability(probability: float) -> int:
    """Switch when the model gives switching a probability of at least one half."""
    return 1 if float(probability) >= 0.5 else 0


def predict_cases(model: nn.Module) -> list[CasePrediction]:
    """Run the model on all four situations and keep whatever it actually outputs."""
    features = np.array(CASES, dtype=np.float32)
    probabilities = switch_probabilities(model, features)
    predictions: list[CasePrediction] = []
    for (car_lane, obstacle_lane), probability in zip(CASES, probabilities):
        predicted = action_from_probability(float(probability))
        predictions.append(
            CasePrediction(
                car_lane=car_lane,
                obstacle_lane=obstacle_lane,
                probability=float(probability),
                predicted_action=predicted,
                correct_action=correct_action(car_lane, obstacle_lane),
            )
        )
    return predictions


def fit(
    model: nn.Module,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    on_epoch=None,
    report_every: int = 100,
) -> TrainHistory:
    """Full-batch Adam on the demonstrations currently in memory.

    `on_epoch`, when given, is called as on_epoch(epoch_index, epochs, loss, accuracy).
    `report_every` is how many epochs pass between those calls. Use 1 to update a
    live chart on every epoch.
    """
    features, labels = _check_training_arrays(features, labels)
    model.cpu()
    inputs = torch.from_numpy(features)
    targets = torch.from_numpy(labels)
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    # Mean binary cross-entropy on raw scores. The sigmoid lives inside this loss.
    loss_fn = nn.BCEWithLogitsLoss()
    history = TrainHistory(
        missing_combinations=missing_combinations(features),
        conflicting_inputs=conflicting_inputs(features, labels),
        n_samples=int(labels.shape[0]),
    )

    model.train()
    for epoch in range(epochs):
        optimiser.zero_grad()
        logits = model(inputs).view(-1)
        loss = loss_fn(logits, targets)
        loss.backward()
        optimiser.step()

        with torch.no_grad():
            predicted = (logits.detach() >= 0).float()
            accuracy = float((predicted == targets).float().mean())
            loss_value = float(loss.detach())
        history.losses.append(loss_value)
        history.accuracies.append(accuracy)
        if on_epoch is not None and (epoch % report_every == 0 or epoch == epochs - 1):
            on_epoch(epoch, epochs, loss_value, accuracy)

    history.training_loss = history.losses[-1]
    history.training_accuracy = _dataset_accuracy(model, inputs, targets)
    history.predictions = predict_cases(model)
    return history


def _dataset_accuracy(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        predicted = (model(inputs).view(-1) >= 0).float()
        return float((predicted == targets).float().mean())


def train_architecture(
    architecture: Architecture,
    features: np.ndarray,
    labels: np.ndarray,
    epochs: int = EPOCHS,
    learning_rate: float = LEARNING_RATE,
    seed: int = SEED,
    on_epoch=None,
) -> tuple[nn.Module, TrainHistory]:
    """Build a fresh network from `seed`, then fit it to these demonstrations.

    Seeding happens immediately before the weights are created, so two networks
    with the same layers start from the same draw. The comparison is then about
    the activation. The seed is fixed so a lesson can be repeated.

    On a balanced set of XNOR demonstrations, the loss minimum of a linear
    network is the constant one-half predictor. Training is allowed to find
    that. Nothing here pushes a model onto a boundary it did not learn.
    """
    set_seed(seed)
    model = architecture.factory()
    history = fit(
        model,
        features,
        labels,
        epochs=epochs,
        learning_rate=learning_rate,
        on_epoch=on_epoch,
    )
    return model, history
