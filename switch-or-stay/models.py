"""The networks compared in the lesson.

Two of them are the main pair. They have the same layers and the same number of
weights. The only difference is a ReLU between the layers.

The others are there so a class can see what does, and does not, change the
result: more linear layers, a deeper ReLU net, and two other activation functions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
from torch import nn


LINEAR_STACK_EXPLANATION = (
    "Stacking linear layers does not create new shapes. "
    "Linear(4, 1) after Linear(2, 4) is still one weighted sum, output = w·x + b. "
    "A sigmoid turns that sum into a probability, and the probability crosses one half "
    "on a straight line. For this balanced game the best straight model stays near "
    "one half everywhere: a line that gets three corners right is confidently wrong "
    "on the fourth, and cross-entropy would rather stay unsure."
)

RELU_EXPLANATION = (
    "ReLU(z) = max(0, z) is a bend. Each hidden unit can be off on one side of a "
    "line and on, on the other. With a few of those bends, the network can give a "
    "high switch score to both opposite corners without also selecting the stay corners."
)


class LinearNet(nn.Module):
    """2 inputs -> Linear(2, 4) -> Linear(4, 1), with no activation in between.

    Why this cannot solve the game
    ------------------------------
    Write the layers as h = W1 x + b1 and y = W2 h + b2. Substitute:

        y = W2 (W1 x + b1) + b2 = (W2 W1) x + (W2 b1 + b2)

    That is one linear layer with weight W2 W1 and bias W2 b1 + b2. Extra linear
    layers add numbers to multiply, not new shapes the model can draw.

    The training loss uses logits (the raw score y). A sigmoid is applied only
    when we read y as a probability of switching. sigmoid(y) = 0.5 exactly when
    y = 0. The set of points where a linear score is zero is a straight line, so
    the sigmoid does not make the decision boundary curved.
    """

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        return self.output(x)


class NonlinearNet(nn.Module):
    """2 inputs -> Linear(2, 4) -> ReLU -> Linear(4, 1).

    Same sizes as LinearNet. The ReLU is the whole difference.
    """

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.activation = nn.ReLU()
        self.output = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.activation(x)
        return self.output(x)


class DeepLinearNet(nn.Module):
    """2 -> 8 -> 8 -> 1, still with no hidden activation.

    This has many more weights than LinearNet and is still one affine function,
    so it has the same limitation: one straight cut.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.layer2 = nn.Linear(8, 8)
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        return self.layer3(x)


class DeepReLUNet(nn.Module):
    """2 -> 8 -> ReLU -> 8 -> ReLU -> 1.

    A deeper nonlinear net. Depth helps here because each hidden block can bend
    the representation again. Depth did not help DeepLinearNet, because nothing
    between those layers bent the sum.
    """

    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(2, 8)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(8, 8)
        self.relu2 = nn.ReLU()
        self.layer3 = nn.Linear(8, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu1(self.layer1(x))
        x = self.relu2(self.layer2(x))
        return self.layer3(x)


class SigmoidNet(nn.Module):
    """2 -> 4 -> sigmoid -> 1.

    Sigmoid is a smooth nonlinearity, so this model can represent a bent rule.
    It often needs the loss curve to fall for longer than ReLU, because the
    slope of a sigmoid is small at both extremes.
    """

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.activation = nn.Sigmoid()
        self.output = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.activation(x)
        return self.output(x)


class TanhNet(nn.Module):
    """2 -> 4 -> tanh -> 1.

    Tanh is another nonlinear activation. Comparing its loss curve with ReLU and
    sigmoid shows that the activation you pick changes training, not just the diagram.
    """

    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.activation = nn.Tanh()
        self.output = nn.Linear(4, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.hidden(x)
        x = self.activation(x)
        return self.output(x)


@dataclass(frozen=True)
class Architecture:
    key: str
    name: str
    diagram: str
    nonlinearity: str
    summary: str
    factory: Callable[[], nn.Module]
    color: str
    primary: bool = False
    linear_family: bool = False


ARCHITECTURES: tuple[Architecture, ...] = (
    Architecture(
        key="linear",
        name="Linear",
        diagram="2 -> 4 -> 1",
        nonlinearity="none",
        summary="Same layers and weights as ReLU, with no bend. One straight cut.",
        factory=LinearNet,
        color="#8EB4FF",
        primary=True,
        linear_family=True,
    ),
    Architecture(
        key="deep_linear",
        name="Deep linear",
        diagram="2 -> 8 -> 8 -> 1",
        nonlinearity="none",
        summary="More linear layers, and still only one straight cut.",
        factory=DeepLinearNet,
        color="#A8B3C7",
        linear_family=True,
    ),
    Architecture(
        key="relu",
        name="ReLU",
        diagram="2 -> 4 -> ReLU -> 1",
        nonlinearity="ReLU",
        summary="The bend that lets this size of network solve the game.",
        factory=NonlinearNet,
        color="#FFB15E",
        primary=True,
    ),
    Architecture(
        key="deep_relu",
        name="Deep ReLU",
        diagram="2 -> 8 -> ReLU -> 8 -> ReLU -> 1",
        nonlinearity="ReLU",
        summary="Two ReLU layers. Depth is useful once an activation can bend.",
        factory=DeepReLUNet,
        color="#FF7B62",
    ),
    Architecture(
        key="sigmoid",
        name="Sigmoid",
        diagram="2 -> 4 -> sigmoid -> 1",
        nonlinearity="sigmoid",
        summary="A smooth nonlinear activation. It can represent the rule and trains more slowly.",
        factory=SigmoidNet,
        color="#D2A6FF",
    ),
    Architecture(
        key="tanh",
        name="Tanh",
        diagram="2 -> 4 -> tanh -> 1",
        nonlinearity="tanh",
        summary="Another nonlinear activation. Compare how fast its loss falls.",
        factory=TanhNet,
        color="#62E0C4",
    ),
)

PRIMARY_LINEAR = "linear"
PRIMARY_NONLINEAR = "relu"


def architecture_by_key(key: str) -> Architecture:
    for architecture in ARCHITECTURES:
        if architecture.key == key:
            return architecture
    raise KeyError(key)


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
