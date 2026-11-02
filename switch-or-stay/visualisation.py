"""Draw the lesson plots with Matplotlib.

The decision figure is a square whose corners are the four game situations.
Background colour is the network's probability of switching. A white dashed line
marks where that probability crosses one half.

Loss curves are the ordinary training workflow picture: how wrong the
probabilities were after each epoch, and how often the hard prediction matched
the demonstration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.colors import LinearSegmentedColormap
from torch import nn

from game import CASES, action_name, correct_action


PLOT_BG = "#12141A"
AXES_BG = "#1C2030"
TEXT = "#ECEEF4"
MUTED = "#C5CAD6"
GRID = "#2C3344"
STAY = "#4C8DFF"
SWITCH = "#FF8A3D"

PROBABILITY_CMAP = LinearSegmentedColormap.from_list(
    "stay_switch",
    ["#1D4E89", "#4C8DFF", "#F4F0E6", "#FF8A3D", "#C4541A"],
)


@dataclass
class LossSeries:
    name: str
    color: str
    losses: list[float]
    accuracies: list[float]
    dashed: bool = False


def _style_axes(ax) -> None:
    ax.set_facecolor(AXES_BG)
    ax.tick_params(colors=MUTED)
    for spine in ax.spines.values():
        spine.set_color("#3A4150")
    ax.title.set_color(TEXT)
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def _figure(width: float, height: float):
    figure = plt.figure(figsize=(width, height), dpi=110)
    figure.patch.set_facecolor(PLOT_BG)
    return figure


def _image_from_figure(figure) -> np.ndarray:
    figure.canvas.draw()
    buffer = np.asarray(figure.canvas.buffer_rgba())
    image = np.ascontiguousarray(buffer[:, :, :3])
    plt.close(figure)
    return image


def probability_grid(model: nn.Module, resolution: int = 161) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """P(switch) on a grid covering a little space around the four points."""
    xs = np.linspace(-0.35, 1.35, resolution, dtype=np.float32)
    ys = np.linspace(-0.35, 1.35, resolution, dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    flat = np.stack([xx.ravel(), yy.ravel()], axis=1)
    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(flat)).view(-1)
        probabilities = torch.sigmoid(logits).cpu().numpy().reshape(xx.shape)
    return xx, yy, probabilities


def _smooth(values: list[float], window: int = 15) -> np.ndarray:
    """Moving average so a model that flips around 50% does not draw a barcode."""
    array = np.asarray(values, dtype=np.float64)
    if array.size < window:
        return array
    if window % 2 == 0:
        window += 1
    kernel = np.ones(window) / window
    padded = np.pad(array, window // 2, mode="edge")
    return np.convolve(padded, kernel, mode="valid")


def render_loss_figure(series: list[LossSeries], save_path: Path | str | None = None) -> np.ndarray:
    figure = _figure(11.6, 4.3)
    loss_axis, accuracy_axis = figure.subplots(1, 2)

    for item in series:
        epochs = np.arange(1, len(item.losses) + 1)
        style = "--" if item.dashed else "-"
        loss_axis.plot(epochs, item.losses, style, color=item.color, linewidth=2.2, label=item.name)
        accuracy_axis.plot(
            epochs,
            _smooth(item.accuracies),
            style,
            color=item.color,
            linewidth=2.2,
            label=item.name,
        )

    if series:
        last_epoch = max(len(item.losses) for item in series)
        loss_axis.axhline(np.log(2), color="#9aa3b5", linestyle=":", linewidth=1.1)
        loss_axis.text(
            8,
            float(np.log(2)) + 0.03,
            "guessing 50%",
            ha="left",
            va="bottom",
            color=MUTED,
            fontsize=8,
        )
    loss_axis.set_title("Training loss")
    loss_axis.set_xlabel("Epoch")
    loss_axis.set_ylabel("Binary cross-entropy")
    accuracy_axis.set_title("Accuracy on the demonstrations")
    accuracy_axis.set_xlabel("Epoch")
    accuracy_axis.set_ylabel("Fraction correct")
    accuracy_axis.set_ylim(-0.02, 1.05)
    for axis in (loss_axis, accuracy_axis):
        _style_axes(axis)
        axis.grid(True, color=GRID, linewidth=0.8)
        legend_location = "upper right" if axis is loss_axis else "lower right"
        legend = axis.legend(facecolor="#171B26", edgecolor="#3A4150", fontsize=8, loc=legend_location)
        for text in legend.get_texts():
            text.set_color(TEXT)
    figure.subplots_adjust(left=0.07, right=0.98, bottom=0.14, top=0.88, wspace=0.28)
    return _finish(figure, save_path)


def render_decision_figure(
    left_model: nn.Module,
    left_title: str,
    right_model: nn.Module,
    right_title: str,
    save_path: Path | str | None = None,
) -> np.ndarray:
    figure = _figure(11.2, 5.1)
    left_axis, right_axis = figure.subplots(1, 2)
    _draw_decision(left_axis, left_model, left_title)
    _draw_decision(right_axis, right_model, right_title)
    figure.subplots_adjust(left=0.07, right=0.96, bottom=0.16, top=0.88, wspace=0.45)
    return _finish(figure, save_path)


def _draw_decision(axis, model: nn.Module, title: str) -> None:
    xx, yy, grid = probability_grid(model)
    image = axis.contourf(xx, yy, grid, levels=np.linspace(0, 1, 21), cmap=PROBABILITY_CMAP, vmin=0, vmax=1)
    if float(grid.min()) < 0.5 < float(grid.max()):
        axis.contour(xx, yy, grid, levels=[0.5], colors=["white"], linewidths=2.0, linestyles="--")
    elif float(grid.max() - grid.min()) < 0.08:
        axis.text(
            0.02,
            0.98,
            "no committed cut: about 1/2 everywhere",
            transform=axis.transAxes,
            va="top",
            color=TEXT,
            fontsize=8,
        )
    for car_lane, obstacle_lane in CASES:
        expected = correct_action(car_lane, obstacle_lane)
        colour = SWITCH if expected == 1 else STAY
        axis.scatter(
            [car_lane],
            [obstacle_lane],
            s=220,
            color=colour,
            edgecolors="white",
            linewidths=1.6,
            zorder=3,
        )
    _annotate_probabilities(axis, model)
    axis.set_xlim(-0.45, 1.45)
    axis.set_ylim(-0.45, 1.45)
    axis.set_xticks([0, 1], ["Left", "Right"])
    axis.set_yticks([0, 1], ["Left", "Right"])
    axis.set_xlabel("Car lane")
    axis.set_ylabel("Obstacle lane")
    axis.set_aspect("equal")
    axis.set_title(title)
    _style_axes(axis)
    axis.grid(False)
    colourbar = axis.figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    colourbar.ax.tick_params(colors=MUTED, labelsize=8)
    colourbar.outline.set_edgecolor("#3A4150")
    colourbar.set_label("P(switch)", color=TEXT)


def _point_probability(model: nn.Module, car_lane: int, obstacle_lane: int) -> float:
    model.eval()
    with torch.no_grad():
        features = torch.tensor([[car_lane, obstacle_lane]], dtype=torch.float32)
        return float(torch.sigmoid(model(features).view(-1)[0]))


def _annotate_probabilities(axis, model: nn.Module) -> None:
    offsets = {
        (0, 0): (0.08, -0.18),
        (0, 1): (0.08, 0.14),
        (1, 0): (-0.46, -0.18),
        (1, 1): (-0.46, 0.14),
    }
    for car_lane, obstacle_lane in CASES:
        probability = _point_probability(model, car_lane, obstacle_lane)
        dx, dy = offsets[(car_lane, obstacle_lane)]
        if abs(probability - 0.5) < 0.05:
            label = f"tie  {probability:.2f}"
        else:
            label = f"{action_name(1 if probability >= 0.5 else 0)}  {probability:.2f}"
        axis.text(
            car_lane + dx,
            obstacle_lane + dy,
            label,
            color=TEXT,
            fontsize=8,
            bbox={"boxstyle": "round,pad=0.18", "facecolor": (0.07, 0.08, 0.10, 0.82), "edgecolor": "none"},
        )


def _finish(figure, save_path: Path | str | None) -> np.ndarray:
    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(path, dpi=120)
    return _image_from_figure(figure)
