"""Plots render without a display."""

import numpy as np

from models import LinearNet, NonlinearNet
from visualisation import LossSeries, render_decision_figure, render_loss_figure


def test_loss_and_decision_figures_are_rgb_images():
    series = [
        LossSeries("Linear", "#8EB4FF", [0.7, 0.4], [0.5, 0.75], dashed=True),
        LossSeries("ReLU", "#FFB15E", [0.7, 0.1], [0.5, 1.0], dashed=False),
    ]
    loss_image = render_loss_figure(series)
    decision_image = render_decision_figure(LinearNet(), "Linear", NonlinearNet(), "ReLU")
    for image in (loss_image, decision_image):
        assert image.ndim == 3
        assert image.shape[2] == 3
        assert image.dtype == np.uint8
        assert image.shape[0] > 50
        assert image.shape[1] > 50
