"""Load the network and training loop written in student_net.py.

The game reads that file again every time you press G, so a class can edit it,
save, and train without restarting the window.
"""

from __future__ import annotations

import importlib.util
import sys
import traceback
from pathlib import Path

import torch
from torch import nn


STUDENT_NET_PATH = Path(__file__).resolve().parent / "student_net.py"
STUDENT_CLASS_NAME = "StudentNet"
TRAIN_FUNCTION_NAME = "train"


class StudentNetError(Exception):
    """The lesson file is missing, unfinished, or does not match the contract."""


def _load_module(path: Path | str):
    """Re-read `path` and execute it as a fresh module."""
    path = Path(path)
    if not path.exists():
        raise StudentNetError(
            f"Create {path.name} in this folder, with class {STUDENT_CLASS_NAME} "
            f"and def {TRAIN_FUNCTION_NAME}(features, labels, report)."
        )
    module_name = f"student_net_live_{path.stat().st_mtime_ns}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise StudentNetError(f"Could not read {path.name}.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except SyntaxError as error:
        raise StudentNetError(f"Syntax error on line {error.lineno}: {error.msg}") from error
    except Exception as error:
        raise StudentNetError(f"{type(error).__name__}: {error}") from error
    return module


def load_student_train(path: Path | str = STUDENT_NET_PATH):
    """Return the `train` function written in `path`. Each call re-reads the file."""
    path = Path(path)
    module = _load_module(path)
    train = getattr(module, TRAIN_FUNCTION_NAME, None)
    if not callable(train):
        raise StudentNetError(
            f"{path.name} needs def {TRAIN_FUNCTION_NAME}(features, labels, report). "
            "Build the network in that function, call report(loss) once per epoch, and return the network."
        )
    return train


def loss_as_float(loss) -> float:
    """Turn one reported loss into a Python float for the chart."""
    if isinstance(loss, torch.Tensor):
        if loss.numel() != 1:
            raise StudentNetError(
                "report(loss) needs one number, the loss for this epoch, not a value per row."
            )
        return float(loss.detach().reshape(-1)[0].cpu())
    try:
        return float(loss)
    except (TypeError, ValueError) as error:
        raise StudentNetError("report(loss) needs the training loss for this epoch, as one number.") from error


def check_trained_model(model) -> nn.Module:
    """Confirm `train` returned a network that scores one number per row."""
    if not isinstance(model, nn.Module):
        returned = type(model).__name__ if model is not None else "None"
        raise StudentNetError(
            f"train() must return the network. It returned {returned}."
        )
    model.eval()
    sample = torch.zeros(3, 2)
    try:
        with torch.no_grad():
            output = model(sample)
    except Exception as error:
        raise StudentNetError(
            f"The trained network failed on a sample round. {type(error).__name__}: {error}"
        ) from error
    flat = torch.as_tensor(output).reshape(-1)
    if flat.shape[0] != sample.shape[0]:
        raise StudentNetError(
            "forward must return one raw score per row. "
            f"A batch of {sample.shape[0]} rows produced {tuple(torch.as_tensor(output).shape)}."
        )
    return model


def describe_student_error(error: BaseException) -> str:
    """A short message for the lesson screen, including the line in student_net.py."""
    if isinstance(error, StudentNetError):
        return str(error)
    if isinstance(error, NotImplementedError):
        text = str(error).strip()
        return text or "Write the network and def train(features, labels, report)."
    frames = traceback.extract_tb(error.__traceback__)
    own = [frame for frame in frames if Path(frame.filename).name == STUDENT_NET_PATH.name]
    where = f"Line {own[-1].lineno} in {own[-1].name}: " if own else ""
    return f"{where}{type(error).__name__}: {error}"


def load_student_class(path: Path | str = STUDENT_NET_PATH) -> type[nn.Module]:
    """Import `StudentNet` from `path` and check one forward pass.

    The returned object is the class, not a trained model. Call it to build a
    fresh network. Each call re-reads the file.
    """
    path = Path(path)
    module = _load_module(path)
    student_class = getattr(module, STUDENT_CLASS_NAME, None)
    if not isinstance(student_class, type) or not issubclass(student_class, nn.Module):
        raise StudentNetError(
            f"{path.name} needs class {STUDENT_CLASS_NAME}(torch.nn.Module)."
        )
    try:
        model = student_class()
    except Exception as error:
        raise StudentNetError(f"{STUDENT_CLASS_NAME}() failed. {type(error).__name__}: {error}") from error

    model.eval()
    sample = torch.zeros(3, 2)
    try:
        with torch.no_grad():
            output = model(sample)
    except NotImplementedError as error:
        raise StudentNetError(
            "StudentNet.forward is not written yet. Add the layers, save, and train again."
        ) from error
    except Exception as error:
        raise StudentNetError(f"forward failed. {type(error).__name__}: {error}") from error

    flat = torch.as_tensor(output).reshape(-1)
    if flat.shape[0] != sample.shape[0]:
        raise StudentNetError(
            "forward must return one raw score per row. "
            f"A batch of {sample.shape[0]} rows produced {tuple(torch.as_tensor(output).shape)}."
        )
    return student_class
