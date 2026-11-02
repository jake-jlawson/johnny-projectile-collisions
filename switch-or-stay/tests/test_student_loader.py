"""The lesson file can be loaded, rejected, and trained."""

import numpy as np
import torch

from student_loader import (
    StudentNetError,
    check_trained_model,
    describe_student_error,
    load_student_class,
    load_student_train,
)
from training import fit


def _write(path, source: str):
    path.write_text(source, encoding="utf-8")


def test_missing_file_explains_what_to_create(tmp_path):
    try:
        load_student_class(tmp_path / "student_net.py")
    except StudentNetError as error:
        assert "StudentNet" in str(error)
    else:
        raise AssertionError("a missing file should be reported")


def test_unfinished_forward_is_reported(tmp_path):
    path = tmp_path / "student_net.py"
    _write(
        path,
        """
import torch
from torch import nn

class StudentNet(nn.Module):
    def forward(self, x):
        raise NotImplementedError("not yet")
""",
    )
    try:
        load_student_class(path)
    except StudentNetError as error:
        assert "forward" in str(error).lower()
    else:
        raise AssertionError("an unfinished network should be reported")


def test_wrong_output_shape_is_reported(tmp_path):
    path = tmp_path / "student_net.py"
    _write(
        path,
        """
from torch import nn

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)

    def forward(self, x):
        return self.layer(x)
""",
    )
    try:
        load_student_class(path)
    except StudentNetError as error:
        assert "one raw score" in str(error)
    else:
        raise AssertionError("two scores per row should be rejected")


def test_a_finished_network_reloads_and_can_train(tmp_path):
    path = tmp_path / "student_net.py"
    _write(
        path,
        """
import torch
from torch import nn

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))
""",
    )
    first = load_student_class(path)
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    labels = np.array([1, 0, 0, 1], dtype=np.float32)
    seen = []
    torch.manual_seed(1)
    history = fit(first(), features, labels, epochs=5, on_epoch=lambda *args: seen.append(args[0]), report_every=1)
    assert seen == [0, 1, 2, 3, 4]
    assert len(history.losses) == 5

    _write(
        path,
        """
from torch import nn

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)

    def forward(self, x):
        return self.layer(x)
""",
    )
    second = load_student_class(path)
    assert second is not first
    with torch.no_grad():
        output = second()(torch.zeros(2, 2))
    assert tuple(output.shape) == (2, 1)


WORKING_TRAIN = """
import torch
from torch import nn

class StudentNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 4)
        self.output = nn.Linear(4, 1)

    def forward(self, x):
        return self.output(torch.relu(self.hidden(x)))

def train(features, labels, report):
    model = StudentNet()
    inputs = torch.as_tensor(features)
    targets = torch.as_tensor(labels)
    loss_fn = nn.BCEWithLogitsLoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.1)
    for _ in range(4):
        optimiser.zero_grad()
        loss = loss_fn(model(inputs).view(-1), targets)
        loss.backward()
        optimiser.step()
        report(loss)
    return model
"""


def test_missing_train_function_is_reported(tmp_path):
    path = tmp_path / "student_net.py"
    _write(path, "x = 1\n")
    try:
        load_student_train(path)
    except StudentNetError as error:
        assert "train(features, labels, report)" in str(error)
    else:
        raise AssertionError("a file without train() should be reported")


def test_written_loop_reports_each_epoch_and_returns_the_network(tmp_path):
    path = tmp_path / "student_net.py"
    _write(path, WORKING_TRAIN)
    train = load_student_train(path)
    features = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=np.float32)
    labels = np.array([1, 0, 0, 1], dtype=np.float32)
    seen = []
    torch.manual_seed(1)
    model = check_trained_model(train(features, labels, seen.append))
    assert len(seen) == 4
    assert all(isinstance(value, torch.Tensor) for value in seen)
    with torch.no_grad():
        scores = model(torch.as_tensor(features))
    assert tuple(scores.shape) == (4, 1)

    _write(
        path,
        """
from torch import nn

class Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 1)
    def forward(self, x):
        return self.layer(x)

def train(features, labels, report):
    model = Tiny()
    report(0.5)
    return model
""",
    )
    again = load_student_train(path)
    assert again is not train
    returned = check_trained_model(again(features, labels, lambda loss: None))
    assert type(returned).__name__ == "Tiny"


def test_train_must_return_one_score_per_row(tmp_path):
    path = tmp_path / "student_net.py"
    _write(
        path,
        """
def train(features, labels, report):
    return None
""",
    )
    try:
        check_trained_model(load_student_train(path)(None, None, lambda loss: None))
    except StudentNetError as error:
        assert "return the network" in str(error)
    else:
        raise AssertionError("returning nothing should be reported")

    _write(
        path,
        """
from torch import nn

class Wide(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(2, 2)
    def forward(self, x):
        return self.layer(x)

def train(features, labels, report):
    return Wide()
""",
    )
    try:
        check_trained_model(load_student_train(path)(None, None, lambda loss: None))
    except StudentNetError as error:
        assert "one raw score" in str(error)
    else:
        raise AssertionError("two scores per row should be rejected")


def test_error_message_names_the_line_in_the_lesson_file(tmp_path):
    path = tmp_path / "student_net.py"
    _write(
        path,
        """
def train(features, labels, report):
    raise RuntimeError("bad step")
""",
    )
    try:
        load_student_train(path)(None, None, lambda loss: None)
    except RuntimeError as error:
        message = describe_student_error(error)
    else:
        raise AssertionError("the training error should surface")
    assert message.startswith("Line 3 in train:")
    assert "RuntimeError: bad step" in message
