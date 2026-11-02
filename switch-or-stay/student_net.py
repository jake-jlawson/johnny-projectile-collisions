"""The network and the training loop you write with your student.

Save this file, click the game, and press G.
The game re-reads this file every time, so you do not restart the window.

train(features, labels, report)
--------------------------------
features   NumPy array, shape (rows, 2).
           Column 0 is the car lane, column 1 is the obstacle lane.
           0 is left, 1 is right.
labels     NumPy array, shape (rows,).
           0 is the stay key, 1 is the switch key.
           These are the keys that were pressed, not a rule written here.
report     Call report(loss) once per epoch. The chart draws that number.
           Pass the loss tensor or a float.

Return the trained network. forward returns one raw score per row, shape
(rows, 1). Do not apply sigmoid in the network. Use a loss that expects raw
scores, such as torch.nn.BCEWithLogitsLoss.

Press U afterwards to let the returned network drive. Those rounds are not saved.
"""

import torch
from torch import nn


# training examples (left=0, right=1)
x = torch.tensor([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1],
], dtype=torch.float32)

# 0 is stay, 1 is switch
y = torch.tensor([
    [1],
    [0],
    [0],
    [1],
], dtype=torch.float32)


model = nn.Sequential(
    nn.Linear(2, 8),
    nn.Tanh(),
    nn.Linear(8, 1),
    nn.Sigmoid(),
)

loss_fn = nn.BCELoss()
row_loss_fn = nn.BCELoss(reduction="none")
optimiser = torch.optim.Adam(model.parameters(), lr=0.1)

# The loss moves in the first few dozen epochs. Later lines show whether it is still moving.
SHOW_EPOCHS = {1, 10, 25, 50, 100, 200, 500, 1000, 2000, 5000, 10000}
LANE = ("left", "right")
ACTION = ("stay", "switch")


def gradient_size():
    total = 0.0
    for parameter in model.parameters():
        if parameter.grad is not None:
            total += float(parameter.grad.detach().abs().sum())
    return total


def print_snapshot(epoch, y_pred, loss):
    output = y_pred.detach().view(-1)
    read_by_loss = torch.sigmoid(output)
    print(
        f"Epoch {epoch:5d}  loss {float(loss.detach()):.4f}  gradient {gradient_size():.2e}\n"
        f"  network output   {[round(value, 4) for value in output.tolist()]}\n"
        f"  loss reads it as {[round(value, 4) for value in read_by_loss.tolist()]}"
    )


def print_rows():
    with torch.no_grad():
        output = model(x).view(-1)
        read_by_loss = torch.sigmoid(output)
        row_loss = row_loss_fn(output, y.view(-1))
    print()
    print(f"{'car':<8}{'obstacle':<10}{'target':<8}{'output':>10}{'loss reads':>12}{'row loss':>10}")
    corners_right = 0
    for row in range(len(x)):
        target = int(y[row].item())
        predicted = 1 if output[row].item() >= 0.5 else 0
        corners_right += int(predicted == target)
        print(
            f"{LANE[int(x[row, 0])]:<8}"
            f"{LANE[int(x[row, 1])]:<10}"
            f"{ACTION[target]:<8}"
            f"{output[row].item():10.4f}"
            f"{read_by_loss[row].item():12.4f}"
            f"{row_loss[row].item():10.4f}"
        )
    print(f"mean loss {float(row_loss.mean()):.4f}    corners right {corners_right}/4")
    print("output is the network's number. BCEWithLogitsLoss runs sigmoid on it again before scoring.")


for epoch in range(1, 10001):
    # forward pass
    y_pred = model(x)
    loss = loss_fn(y_pred, y)

    # backward pass
    optimiser.zero_grad()
    loss.backward()
    if epoch in SHOW_EPOCHS:
        print_snapshot(epoch, y_pred, loss)
    optimiser.step()

print_rows()

