# Switch or Stay

A two-lane game for a lesson on why a neural network needs a **nonlinear activation**.

The student drives a car. Each round, an obstacle sits in the left lane or the right lane. The car must switch lanes when the obstacle is in its lane, and stay when it is not. That rule is an XNOR pattern. No single straight cut can separate the "switch" situations from the "stay" situations.

The student plays, the game saves their keypresses, and several networks train by imitation. A linear network and a deeper linear network cannot learn the rule. A network of the same size with a ReLU between the layers can.

## Setup

From this folder, with Python 3.10 or newer:

```powershell
python -m pip install -r requirements.txt
python main.py
```

Run the tests with:

```powershell
python -m pytest
```

The window is drawn with Pygame. Training uses PyTorch. The CSV log uses the standard `csv` module. NumPy builds the training arrays. Matplotlib draws the loss curves and the decision pictures.

## Write the network and the training loop

`student_net.py` is unfinished on purpose. The game calls `train(features, labels, report)` in that file. It re-reads the file on every press of `G`, so the window stays open while you type.

1. Play until the Data screen has a success on every card. Press `G` only after that, or the loop has no rows to learn from.
2. In `StudentNet`, add the layers in `__init__` and use them in `forward`. Return one raw score per row. Leave sigmoid out of the network.
3. In `train`, turn `features` and `labels` into tensors, pick a loss and an optimiser, and loop. Call `report(loss)` once per epoch. Return the network. Delete both `raise` lines. Save.
4. Click the game and press `G`. Each `report(loss)` adds a point to the orange line. A few hundred epochs takes about fifteen seconds, so the line is visible as it moves.
5. Press `U`. The network `train` returned drives the car. Those rounds are not added to the CSV.
6. Put `torch.relu` between the layers, save, and press `G` again. The same key runs whatever is saved now. Without the activation the loss usually stays near 0.69. With it, the line can fall toward zero.

`features` has shape `(rows, 2)`: car lane, obstacle lane. `labels` has shape `(rows,)`: the stay or switch key that was pressed. `T` still trains the six comparison networks with the loop in `training.py`. `G` runs only the loop in `student_net.py`. `Y` opens its chart.

## How to run the lesson

1. **Play.** Press `A` to stay and `S` to switch. The rule is on screen: switch only when the obstacle is in your lane. Every human keypress is appended to `data/gameplay.csv` immediately.
2. **Check the data.** Open the Data screen. Train once each of the four cards has at least one success. Misses stay in the file; the default training run skips them.
3. **Train.** Press `T`. Every network sees the same rows, the same Adam settings, and the same number of epochs. The inputs are only `car_lane` and `obstacle_lane`. The target is the key the student pressed.
4. **Watch the AIs.** Press `L` for the linear network and `N` for the ReLU network, or use `1` through `6` for the other architectures. AI rounds are scored on screen and are **not** written into the training file. `R` clears those scores and leaves the CSV alone.
5. **Read the loss curves.** Dashed lines have no hidden activation. On a balanced set of correct demonstrations they sit on the loss of guessing 50%. The solid curves fall toward zero.
6. **Look at the boundaries.** Orange points are situations where the correct action is switch. Blue points should stay. The background is the network's probability of switching. The linear picture stays near one half, because a straight cut that gets three corners right is confidently wrong on the fourth, and cross-entropy would rather stay unsure. The ReLU picture can colour the two switch corners without colouring the stay corners.

Copies of the two plots are written to `data/loss_curves.png` and `data/decision_boundaries.png` after training, so they can be dropped into slides.

## What the networks are

| Key | Network | Shape | Hidden activation | What to notice |
|---|---|---|---|---|
| 1 | Linear | 2 → 4 → 1 | none | Same 17 weights as ReLU. Still one straight cut. |
| 2 | Deep linear | 2 → 8 → 8 → 1 | none | More layers and more weights. Still one straight cut. |
| 3 | ReLU | 2 → 4 → ReLU → 1 | ReLU | The bend. This is the network that solves the game. |
| 4 | Deep ReLU | 2 → 8 → ReLU → 8 → ReLU → 1 | ReLU | Depth helps once an activation can bend. |
| 5 | Sigmoid | 2 → 4 → sigmoid → 1 | sigmoid | A smooth nonlinearity. Compare its loss curve with ReLU. |
| 6 | Tanh | 2 → 4 → tanh → 1 | tanh | Another nonlinearity. The shape of the loss curve changes. |

The output of every network is a raw score. Training uses `BCEWithLogitsLoss`, which applies the sigmoid inside the loss. The probability shown in the game is that same sigmoid, read afterwards. A sigmoid on the final score does not bend the boundary: the probability crosses 50% exactly where the score crosses zero.

On a complete, balanced set of correct demonstrations the linear networks stay near 50% on all four corners. The on-screen 50% rule then has to pick one action, so that car does the same thing in every situation and is right about half the time. The nonlinear networks can reach all four corners. If a situation is missing from the training file, the screen says so, and the predictions are whatever the network actually produced.

## Keys

| Key | Action |
|---|---|
| A | Stay in the current lane |
| S | Switch lanes |
| H | Human play, recorded |
| L | Linear AI |
| N | ReLU AI |
| 1–6 | Choose a network to drive |
| Y | Open the screen for the network you are writing |
| G | Run `train()` in `student_net.py` and watch the loss |
| U | Let that network drive the car |
| T | Train every network |
| I | Next training run uses successes only, or every action including misses |
| R | Reset AI scores, keep the CSV |
| P, D, C, V | Play, Data, loss Curves, boundaries |
| Left / Right | Change the network on the right of the boundary picture |
| Esc | Quit |

The same actions are buttons on screen.

## Project files

| File | Role |
|---|---|
| `main.py` | Window, game drawing, and the lesson screens |
| `game.py` | The rule, the truth table, and the shuffled rounds |
| `data_logger.py` | Append-only CSV load and save |
| `student_net.py` | The unfinished network and training loop you write during the lesson |
| `models.py` | The six networks, with comments on why a linear stack stays linear |
| `training.py` | Adam, binary cross-entropy, and the four-corner predictions |
| `visualisation.py` | Loss curves and decision pictures |
| `tests/` | Rules, logging, features, and the linear-versus-nonlinear result |

`models.py` and `training.py` are the files to put on screen if you want to read the code with the student. The algebra for "two linear layers are still one linear layer" is the comment on `LinearNet`.

## The truth table

| Car lane | Obstacle lane | Correct action |
|---|---|---|
| left (0) | left (0) | switch (1) |
| left (0) | right (1) | stay (0) |
| right (1) | left (0) | stay (0) |
| right (1) | right (1) | switch (1) |

Scenarios are dealt in shuffled cycles of these four rows, so a short game still covers the table.

## If something looks wrong

- **ReLU did not get 4 out of 4.** Open Data. Each card needs a successful demonstration. Default training ignores misses. Including mistakes on purpose is a separate discussion: the network imitates what was pressed, including errors.
- **The CSV will not save.** Close `data/gameplay.csv` if a spreadsheet has it open, then play another round.
- **Keys do nothing.** Click the game window so it has focus.
