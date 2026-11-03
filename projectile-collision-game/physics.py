import time
from types import SimpleNamespace

import numpy as np


# global constants
constants = SimpleNamespace(
    g=9.81,
)

# global parameters
initial_speed = 20
theta = 45
# cos and sin take radians. theta stays in degrees, the way it was written.
parameters = SimpleNamespace(
    initial_speed=initial_speed,
    theta=theta,
    v0=np.array([
        initial_speed * np.cos(np.deg2rad(theta)),
        initial_speed * np.sin(np.deg2rad(theta)),
    ]),
    initial_position=np.array([3.0, 4.0]),
)

state = SimpleNamespace(
    position=np.array(parameters.initial_position, dtype=float),
)

# Clock reading at launch. The first update is t = 0.
runtime = None


def update_state(state):
    # update of the time
    global runtime
    now = time.perf_counter()
    if runtime is None:
        runtime = now
    t = now - runtime

    v0 = parameters.v0
    initial_position = parameters.initial_position

    # Position since launch. x steps left, the way the launcher faces.
    # y is up: +v0*t, then gravity pulls it back down.
    new_position = np.array([
        initial_position[0] - v0[0] * t,
        initial_position[1] + v0[1] * t - (1 / 2) * constants.g * t**2,
    ])

    state.position = new_position
    print(state.position, t)


def update_physics(dt):
    """One projectile, launched at t = 0 when the game starts."""
    update_state(state)


def detect_collisions():
    """Hits, edges, and the score go here."""
    return
