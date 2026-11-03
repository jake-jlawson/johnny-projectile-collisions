import pygame

from physics import parameters, state
from player import player
from settings import AIM_LENGTH, PIXELS_PER_METER


def world_to_screen(position):
    """Place the projectile on screen. t = 0 sits on the launcher.

    State y grows upward. Screen y grows downward, so the height is flipped.
    """
    initial = parameters.initial_position
    screen_x = player.x + (position[0] - initial[0]) * PIXELS_PER_METER
    screen_y = player.y - (position[1] - initial[1]) * PIXELS_PER_METER
    return int(screen_x), int(screen_y)


def render_screen(screen):
    screen.fill((0, 0, 0))

    end_x = player.x + player.aim_x * AIM_LENGTH
    end_y = player.y + player.aim_y * AIM_LENGTH
    pygame.draw.line(screen, (255, 255, 255), (player.x, player.y), (end_x, end_y), 3)

    # Wings sit back along the aim, one to each side, so the tip is an arrow.
    wing = 16
    spread = 7
    back_x = end_x - player.aim_x * wing
    back_y = end_y - player.aim_y * wing
    side_x = -player.aim_y * spread
    side_y = player.aim_x * spread
    pygame.draw.line(screen, (255, 255, 255), (end_x, end_y), (back_x + side_x, back_y + side_y), 3)
    pygame.draw.line(screen, (255, 255, 255), (end_x, end_y), (back_x - side_x, back_y - side_y), 3)

    pygame.draw.circle(screen, (255, 255, 255), (int(player.x), int(player.y)), 10)

    projectile_x, projectile_y = world_to_screen(state.position)
    pygame.draw.circle(screen, (255, 180, 40), (projectile_x, projectile_y), 8)

    pygame.display.flip()
