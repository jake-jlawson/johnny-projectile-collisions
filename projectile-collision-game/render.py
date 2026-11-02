import pygame

from player import player
from settings import AIM_LENGTH


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

    pygame.display.flip()
