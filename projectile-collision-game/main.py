import pygame

from input_handler import get_inputs
from physics import detect_collisions, update_physics
from player import update_player
from render import render_screen
from settings import HEIGHT, WIDTH


def main():
    pygame.init()
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    clock = pygame.time.Clock()
    game_running = True

    while game_running:
        dt = clock.tick(60) / 1000
        pygame.display.set_caption(f"Projectile  {clock.get_fps():.0f} fps")

        inputs = get_inputs()
        game_running = inputs.running

        update_player(inputs)
        update_physics(dt)
        detect_collisions()

        render_screen(screen)

    pygame.quit()


if __name__ == "__main__":
    main()
