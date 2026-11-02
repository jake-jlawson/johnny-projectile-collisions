import pygame


class Inputs:
    """What the player did during this one frame."""

    def __init__(self, mouse_x, mouse_y, clicked, running):
        self.mouse_x = mouse_x
        self.mouse_y = mouse_y
        self.clicked = clicked
        self.running = running


def get_inputs():
    """Read the mouse, a new left click, and whether the window should close."""
    clicked = False
    running = True
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
            running = False
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            clicked = True
    mouse_x, mouse_y = pygame.mouse.get_pos()
    return Inputs(mouse_x, mouse_y, clicked, running)
