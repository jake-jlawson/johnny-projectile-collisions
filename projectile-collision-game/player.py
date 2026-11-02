from settings import PLAYER_X, PLAYER_Y


class Player:
    def __init__(self):
        self.x = PLAYER_X
        self.y = PLAYER_Y
        # Unit vector. Starts pointing left, toward the other side of the screen.
        self.aim_x = -1.0
        self.aim_y = 0.0


player = Player()


def update_player(inputs):
    """Point the aim vector from the launcher toward the mouse.

    inputs.clicked is true on the frame the left button goes down.
    Launching a shot from that click comes later.
    """
    if inputs.clicked:
        print("clicked")
        print(player.aim_x, player.aim_y)

    dx = inputs.mouse_x - player.x
    dy = inputs.mouse_y - player.y
    distance = (dx * dx + dy * dy) ** 0.5
    if distance > 0:
        player.aim_x = dx / distance
        player.aim_y = dy / distance

    
