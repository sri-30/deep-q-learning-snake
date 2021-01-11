import snake_env as s
import pygame
import matplotlib.pyplot as plt

player = s.SNAKE()

player.open_screen()
player.render()
action = 0

# For playing the snake game and seeing how the environment works and looks

while 1:
    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_UP:
                action = 0
            elif event.key == pygame.K_LEFT:
                action = 1
            elif event.key == pygame.K_RIGHT:
                action = 2
            elif event.key == pygame.K_SPACE:
                player.eat_food()
                plt.imshow(player.get_state())
                plt.show()
    player.step(action)
    action = 0
    player.clock.tick(10)   # Increase this to increase the speed of the game in frame updates per second
    player.render()
    if player.done:
        player.reset()
