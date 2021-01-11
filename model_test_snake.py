import snake_env as s
import pygame
import deep_q_agent as dq

# For testing Models

player = s.SNAKE()
action = 0
player.step(0)
state_shape = player.state_shape
action_size = player.action_space[0]
n_episodes = 2000

# Use this to load the model
input_dir = "snake_models/model_output_snake_3/snake_weights_1050.hdf5"

player.open_screen()
player.render()
agent = dq.DQNAgent(state_shape, action_size, 1, 0.95, 0.001)
agent.load(input_dir)
agent.epsilon = 0

while 1:
    player.reset()
    state = player.get_state()
    done = False
    i = 0
    while not done:
        i += 1
        pygame.event.get()
        action = agent.get_action(state)
        next_state, done, reward = player.step(action)
        state = next_state
        player.clock.tick(20)
        player.render()
        if player.done:
            player.reset()
            break
