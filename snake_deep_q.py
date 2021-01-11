import snake_env as s
import deep_q_agent as dq
import os

# For Training Models

player = s.SNAKE()

action = 0
player.step(0)
state_shape = player.state_shape
action_size = player.action_space[0]
num_episodes = 3000
output_dir = "snake_models/model_output_snake_4/snake_"

# Use this to load a pre-trained model
# input_dir = "snake_models/model_output_snake_3/snake_weights_1050.hdf5"
# agent.load(input_dir)

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

agent = dq.DQNAgent(state_shape, action_size, 1, 0.95, 0.001)

done = False
batch_size = 32
steps = 0

for e in range(num_episodes):
    player.reset()
    state = player.get_state()
    for time in range(1000):
        steps += 1
        action = agent.get_action(state)
        next_state, done, reward = player.step(action)
        agent.store_memory(state, action, reward, next_state, done)
        state = next_state
        if steps % 100 == 0:  # Target Network Weights are updated every 100 steps
            agent.update_target()
        if done:
            print(
                f"episode: {e}/{num_episodes}, lifespan: {time}, food:{player.food_score}, e: {agent.epsilon}")
            break
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)
    if e % 50 == 0:
        agent.save(output_dir + "weights_" + f"{e}" + ".hdf5")
