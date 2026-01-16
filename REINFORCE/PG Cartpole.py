import gym
import numpy as np

from RL_brain import PolicyGradient
import matplotlib.pyplot as plt

RENDER = False
DISPLAY_REWARD_THRESHOLD = 100

env = gym.make('CartPole-v0', render_mode='human' if RENDER else None)
env = env.unwrapped
seed = 1

RL = PolicyGradient(
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.001,
    reward_decay=0.99,
)

# running_reward = None
for i_episode in range(700):
    observation, info = env.reset(seed=seed)

    while True:
        action = RL.choose_action(observation)
        observation_, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        RL.store_transition(observation, action, reward)
        if done:
            ep_rs_sum = sum(RL.ep_rs)
            if 'running_reward' not in globals():
                running_reward = ep_rs_sum
            else:
                running_reward = running_reward * 0.99 + ep_rs_sum * 0.01
            if running_reward > DISPLAY_REWARD_THRESHOLD:
                RENDER = True
            print("episode:", i_episode, "  reward:", int(running_reward))

            vt = RL.learn()

            break

        observation = observation_