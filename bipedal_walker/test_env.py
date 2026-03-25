from environment import BipedalWalkerEnv
from config import CONFIG
import numpy as np

env = BipedalWalkerEnv(CONFIG)

obs = env.reset()
print(f"Obs shape: {obs.shape}, Obs size from env: {env.observation_size}")
print(f"Action size: {env.action_size}")

# Run a few random steps
total_reward = 0
info = {}

for i in range(100):
    action = np.random.uniform(-1, 1, size=env.action_size)
    obs, reward, done, info = env.step(action)
    total_reward += reward

    if done:
        print(f"Episode ended at step {i + 1}")
        break

print(f"Total reward over episode: {total_reward:.2f}")
print(f"Final torso x: {info['torso_x']:.1f}")
print("Environment works!")