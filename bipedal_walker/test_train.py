from config import CONFIG

# Quick test with very few steps
CONFIG['total_training_timesteps'] = 4096
CONFIG['rollout_steps'] = 512
CONFIG['log_interval'] = 1

from train import train

train(CONFIG)

print("Training loop works!")