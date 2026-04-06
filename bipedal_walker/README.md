# PPO Bipedal Walker

A 2D ragdoll walker trained with Proximal Policy Optimization (PPO) using a custom pymunk physics simulation and pygame visualization.

## Install dependencies

```bash
pip install pymunk pygame torch numpy
```

## Train

```bash
cd bipedal_walker
python train.py
```

Training prints rolling reward statistics every few updates and saves the best checkpoint to `checkpoints/ppo_bipedal.pt`. Hit Ctrl+C to stop early (the model is saved on exit).

## Visualize

Random actions (no model):
```bash
python visualize.py
```

Trained model:
```bash
python visualize.py --model checkpoints/ppo_bipedal.pt
```

Controls: ESC = quit, SPACE = pause, R = reset episode.

## Configuration

All hyperparameters live in `config.py` in a single `CONFIG` dict: physics settings, body dimensions, joint limits, PPO hyperparameters, reward shaping weights, and logging/save paths. Edit that file to tweak anything.
