import torch
import numpy as np
from agent import ActorCritic, RolloutBuffer

# Define sizes
obs_size, act_size = 19, 6

# Initialize agent
agent = ActorCritic(obs_size, act_size, hidden_size=64)

# Test get_action
obs = torch.randn(obs_size)
action, log_prob, value = agent.get_action(obs)

print(f"Action shape: {action.shape}, log_prob: {log_prob.item():.3f}, value: {value.item():.3f}")

# Test evaluate
obs_batch = torch.randn(32, obs_size)
act_batch = torch.randn(32, act_size)

log_probs, values, entropy = agent.evaluate(obs_batch, act_batch)

print(f"Batch log_probs: {log_probs.shape}, values: {values.shape}, entropy: {entropy.item():.3f}")

print("Agent works!")