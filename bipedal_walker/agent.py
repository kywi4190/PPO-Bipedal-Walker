"""
PPO agent for the bipedal walker.

This file has everything the training loop needs:
  - ActorCritic: the neural network (separate actor + critic MLPs)
  - RolloutBuffer: collects a batch of experience, then computes GAE
  - ppo_update(): runs the actual PPO-Clip optimization

The idea behind PPO (Proximal Policy Optimization) is simple: collect a bunch
of experience, then do several passes of gradient descent on it, but clip the
policy update so you don't change too much at once and destabilize training.
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Normal

from config import CONFIG


# ── Neural Network ────────────────────────────────────────────────────────────

class ActorCritic(nn.Module):
    """
    Two separate MLPs — one for the actor (policy) and one for the critic
    (value function). Keeping them separate is a bit more stable because the
    actor and critic can have different gradient magnitudes, and a shared
    backbone can create tug-of-war between the two losses.
    """

    def __init__(self, obs_size, action_size, hidden_size=CONFIG["hidden_size"]):
        super().__init__()

        # -- actor network: obs -> action mean --
        self.actor = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_size),
            nn.Tanh(),  # squash action means to [-1, 1]
        )

        # -- critic network: obs -> single value --
        self.critic = nn.Sequential(
            nn.Linear(obs_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1),
        )

        # learnable log standard deviation — one scalar per action dimension.
        # starts at 0 => std = exp(0) = 1, which gives decent exploration
        # at the beginning. the network will shrink this as it gets more
        # confident about what actions to take.
        self.log_std = nn.Parameter(torch.zeros(action_size))

    # -- helper to build the gaussian distribution for the current policy --
    def _get_dist(self, obs):
        """Returns a Normal distribution over actions for the given obs."""
        action_mean = self.actor(obs)
        # exp(log_std) = std, guaranteed positive
        action_std = self.log_std.exp()
        return Normal(action_mean, action_std)

    def get_action(self, obs):
        """
        Used during rollout collection. Given a single observation:
          1. sample an action from the current policy
          2. compute the log-probability of that action (needed later for PPO)
          3. get the value estimate (needed later for GAE)

        Returns (action, log_prob, value) — all tensors.
        """
        dist = self._get_dist(obs)
        action = dist.sample()
        # sum log-probs across action dimensions — we want one scalar per
        # timestep, not one per joint
        log_prob = dist.log_prob(action).sum(dim=-1)
        value = self.critic(obs).squeeze(-1)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        """
        Used during the PPO update. Given batches of observations and the
        actions that were actually taken, recompute everything under the
        *current* policy (which may have changed since we collected the data).

        Returns (log_probs, values, entropy) — all tensors.
        """
        dist = self._get_dist(obs)
        log_probs = dist.log_prob(actions).sum(dim=-1)
        values = self.critic(obs).squeeze(-1)
        # entropy measures how "spread out" the distribution is — higher
        # entropy = more exploration. we'll add this as a bonus to the loss
        # so the policy doesn't collapse to a single action too early.
        entropy = dist.entropy().sum(dim=-1).mean()
        return log_probs, values, entropy


# ── Rollout Buffer ────────────────────────────────────────────────────────────

class RolloutBuffer:
    """
    Stores one rollout worth of experience (observations, actions, rewards,
    etc.) and then computes advantages using GAE once the rollout is done.

    Nothing fancy — just lists that get converted to numpy arrays / tensors
    when we need them.
    """

    def __init__(self):
        self.clear()

    def add(self, obs, action, log_prob, reward, done, value):
        """Append a single transition to the buffer."""
        self.observations.append(obs)
        self.actions.append(action)
        self.log_probs.append(log_prob)
        self.rewards.append(reward)
        self.dones.append(done)
        self.values.append(value)

    def compute_returns_and_advantages(self, last_value, gamma, gae_lambda):
        """
        Generalized Advantage Estimation (GAE) — the standard way to compute
        advantages in PPO.

        The idea: instead of using raw returns (high variance) or just the
        one-step TD error (high bias), GAE blends them with a lambda parameter.
        lambda=0 gives pure TD (low variance, high bias), lambda=1 gives pure
        MC returns (high variance, low bias). 0.95 is the sweet spot.

        We walk backward through the rollout because each advantage depends
        on the next one (it's a recursive formula).
        """
        rewards = np.array(self.rewards)
        dones = np.array(self.dones)
        values = np.array(self.values)

        n = len(rewards)
        self.advantages = np.zeros(n, dtype=np.float32)
        self.returns = np.zeros(n, dtype=np.float32)

        # gae tracks the running advantage as we walk backward
        gae = 0.0
        for t in reversed(range(n)):
            # if this is the last step, the "next value" is the bootstrap
            # value we passed in. otherwise it's the value at t+1.
            if t == n - 1:
                next_value = last_value
            else:
                next_value = values[t + 1]

            # mask out the next value if the episode ended — no bootstrapping
            # across episode boundaries
            not_done = 1.0 - dones[t]

            # delta = one-step TD error: r + gamma*V(s') - V(s)
            delta = rewards[t] + gamma * next_value * not_done - values[t]

            # GAE recursive formula: A_t = delta_t + gamma*lambda*A_{t+1}
            gae = delta + gamma * gae_lambda * not_done * gae
            self.advantages[t] = gae

        # returns = advantages + values (used as the target for the critic)
        self.returns = self.advantages + values

    def get_batches(self, batch_size):
        """
        Yield random minibatches of the stored rollout as PyTorch tensors.
        Shuffling is important — without it the optimizer would see correlated
        sequences, which hurts learning.
        """
        n = len(self.rewards)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # convert everything to numpy arrays for easy slicing
        observations = np.array(self.observations)
        actions = np.array(self.actions)
        log_probs = np.array(self.log_probs)

        # shuffle the indices
        indices = np.arange(n)
        np.random.shuffle(indices)

        # yield chunks of size batch_size
        for start in range(0, n, batch_size):
            end = start + batch_size
            batch_idx = indices[start:end]

            yield {
                "observations": torch.tensor(
                    observations[batch_idx], dtype=torch.float32, device=device
                ),
                "actions": torch.tensor(
                    actions[batch_idx], dtype=torch.float32, device=device
                ),
                "old_log_probs": torch.tensor(
                    log_probs[batch_idx], dtype=torch.float32, device=device
                ),
                "advantages": torch.tensor(
                    self.advantages[batch_idx], dtype=torch.float32, device=device
                ),
                "returns": torch.tensor(
                    self.returns[batch_idx], dtype=torch.float32, device=device
                ),
            }

    def clear(self):
        """Reset all storage for the next rollout."""
        self.observations = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.advantages = None
        self.returns = None


# ── PPO Update ────────────────────────────────────────────────────────────────

def ppo_update(agent, optimizer, buffer, config=CONFIG):
    """
    The main PPO-Clip update. Takes a full rollout buffer and does several
    epochs of minibatch gradient descent on it.

    The key insight of PPO: we want to improve the policy using this batch of
    data, but if we change the policy too much, the data becomes "stale" and
    training goes off the rails. The clipping mechanism prevents this by
    capping how much the probability ratio can change.

    Returns a dict of loss statistics for logging/debugging.
    """
    clip_epsilon = config["clip_epsilon"]
    vf_coef = config["vf_coef"]
    entropy_coef = config["entropy_coef"]
    max_grad_norm = config["max_grad_norm"]
    num_epochs = config["ppo_epochs"]
    minibatch_size = config["minibatch_size"]

    # compute GAE advantages and returns before we start updating
    # we need the value of the last state to bootstrap the advantage calculation
    device = next(agent.parameters()).device
    last_obs = buffer.observations[-1]
    with torch.no_grad():
        last_value = agent.critic(
            torch.tensor(last_obs, dtype=torch.float32, device=device)
        ).item()
    buffer.compute_returns_and_advantages(
        last_value, config["gamma"], config["gae_lambda"]
    )

    # accumulators for logging
    total_policy_loss = 0.0
    total_value_loss = 0.0
    total_entropy = 0.0
    total_approx_kl = 0.0
    num_updates = 0

    for _epoch in range(num_epochs):
        for batch in buffer.get_batches(minibatch_size):
            obs = batch["observations"]
            actions = batch["actions"]
            old_log_probs = batch["old_log_probs"]
            advantages = batch["advantages"]
            returns = batch["returns"]

            # normalize advantages within the minibatch — this reduces
            # variance and makes training way more stable
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # ask the *current* policy to evaluate the old actions
            new_log_probs, values, entropy = agent.evaluate(obs, actions)

            # -- policy loss (the PPO-Clip objective) --
            # ratio = pi_new(a|s) / pi_old(a|s)
            # if the ratio is close to 1, the policy hasn't changed much
            ratio = (new_log_probs - old_log_probs).exp()

            # unclipped objective: ratio * advantage
            surr1 = ratio * advantages
            # clipped objective: cap the ratio to [1-eps, 1+eps]
            # this is the clipped surrogate — it stops the policy from
            # changing too much at once
            surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
            # take the min of clipped and unclipped — this is pessimistic,
            # which is the whole point of PPO (be conservative)
            policy_loss = -torch.min(surr1, surr2).mean()

            # -- value loss --
            # simple MSE between predicted values and computed returns
            value_loss = ((values - returns) ** 2).mean() * vf_coef

            # -- entropy bonus --
            # negative because we want to *maximize* entropy (more exploration)
            # but optimizers minimize loss
            entropy_bonus = -entropy.mean() * entropy_coef

            # -- total loss --
            loss = policy_loss + value_loss + entropy_bonus

            # -- gradient step --
            optimizer.zero_grad()
            loss.backward()
            # clip gradients to prevent exploding gradients from wrecking
            # the network weights
            nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()

            # -- logging stats --
            # approximate KL divergence between old and new policy.
            # this isn't used in the loss, but it's a great diagnostic —
            # if KL spikes, the policy changed too much and you might want
            # to lower the learning rate or increase clipping.
            with torch.no_grad():
                approx_kl = ((ratio - 1.0) - ratio.log()).mean().item()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
            total_entropy += entropy.mean().item()
            total_approx_kl += approx_kl
            num_updates += 1

    # average everything over all the minibatch updates
    return {
        "policy_loss": total_policy_loss / num_updates,
        "value_loss": total_value_loss / num_updates,
        "entropy": total_entropy / num_updates,
        "approx_kl": total_approx_kl / num_updates,
    }
