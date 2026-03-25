"""
train.py - Main PPO training loop for the bipedal walker.

This is the file that actually runs training. It wires together the environment,
the actor-critic network, and the PPO update function into one big loop:
collect experience -> update the policy -> log stats -> repeat.

Run this directly to start training:
    python train.py

Hit Ctrl+C at any time to stop early -- it'll save the model before exiting
so you don't lose all that sweet training progress.
"""

import os
import time

import numpy as np
import torch

from config import CONFIG
from environment import BipedalWalkerEnv
from agent import ActorCritic, RolloutBuffer, ppo_update


def train(config):
    """
    The main training loop. Sets up the env + agent, then alternates between
    collecting rollouts and running PPO updates until we've used up the
    training budget (or the user hits Ctrl+C).
    """

    # ══════════════════════════════════════════════════════════════════════
    #  SETUP
    #  Create the environment, neural network, optimizer, and all the
    #  bookkeeping variables we need to track training progress.
    # ══════════════════════════════════════════════════════════════════════

    # the environment wraps the physics sim + reward computation
    env = BipedalWalkerEnv(config)

    # check if we have a GPU -- probably not on most laptops but hey,
    # maybe you're running this on a lab machine with a beefy card
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the actor-critic network (the walker's brain) and put it
    # on the right device
    agent = ActorCritic(
        env.observation_size,
        env.action_size,
        hidden_size=config["hidden_size"],
    ).to(device)

    # Adam optimizer -- the standard go-to for deep RL.  It adapts the
    # learning rate per-parameter which helps when actor and critic
    # gradients have very different magnitudes.
    optimizer = torch.optim.Adam(agent.parameters(), lr=config["learning_rate"])

    # the rollout buffer stores transitions while we collect experience,
    # then we dump the whole thing into PPO for the update
    buffer = RolloutBuffer()

    # ── tracking variables ────────────────────────────────────────────
    # these let us monitor training progress and decide when to save
    total_timesteps = 0                    # total env steps so far
    episode_count = 0                      # completed episodes
    update_count = 0                       # PPO updates performed

    episode_rewards = []                   # final reward for each episode
    episode_lengths = []                   # step count for each episode
    best_mean_reward = float("-inf")       # best rolling average we've seen

    # stuff for the episode that's currently in progress
    current_ep_reward = 0.0
    current_ep_length = 0

    # ── startup banner ────────────────────────────────────────────────
    # print out all the key hyperparameters so we know exactly what
    # configuration we're training with (super handy when comparing runs)
    print("=" * 60)
    print("  PPO Bipedal Walker Training")
    print("=" * 60)
    print(f"  Device:              {device}")
    print(f"  Observation size:    {env.observation_size}")
    print(f"  Action size:         {env.action_size}")
    print(f"  Hidden size:         {config['hidden_size']}")
    print(f"  Learning rate:       {config['learning_rate']}")
    print(f"  Rollout steps:       {config['rollout_steps']}")
    print(f"  PPO epochs:          {config['ppo_epochs']}")
    print(f"  Minibatch size:      {config['minibatch_size']}")
    print(f"  Gamma:               {config['gamma']}")
    print(f"  GAE lambda:          {config['gae_lambda']}")
    print(f"  Clip epsilon:        {config['clip_epsilon']}")
    print(f"  Total timesteps:     {config['total_timesteps']:,}")
    print(f"  Save path:           {config['save_path']}")
    print("=" * 60)
    print()

    # start the clock so we can report total wall-clock time later
    start_time = time.time()

    # get the very first observation from the environment
    obs = env.reset()

    # ══════════════════════════════════════════════════════════════════════
    #  MAIN TRAINING LOOP
    #  This is where the magic happens. We repeat:
    #    1) collect a rollout (run the policy in the env)
    #    2) update the policy with PPO
    #    3) log stats and save if we've improved
    #  until we've hit the total training timestep budget.
    # ══════════════════════════════════════════════════════════════════════

    try:
        while total_timesteps < config["total_timesteps"]:

            # ──────────────────────────────────────────────────────────
            #  ROLLOUT COLLECTION PHASE
            #  Run the current policy for rollout_steps steps, storing
            #  every (obs, action, reward, ...) transition in the buffer.
            #  This is on-policy data -- we'll only use it once for the
            #  PPO update, then throw it away and collect fresh data.
            # ──────────────────────────────────────────────────────────

            for _step in range(config["rollout_steps"]):

                # numpy obs -> pytorch tensor so the network can process it
                obs_tensor = torch.tensor(
                    obs, dtype=torch.float32, device=device
                )

                # ask the policy what to do -- no_grad because we're just
                # collecting data here, not computing gradients
                with torch.no_grad():
                    action, log_prob, value = agent.get_action(obs_tensor)

                # environment wants numpy actions, not tensors
                action_np = action.cpu().numpy()

                # take a step in the environment and see what happens
                next_obs, reward, done, info = env.step(action_np)

                # stash this transition in the buffer for PPO to chew on later
                buffer.add(
                    obs=obs,
                    action=action_np,
                    log_prob=log_prob.cpu().item(),
                    reward=reward,
                    done=done,
                    value=value.cpu().item(),
                )

                # keep a running tally of the current episode's reward and length
                current_ep_reward += reward
                current_ep_length += 1
                total_timesteps += 1

                if done:
                    # episode finished! record the final stats
                    episode_rewards.append(current_ep_reward)
                    episode_lengths.append(current_ep_length)
                    episode_count += 1

                    # zero out the trackers for the next episode
                    current_ep_reward = 0.0
                    current_ep_length = 0

                    # reset the environment to start a fresh episode
                    obs = env.reset()
                else:
                    # episode still going, just move to the next state
                    obs = next_obs

            # ──────────────────────────────────────────────────────────
            #  PPO UPDATE PHASE
            #  We've collected a full rollout of experience. Now feed it
            #  into the PPO algorithm to improve the policy.  ppo_update
            #  handles GAE advantage computation internally (it bootstraps
            #  using the value of the last observation in the buffer).
            # ──────────────────────────────────────────────────────────

            update_stats = ppo_update(agent, optimizer, buffer, config)

            # done with this batch of data -- clear the buffer for the
            # next rollout
            buffer.clear()
            update_count += 1

            # ──────────────────────────────────────────────────────────
            #  LOGGING PHASE
            #  Every log_interval updates, print a nice status block so
            #  we can watch training progress.  Not too often (would be
            #  spammy) but often enough to see trends.
            # ──────────────────────────────────────────────────────────

            if update_count % config["log_interval"] == 0:

                # how far through the training budget are we?
                pct = 100.0 * total_timesteps / config["total_timesteps"]

                # compute rolling mean reward/length over the last 20
                # completed episodes (or fewer if we haven't done 20 yet)
                n_recent = 20
                if len(episode_rewards) > 0:
                    recent_rewards = episode_rewards[-n_recent:]
                    mean_reward = np.mean(recent_rewards)
                    recent_lengths = episode_lengths[-n_recent:]
                    mean_length = np.mean(recent_lengths)
                else:
                    # no completed episodes yet -- possible if episodes
                    # are really long relative to rollout_steps
                    mean_reward = float("nan")
                    mean_length = float("nan")

                # print the stats in a clean, scannable format
                print(
                    f"=== Update {update_count} | "
                    f"{total_timesteps:,} / {config['total_timesteps']:,} "
                    f"steps ({pct:.1f}%) ==="
                )
                print(
                    f"  Episodes: {episode_count} | "
                    f"Mean reward (last "
                    f"{min(n_recent, len(episode_rewards))}): "
                    f"{mean_reward:.1f} | Best: {best_mean_reward:.1f}"
                )
                print(f"  Mean ep length: {mean_length:.1f} steps")
                print(
                    f"  Policy loss: {update_stats['policy_loss']:.4f} | "
                    f"Value loss: {update_stats['value_loss']:.4f} | "
                    f"Entropy: {update_stats['entropy']:.3f} | "
                    f"KL: {update_stats['approx_kl']:.4f}"
                )
                print()

            # ──────────────────────────────────────────────────────────
            #  MODEL SAVING
            #  Save the model whenever the rolling mean reward beats our
            #  previous best. That way we always have the best-performing
            #  checkpoint on disk, even if training degrades later.
            # ──────────────────────────────────────────────────────────

            if len(episode_rewards) > 0:
                recent_mean = np.mean(episode_rewards[-20:])

                if recent_mean > best_mean_reward:
                    best_mean_reward = recent_mean
                    _save_model(agent, config)
                    print(
                        f"  >> New best mean reward: "
                        f"{best_mean_reward:.1f} -- model saved!"
                    )
                    print()

    except KeyboardInterrupt:
        # ──────────────────────────────────────────────────────────────
        #  CTRL+C HANDLER
        #  The user wants to stop early. Save the model so all that
        #  training time doesn't go to waste, then exit gracefully.
        # ──────────────────────────────────────────────────────────────
        elapsed = time.time() - start_time
        print(f"\n\nCtrl+C detected after {elapsed / 60:.1f} minutes!")
        print("Saving model before exit...")
        _save_model(agent, config)
        print(f"Model saved to: {config['save_path']}")
        print(
            f"Made it through {total_timesteps:,} timesteps "
            f"and {episode_count} episodes."
        )
        return

    # ══════════════════════════════════════════════════════════════════════
    #  END OF TRAINING
    #  We made it through the entire training budget! Save the final model
    #  and print a summary of how things went.
    # ══════════════════════════════════════════════════════════════════════

    # save the final model (even if it's not a new best, we want the
    # latest weights on disk)
    _save_model(agent, config)

    elapsed = time.time() - start_time

    print("=" * 60)
    print("  Training Complete!")
    print("=" * 60)
    print(f"  Total episodes:      {episode_count}")
    print(f"  Total timesteps:     {total_timesteps:,}")
    if len(episode_rewards) > 0:
        print(f"  Final mean reward:   {np.mean(episode_rewards[-20:]):.1f}")
    print(f"  Best mean reward:    {best_mean_reward:.1f}")
    print(f"  Training time:       {elapsed / 60:.1f} minutes")
    print(f"  Model saved to:      {config['save_path']}")
    print("=" * 60)


def _save_model(agent, config):
    """
    Save the actor-critic network weights to disk.
    Creates the checkpoint directory if it doesn't exist yet.
    """
    save_path = config["save_path"]
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save(agent.state_dict(), save_path)


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
#  Just import the config and kick off training.  The try/except around
#  the call is a safety net in case KeyboardInterrupt fires during setup
#  (before train()'s own handler is active).
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    from config import CONFIG

    try:
        train(CONFIG)
    except KeyboardInterrupt:
        # if the interrupt slips past train()'s handler somehow
        print("\nInterrupted! Bye!")
