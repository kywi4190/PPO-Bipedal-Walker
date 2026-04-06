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

import math
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

    # ── diagnostic tracking (for reward analysis) ──────────────
    diag_ep_breakdowns = []    # per-episode reward component sums
    diag_ep_distances = []     # torso_x at end - start_x
    diag_ep_falls = []         # True if episode ended in fall
    diag_ep_end_angles = []    # torso angle at episode end
    diag_ep_action_mags = []   # mean |action| over the episode
    current_ep_breakdown = {}
    current_ep_action_mag_sum = 0.0
    current_ep_action_steps = 0

    # ── snapshot tracking (for end-of-training summary) ────────────
    # capture diagnostics at 5 evenly-spaced moments across training
    n_snapshots = 10 if config["total_timesteps"] >= 10_000_000 else 5
    snapshot_interval = config["total_timesteps"] // n_snapshots
    next_snapshot_at = snapshot_interval
    training_snapshots = []

    # ── startup banner ────────────────────────────────────────────────
    # print out all the key hyperparameters so we know exactly what
    # configuration we're training with (super handy when comparing runs)
    print("=" * 60)
    print("  PPO Bipedal Walker Training")
    print("=" * 60)
    print(f"  Profile:             {config.get('profile', 'custom')}")
    print(f"  Device:              {device}")
    if device.type == "cuda":
        print(f"  GPU:                 {torch.cuda.get_device_name(0)}")
        gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  GPU memory:          {gpu_mem:.1f} GB")
    print(f"  Observation size:    {env.observation_size}")
    print(f"  Action size:         {env.action_size}")
    print(f"  Hidden size:         {config['hidden_size']}")
    print(f"  Learning rate:       {config['learning_rate']} (cosine annealing)")
    print(f"  Rollout steps:       {config['rollout_steps']}")
    print(f"  PPO epochs:          {config['ppo_epochs']}")
    print(f"  Minibatch size:      {config['minibatch_size']}")
    print(f"  Gamma:               {config['gamma']}")
    print(f"  GAE lambda:          {config['gae_lambda']}")
    print(f"  Clip epsilon:        {config['clip_epsilon']}")
    print(f"  Total timesteps:     {config['total_timesteps']:,}")
    print(f"  Save path:           {config['save_path']}")
    print("=" * 60)

    # ── reward sanity check ─────────────────────────────────────────
    ms = config["max_episode_steps"]
    _ab = config["alive_bonus"]
    _frs = config["forward_reward_scale"]
    _vbs = config["velocity_bonus_scale"]
    _fp = config["fall_penalty"]
    _total_alive = _ab * (ms + (ms - 1) / 2.0)
    _est_A = _total_alive
    _wv = 3.0
    _sv = _wv * (1.0 + _vbs * _wv)
    _est_B = _total_alive + _sv * _frs * ms
    _fn = 50
    _alive_C = _ab * (_fn + (_fn - 1) * _fn / (2.0 * ms))
    _est_C = _alive_C + 40.0 + _fp
    print()
    print("  -- Reward Sanity Check --")
    print(f"  Full-ep alive (compounding): {_total_alive:.0f}")
    print(f"  Fall penalty:                {_fp:.0f}")
    print(f"  A (stand {ms} steps):       {_est_A:+.0f}")
    print(f"  B (walk 3px/s, {ms} steps): {_est_B:+.0f}")
    print(f"  C (lurch 100px, fall@50):    {_est_C:+.0f}")
    if _est_B > _est_A > _est_C:
        print(f"  OK: B({_est_B:.0f}) >> A({_est_A:.0f}) >> C({_est_C:.0f})")
    else:
        print("  WARNING: ordering violated! Check reward magnitudes.")
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
                with torch.inference_mode():
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

                # accumulate diagnostics for the current episode
                for key, val in info["reward_breakdown"].items():
                    current_ep_breakdown[key] = (
                        current_ep_breakdown.get(key, 0.0) + val
                    )
                current_ep_action_mag_sum += float(
                    np.mean(np.abs(action_np))
                )
                current_ep_action_steps += 1

                if done:
                    # episode finished! record the final stats
                    episode_rewards.append(current_ep_reward)
                    episode_lengths.append(current_ep_length)
                    episode_count += 1

                    # record diagnostic data for this completed episode
                    diag_ep_breakdowns.append(dict(current_ep_breakdown))
                    diag_ep_distances.append(
                        info["torso_x"] - config["torso_start_x"]
                    )
                    diag_ep_falls.append(info.get("fell", True))
                    diag_ep_end_angles.append(
                        info.get("torso_angle", 0.0)
                    )
                    diag_ep_action_mags.append(
                        current_ep_action_mag_sum
                        / max(1, current_ep_action_steps)
                    )

                    # zero out the trackers for the next episode
                    current_ep_reward = 0.0
                    current_ep_length = 0
                    current_ep_breakdown = {}
                    current_ep_action_mag_sum = 0.0
                    current_ep_action_steps = 0

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

            # pass current obs (s_{T+1}) so GAE can bootstrap off the
            # correct next-state value, not the last stored observation
            update_stats = ppo_update(agent, optimizer, buffer, config, last_obs=obs)

            # done with this batch of data -- clear the buffer for the
            # next rollout
            buffer.clear()
            update_count += 1

            # ── learning rate schedule ─────────────────────────────────
            # cosine annealing: keeps LR higher in mid-training, smooth decay to 0
            frac = total_timesteps / config["total_timesteps"]
            lr = config["learning_rate"] * 0.5 * (1.0 + math.cos(math.pi * frac))
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr

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
                    f"KL: {update_stats['approx_kl']:.4f} | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

                # ── diagnostics block ──────────────────────────────────
                n_diag = min(n_recent, len(diag_ep_breakdowns))
                if n_diag > 0:
                    avg_bd = {}
                    for bd in diag_ep_breakdowns[-n_diag:]:
                        for key, val in bd.items():
                            avg_bd[key] = avg_bd.get(key, 0.0) + val
                    for key in avg_bd:
                        avg_bd[key] /= n_diag

                    bd_parts = []
                    for key in [
                        "forward_reward", "alive_bonus",
                        "energy_penalty", "fall_penalty",
                    ]:
                        if key in avg_bd:
                            short = (
                                key.replace("_reward", "")
                                .replace("_penalty", "")
                                .replace("_bonus", "")
                            )
                            bd_parts.append(
                                f"{short}: {avg_bd[key]:+.2f}"
                            )

                    recent_dists = diag_ep_distances[-n_diag:]
                    recent_falls = diag_ep_falls[-n_diag:]
                    recent_angles = diag_ep_end_angles[-n_diag:]
                    recent_amags = diag_ep_action_mags[-n_diag:]
                    fall_pct = (
                        100.0 * sum(recent_falls) / len(recent_falls)
                    )

                    print(
                        f"  --- Reward Breakdown "
                        f"(last {n_diag} eps) ---"
                    )
                    print(f"  {' | '.join(bd_parts)}")
                    print(
                        f"  --- Behavior "
                        f"(last {n_diag} eps) ---"
                    )
                    print(
                        f"  dist: {np.mean(recent_dists):.1f} | "
                        f"steps: {mean_length:.0f}"
                        f"/{config['max_episode_steps']} | "
                        f"falls: {fall_pct:.0f}% | "
                        f"end_angle: "
                        f"{np.mean(np.abs(recent_angles)):.2f}rad"
                        f" | action_mag: "
                        f"{np.mean(recent_amags):.2f}"
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

            # ──────────────────────────────────────────────────────────
            #  SNAPSHOT CAPTURE
            #  At 5 evenly-spaced moments, snapshot the current
            #  diagnostics for the end-of-training summary.
            # ──────────────────────────────────────────────────────────

            if (total_timesteps >= next_snapshot_at
                    and len(diag_ep_breakdowns) > 0):
                n_snap = min(20, len(diag_ep_breakdowns))
                snap_bd = {}
                for bd in diag_ep_breakdowns[-n_snap:]:
                    for key, val in bd.items():
                        snap_bd[key] = snap_bd.get(key, 0.0) + val
                for key in snap_bd:
                    snap_bd[key] /= n_snap

                snap_falls = diag_ep_falls[-n_snap:]
                snap_angles = diag_ep_end_angles[-n_snap:]
                training_snapshots.append({
                    "timestep": total_timesteps,
                    "mean_reward": float(
                        np.mean(episode_rewards[-n_snap:])
                    ),
                    "mean_length": float(
                        np.mean(episode_lengths[-n_snap:])
                    ),
                    "breakdown": dict(snap_bd),
                    "dist": float(np.mean(diag_ep_distances[-n_snap:])),
                    "fall_pct": 100.0 * sum(snap_falls) / len(snap_falls),
                    "end_angle": float(
                        np.mean(np.abs(snap_angles))
                    ),
                    "action_mag": float(
                        np.mean(diag_ep_action_mags[-n_snap:])
                    ),
                })
                next_snapshot_at += snapshot_interval

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
        _print_training_summary(training_snapshots, config)
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
    _print_training_summary(training_snapshots, config)


def _save_model(agent, config):
    """
    Save the actor-critic network weights to disk.
    Creates the checkpoint directory if it doesn't exist yet.
    """
    save_path = config["save_path"]
    save_dir = os.path.dirname(save_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    torch.save({
        "model_state_dict": agent.state_dict(),
        "hidden_size": config["hidden_size"],
    }, save_path)


def _print_training_summary(snapshots, config):
    """
    Print a compact training summary from snapshots captured at 5
    evenly-spaced moments. Designed for reward shaping analysis.
    """
    if not snapshots:
        print("\n  (No snapshots captured — training may have been too short)")
        return

    print()
    print("=" * 70)
    print("  Training Summary  ")
    print("=" * 70)

    # header
    max_steps = config["total_timesteps"]
    print(f"  config: max_steps={max_steps:,} | max_ep={config['max_episode_steps']}"
          f" | gamma={config['gamma']} | lr={config['learning_rate']}")
    print(f"  reward weights: fwd={config['forward_reward_scale']}"
          f" alive={config['alive_bonus']}"
          f" energy={config['energy_penalty_scale']}"
          f" vel_bonus={config['velocity_bonus_scale']}"
          f" fall={config['fall_penalty']}")
    print("-" * 70)

    for i, s in enumerate(snapshots):
        pct = 100.0 * s["timestep"] / max_steps
        print(f"  [{i+1}/5] step {s['timestep']:,} ({pct:.0f}%)"
              f" | reward: {s['mean_reward']:.1f}"
              f" | ep_len: {s['mean_length']:.0f}/{config['max_episode_steps']}"
              f" | falls: {s['fall_pct']:.0f}%")
        bd = s["breakdown"]
        parts = []
        for key in ["forward_reward", "alive_bonus",
                     "energy_penalty", "fall_penalty"]:
            if key in bd:
                short = (key.replace("_reward", "")
                         .replace("_penalty", "")
                         .replace("_bonus", ""))
                parts.append(f"{short}:{bd[key]:+.1f}")
        print(f"    components: {' | '.join(parts)}")
        print(f"    dist:{s['dist']:.1f}"
              f" | end_angle:{s['end_angle']:.2f}rad"
              f" | action_mag:{s['action_mag']:.2f}")

    print("=" * 70)


# ══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
#  Just import the config and kick off training.  The try/except around
#  the call is a safety net in case KeyboardInterrupt fires during setup
#  (before train()'s own handler is active).
# ══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse
    from config import get_config

    parser = argparse.ArgumentParser(description="Train PPO bipedal walker")
    parser.add_argument(
        "--profile", type=str, default="short",
        choices=["short", "medium", "long", "xlong", "max"],
        help="training profile (default: short)",
    )
    # backward-compat aliases
    parser.add_argument("--long", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--xlong", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--max", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.xlong:
        profile = "xlong"
    elif args.long:
        profile = "long"
    elif args.max:
        profile = "max"
    else:
        profile = args.profile
    config = get_config(profile)

    try:
        train(config)
    except KeyboardInterrupt:
        # if the interrupt slips past train()'s handler somehow
        print("\nInterrupted! Bye!")
