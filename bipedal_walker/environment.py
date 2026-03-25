"""
environment.py - Gym-style environment wrapper for the bipedal walker.

This sits between the PPO agent and the raw physics simulation.  The agent
calls reset() to start an episode, then step(action) in a loop until it
gets done=True.  We handle reward computation, episode termination, and
packaging everything into the (obs, reward, done, info) tuple that RL
algorithms expect.

No actual gym dependency -- we just follow the same interface pattern.
"""

import math
import numpy as np

from config import CONFIG
from physics import PhysicsWorld


class BipedalWalkerEnv:
    """
    Wraps PhysicsWorld to give a standard RL environment interface.

    Observation: 13 floats from the ragdoll's state vector
    Action:      4 floats in [-1, 1], one per joint motor
    """

    def __init__(self, config=CONFIG):
        self.config = config
        self.world = PhysicsWorld(config)

        # observation is 13 floats (see physics.py RagdollBody.get_state)
        self.observation_size = 13
        # 4 motors: left_hip, right_hip, left_knee, right_knee
        self.action_size = 4

        # episode tracking
        self.step_count = 0
        self.prev_torso_x = config["torso_start_x"]

    def reset(self):
        """
        Reset the physics world and all tracking variables.
        Returns the initial observation as a numpy array.
        """
        obs = self.world.reset()
        self.step_count = 0
        self.prev_torso_x = self.config["torso_start_x"]
        return obs

    def step(self, action):
        """
        Take one step in the environment.

        Args:
            action: numpy array of 4 floats, will be clipped to [-1, 1]

        Returns:
            obs:    numpy array of 13 floats (new state)
            reward: float (total reward for this step)
            done:   bool (episode over?)
            info:   dict with debugging goodies
        """
        # clip actions so the agent can't go crazy
        action = np.clip(action, -1.0, 1.0)

        # advance the simulation
        obs = self.world.step(action)

        # figure out what happened
        torso_x = self.world.ragdoll.get_torso_x()
        torso_angle = self.world.ragdoll.get_torso_angle()
        fallen = self.world.ragdoll.is_fallen()

        # ── reward computation ───────────────────────────────────────
        # each component is tracked separately so we can debug what the
        # agent is actually optimizing for

        # primary signal: move to the right.  delta-x since last step
        # times the scale factor.  positive = good, negative = bad.
        forward_reward = (
            (torso_x - self.prev_torso_x) * self.config["forward_reward_scale"]
        )

        # small cookie for staying alive -- adds up over long episodes
        # and gives a baseline incentive to not just faceplant immediately
        alive_bonus = self.config["alive_bonus"]

        # reward for keeping the torso upright.  cos(0) = 1.0 (perfect),
        # cos(60°) ≈ 0.5, cos(90°) = 0 (horizontal).  smooth gradient
        # that nudges the agent toward vertical without being too harsh.
        upright_reward = (
            self.config["upright_reward_scale"] * math.cos(torso_angle)
        )

        # penalize large actions to discourage spastic flailing.
        # sum of squared actions keeps it smooth and differentiable.
        energy_penalty = (
            -self.config["energy_penalty_scale"] * float(np.sum(action ** 2))
        )

        # big negative reward for falling -- this is the "don't do that" signal
        fall_penalty = self.config["fall_penalty"] if fallen else 0.0

        total_reward = (
            forward_reward + alive_bonus + upright_reward
            + energy_penalty + fall_penalty
        )

        # ── episode termination ──────────────────────────────────────
        self.step_count += 1
        done = fallen or self.step_count >= self.config["max_episode_steps"]

        # save torso x for next step's velocity calculation
        self.prev_torso_x = torso_x

        # ── info dict for debugging / logging ────────────────────────
        info = {
            "torso_x": torso_x,
            "episode_steps": self.step_count,
            "reward_breakdown": {
                "forward_reward": forward_reward,
                "alive_bonus": alive_bonus,
                "upright_reward": upright_reward,
                "energy_penalty": energy_penalty,
                "fall_penalty": fall_penalty,
            },
        }

        return obs, total_reward, done, info

    def get_body_positions(self):
        """Passthrough to physics world for rendering."""
        return self.world.get_body_positions()
