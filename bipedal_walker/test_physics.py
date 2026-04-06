from physics import PhysicsWorld
from config import CONFIG
import numpy as np

world = PhysicsWorld(CONFIG)

state = world.reset()
print(f"State shape: {state.shape}")

actions = np.zeros(6)
new_state = world.step(actions)

print(f"New state shape: {new_state.shape}")
print(f"Torso x: {world.ragdoll.get_torso_x():.1f}")
print("Physics engine works!")