# test_grid_env.py

from grid_env import GridEnvironment
import numpy as np
import random

# Define grid, bombs, risky, and safe zones
grid_size = (10, 10)
bombs = [(1, 2), (2, 3), (2, 8), (4, 4), (4, 6), (3, 7), (4, 2), (5, 1), (7, 7), (6, 2), (8, 3)]
risky_zones = [(1, 4), (3, 3), (3, 9), (4, 0), (4, 5), (6, 4), (7, 4), (7, 5), (8, 4), (9, 2)]
safe_zones = [(1, 1), (1, 6), (5, 5), (5, 7), (7, 1), (7, 8), (8, 5)]
goal_pos = (grid_size[0] - 1, grid_size[1] - 1)

# Initialize the environment
env = GridEnvironment(grid_size=grid_size, bombs=bombs, risky_zones=risky_zones, safe_zones=safe_zones)

# Reset the environment
# state = env.reset()
# print("Initial State:", state)

# done = False
# step_counter = 0

# while not done and step_counter < 5:  # Limit steps for testing
#     action = env.action_space.sample()  # Random action for testing
#     state, reward, done, _ = env.step(env.hsh[action])
    
#     # Display the step info
#     print(f"Step {step_counter}: Action={env.hsh[action]}, State={state}, Reward={reward}, Done={done}")
#     step_counter += 1
    
#     # Render the environment (optional, for visualization)
#     env.render()

# Simulate consecutive risky steps

# Set the agent's position manually near risky zones for controlled testing
env.agent_pos = (4, 1)  # A position in a risky zone
env.risky_history = []  # Mock previous risky steps
print("Initial State:", env.agent_pos)
env.render()

print("Testing consecutive risky penalty...")
for _ in range(30):
    # Move to a neighboring risky zone to simulate consecutive risky condition
    # action = env.hsh[random.randint(0, 3)]  # Example action to stay in the risky area
    action = env.hsh[int(input("Enter action (0=up, 1=down, 2=left, 3=right): "))]  # Manual input for testing
    print(f"Action={action}")
    state, reward, done, _ = env.step(action)

    print(env.risky_history)
    
    # Print the results to observe the penalty for consecutive risky steps
    print(f"Action={action}, State={state}, Reward={reward}, Done={done}, Consecutive Risky={len(env.risky_history) >= 3}")
    env.render()
