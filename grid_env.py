import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gym
from gym import spaces

class GridEnvironment(gym.Env):
    def __init__(
        self, 
        grid_size = (10, 10), 
        obstacles = [(1, 2), (2, 3), (2, 8), (4, 4), (4, 6), (3, 7), (4, 2), (5, 1), (7, 7), (6, 2), (8, 3)],  # Block some optimal paths
        risky_zones = [(1, 4), (3, 3), (3, 9), (4, 0), (4, 5), (6, 4), (7, 5), (8, 4), (9, 2)],  # Risky zones placed near potential paths
        safe_zones = [(1, 1), (1, 6), (5, 5), (5, 7), (7, 1), (7, 8), (8, 5)]  # Safe zones in strategic positions to incentivize exploration
    ):
        super(GridEnvironment, self).__init__()
        self.grid_size = grid_size
        self.obstacles = obstacles  # list of obstacle coordinates
        self.risky_zones = risky_zones  # list of risky zone coordinates
        self.safe_zones = safe_zones  # list of safe zone coordinates
        self.agent_pos = (0, 0)  # start at bottom-left corner (flipped)
        self.goal_pos = (grid_size[0] - 1, grid_size[1] - 1)  # goal at top-right corner
        self.risky_counter = 0  # track consecutive steps in risky zones
        self.safe_visit_counter = 0  # track steps since last visit to safe zone
        self.step_counter = 0  # total steps taken
        
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=0, high=grid_size[0] - 1, shape=(2,), dtype=np.int32)
        
        self.hsh = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    def reset(self):
        self.agent_pos = (0, 0)
        self.risky_counter = 0
        self.safe_visit_counter = 0
        self.step_counter = 0
        return self.agent_pos

    def is_valid_move(self, new_pos):
        if new_pos in self.obstacles or not (0 <= new_pos[0] < self.grid_size[0] and 0 <= new_pos[1] < self.grid_size[1]):
            return False
        return True

    def step(self, action):
        self.step_counter += 1
        new_pos = list(self.agent_pos)
        if action == 'up':
            new_pos[0] += 1
        elif action == 'down':
            new_pos[0] -= 1
        elif action == 'left':
            new_pos[1] -= 1
        elif action == 'right':
            new_pos[1] += 1

        new_pos = tuple(new_pos)
        if self.is_valid_move(new_pos):
            self.agent_pos = new_pos
            reward = 0
            done = False

            # Temporal Safety Constraints
            if new_pos in self.risky_zones:
                self.risky_counter += 1
                if self.risky_counter > 3:  # If more than 3 consecutive steps in risky zones, penalty
                    reward = -5
            else:
                self.risky_counter = 0  # Reset if agent leaves risky zone

            self.safe_visit_counter += 1
            if self.safe_visit_counter > 5 and new_pos not in self.safe_zones:  # If safe zone not visited in 5 steps
                reward = -3  # Penalty for not visiting safe zone in time

            if new_pos in self.safe_zones:
                self.safe_visit_counter = 0  # Reset if agent visits a safe zone

            done = new_pos == self.goal_pos  # Reached goal
            return new_pos, reward, done, {}
        else:
            return self.agent_pos, -10, False, {}  # collision penalty

    def render(self):
        grid = np.zeros(self.grid_size)
        for obstacle in self.obstacles:
            grid[obstacle] = -1  # obstacles
        for risky in self.risky_zones:
            grid[risky] = -0.5  # risky zones
        for safe in self.safe_zones:
            grid[safe] = 1  # safe zones

        grid[self.agent_pos] = 0.5 # agent's current position
        grid[self.goal_pos] = 1.5  # goal
        
        # Create a custom color map
        cmap = mcolors.ListedColormap(['grey', 'red', 'lightyellow', 'blue', 'lightgreen', 'green'])
        bounds = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        img = plt.imshow(grid, cmap=cmap, norm=norm)
        plt.gca().invert_yaxis()  # Flip the vertical axis to match Cartesian plane
        # plt.colorbar(ticks=[-1, -0.5, 0, 0.5, 1, 1.5], label='Zone Legend')
        
        cbar = plt.colorbar(img, ticks=[-1, -0.5, 0, 0.5, 1, 1.5])
        
        # Set the colorbar labels (tick labels)
        cbar.set_ticks([-1, -0.5, 0, 0.5, 1, 1.5])
        cbar.set_ticklabels(['Obstacles', 'Risky Zones', 'Nuetral', 'Agent', 'Safe Zones', 'Goal'])
        
        plt.show()

# Strategically place obstacles, risky zones, and safe zones
obstacles = [(1, 2), (2, 3), (2, 8), (4, 4), (4, 6), (3, 7), (4, 2), (5, 1), (7, 7), (6, 2), (8, 3)]  # Block some optimal paths
risky_zones = [(1, 4), (3, 3), (3, 9), (4, 0), (4, 5), (6, 4), (7, 5), (8, 4), (9, 2)]  # Risky zones placed near potential paths
safe_zones = [(1, 1), (1, 6), (5, 5), (5, 7), (7, 1), (7, 8), (8, 5)]  # Safe zones in strategic positions to incentivize exploration

grid_size = (10, 10)

# Usage Example
if __name__ == "__main__":
    env = GridEnvironment(grid_size=grid_size, obstacles=obstacles, risky_zones=risky_zones, safe_zones=safe_zones)

    state = env.reset()
    print("Initial State:", state)

    done = False
    while not done:
        action = env.action_space.sample()  # Sample a random action
        state, reward, done, _ = env.step(env.hsh[action])
        print(f"Action: {env.hsh[action]}, State: {state}, Reward: {reward}, Done: {done}")
        env.render()

exit()

# Create environment
env = GridEnvironment(grid_size=grid_size, obstacles=obstacles, risky_zones=risky_zones, safe_zones=safe_zones)
env.render()

# Simulate steps (Example)
state, reward, done = env.step('right')  # Move right
env.render()
print(state, reward, done)
state, reward, done = env.step('up')  # Move up
env.render()
print(state, reward, done)

exit(0)

# safety policy
'''
    ltl = G (¬obstacle) ∧ G (risky_zone -> (X ¬risky_zone ∨ XX ¬risky_zone)) ∧ G (¬risky_zone U (risky_zone ∧ F[5] (safe_zone ∨ goal_zone))) ∧ FG goal_zone
    
    Explanation of the Combined LTL Formula:
    G (¬obstacle): The agent must always avoid obstacles.
    G (risky_zone -> (X ¬risky_zone ∨ XX ¬risky_zone)): If the agent enters a risky zone, it must leave it within 2 moves.
    G (¬risky_zone U (risky_zone ∧ F[5] (safe_zone ∨ goal_zone))): After leaving a risky zone, the agent must visit a safe zone or the goal within 5 moves.
    FG goal_zone: The agent must eventually reach the goal.
    
    Notes:
    X represents "in the next state" in LTL.
    F represents "eventually" in LTL, while F[5] limits the "eventually" to within 5 steps.
    U is the "until" operator, which ensures that one condition holds until another becomes true.
'''