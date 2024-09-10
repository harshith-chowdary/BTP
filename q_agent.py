import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gym
from gym import spaces
import random

goal_reward = 1000

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
        self.last_risky_step = 0
        self.last_unsaved_risky_step = float('inf')
        self.step_counter = 0  # total steps taken
        
        self.action_space = spaces.Discrete(4)  # 0: up, 1: down, 2: left, 3: right
        self.observation_space = spaces.Box(low=0, high=grid_size[0] - 1, shape=(2,), dtype=np.int32)
        
        self.hsh = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}

    def reset(self, origin = 0):
        if origin:
            self.agent_pos = (0, 0)
        else:
            self.agent_pos = (random.randint(0, 9), random.randint(0, 9))
            
        while self.agent_pos in self.obstacles or self.agent_pos == self.goal_pos:
            self.agent_pos = (random.randint(0, 9), random.randint(0, 9))
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
        
        action = self.hsh[action]
        
        if action == 'up':
            new_pos[0] += 1
        elif action == 'down':
            new_pos[0] -= 1
        elif action == 'left':
            new_pos[1] -= 1
        elif action == 'right':
            new_pos[1] += 1

        new_pos = tuple(new_pos)
        
        # if self.is_valid_move(new_pos):
        #     self.agent_pos = new_pos
        #     reward = -1  # Default penalty for moving
        #     done = False

        #     # Assign penalties and rewards
        #     if new_pos in self.risky_zones:
        #         reward = -30  # Penalty for risky zones
        #     elif new_pos in self.safe_zones:
        #         reward = 50  # Reward for visiting safe zones
        #     elif new_pos == self.goal_pos:
        #         done = True
        #         reward = 10000  # Large reward for reaching the goal

        #     return new_pos, reward, done, {}
        
        if self.is_valid_move(new_pos):
            self.agent_pos = new_pos
            reward = -1
            done = False

            # # Temporal Safety Constraints
            if new_pos in self.risky_zones:
                if self.step_counter - self.last_risky_step < 3: 
                    reward = -100
                    
                self.last_risky_step = self.step_counter
                self.last_unsaved_risky_step = min(self.last_unsaved_risky_step, self.step_counter)

            if new_pos not in self.safe_zones and self.step_counter - self.last_unsaved_risky_step > 5:
                reward = -100
                
            if new_pos in self.safe_zones:
                self.last_unsaved_risky_step = float('inf')

            if new_pos == self.goal_pos:  # Reached goal
                done = True
                reward = goal_reward
                
            return new_pos, reward, done, {}
        else:
            return self.agent_pos, -100, False, {}  # collision penalty

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

# Q-learning agent class
class QLearningAgent:
    def __init__(self, action_space, grid_size, alpha=0.1, gamma=0.99, epsilon=0.1):
        self.action_space = action_space
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table initialized to zeros
        self.q_table = np.zeros((*grid_size, action_space.n))

    def choose_action(self, state):
        if np.random.rand() < self.epsilon:
            return self.action_space.sample()  # Exploration
        else:
            return np.argmax(self.q_table[state[0], state[1]])  # Exploitation

    def update_q_table(self, state, action, reward, next_state, done):
        best_next_action = np.argmax(self.q_table[next_state[0], next_state[1]])
        td_target = reward + (0 if done else self.gamma * self.q_table[next_state[0], next_state[1], best_next_action])
        td_error = td_target - self.q_table[state[0], state[1], action]
        self.q_table[state[0], state[1], action] += self.alpha * td_error

if __name__ == "__main__":
    # Define the grid with obstacles, risky zones, and safe zones
    obstacles = [(1, 2), (2, 3), (2, 8), (4, 4), (4, 6), (3, 7), (4, 2), (5, 1), (7, 7), (6, 2), (8, 3)]
    risky_zones = [(1, 4), (3, 3), (3, 9), (4, 0), (4, 5), (6, 4), (7, 5), (8, 4), (9, 2)]
    safe_zones = [(1, 1), (1, 6), (5, 5), (5, 7), (7, 1), (7, 8), (8, 5)]

    grid_size = (10, 10)

    # Create environment and agent
    env = GridEnvironment(grid_size=grid_size, obstacles=obstacles, risky_zones=risky_zones, safe_zones=safe_zones)
    agent = QLearningAgent(env.action_space, grid_size)

    episodes = 100000
    for episode in range(episodes):
        state = env.reset()
        done = False
        
        cumulative_reward = 0

        while not done and cumulative_reward > -1000:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.update_q_table(state, action, reward, next_state, done)
            state = next_state
            
            # cumulative_reward += reward

        # Debugging output for every 100 episodes
        if (episode + 1) % 1000 == 0: 
            print(f"Episode {episode + 1} completed")

    # After training, visualize the agent's performance
    state = env.reset(1)
    done = False
    
    total_cost = 0
    
    while not done:
        action = agent.choose_action(state)
        state, reward, done, _ = env.step(action)
        
        total_cost += reward
        print(f"Action: {action}, State: {state}, Reward: {reward}, Done: {done}")
        
        env.render()
        
    print(f"\nTotal Cost: {goal_reward - total_cost}")
