# grid_env.py

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import gym
from gym import spaces
import random
from reward_machine import RewardMachine

goal_reward = 1000

class GridEnvironment(gym.Env):
    def __init__(self, grid_size=(10, 10), bombs=[], risky_zones=[], safe_zones=[]):
        super(GridEnvironment, self).__init__()
        self.grid_size = grid_size
        self.bombs = bombs
        self.risky_zones = risky_zones
        self.safe_zones = safe_zones
        self.agent_pos = (0, 0)
        self.goal_pos = (grid_size[0] - 1, grid_size[1] - 1)
        
        # Initialize the Reward Machine with Büchi automaton
        self.reward_machine = RewardMachine()
        
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=grid_size[0] - 1, shape=(2,), dtype=np.int32)
        
        self.hsh = {0: 'up', 1: 'down', 2: 'left', 3: 'right'}
        self.risky_history = []  # Track history of recent moves to check for consecutive risky

    def reset(self, origin=0):
        self.agent_pos = (0, 0) if origin else self._random_position()
        while self.agent_pos in self.bombs or self.agent_pos == self.goal_pos:
            self.agent_pos = self._random_position()
        
        self.reward_machine.reset()
        self.risky_history = []  # Reset risky history on environment reset
        return self.agent_pos

    def _random_position(self):
        return (random.randint(0, 9), random.randint(0, 9))

    def is_valid_move(self, new_pos):
        return (
            new_pos not in self.bombs 
            and 0 <= new_pos[0] < self.grid_size[0] 
            and 0 <= new_pos[1] < self.grid_size[1]
        )

    def get_env_state(self, pos):
        """Identify the state properties for the reward machine."""
        is_bomb = pos in self.bombs
        is_risky = pos in self.risky_zones
        is_safe = pos in self.safe_zones
        is_goal = pos == self.goal_pos
        
        # Update risky history
        if is_risky:
            self.risky_history.append(True)
            if len(self.risky_history) > 3:
                self.risky_history.pop(0)
        else:
            self.risky_history = []
        
        # Check consecutive risky condition (last three moves were risky)
        consecutive_risky = len(self.risky_history) == 3 and all(self.risky_history)

        print('consecutive_risky:', consecutive_risky)
        
        return {
            'bomb': is_bomb,
            'risky': is_risky,
            'safe': is_safe,
            'goal': is_goal,
            'consecutive_risky': consecutive_risky
        }

    def step(self, action):
        new_pos = self._move_agent(action)
        
        if self.is_valid_move(new_pos):
            self.agent_pos = new_pos
            env_state = self.get_env_state(new_pos)
            reward, done = self.reward_machine.get_reward(env_state)

            return new_pos, reward, done, {}
        else:
            print("Stepped on Bomb!")
            return self.agent_pos, -300, False, {}  # collision penalty

    def _move_agent(self, action):
        new_pos = list(self.agent_pos)
        if action == 'up': new_pos[0] += 1
        elif action == 'down': new_pos[0] -= 1
        elif action == 'left': new_pos[1] -= 1
        elif action == 'right': new_pos[1] += 1
        return tuple(new_pos)

    def render(self):
        grid = np.zeros(self.grid_size)
        for bomb in self.bombs:
            grid[bomb] = -1
        for risky in self.risky_zones:
            grid[risky] = -0.5
        for safe in self.safe_zones:
            grid[safe] = 1

        grid[self.agent_pos] = 0.5
        grid[self.goal_pos] = 1.5
        
        cmap = mcolors.ListedColormap(['grey', 'red', 'lightyellow', 'blue', 'lightgreen', 'green'])
        bounds = [-1.25, -0.75, -0.25, 0.25, 0.75, 1.25, 1.75]
        norm = mcolors.BoundaryNorm(bounds, cmap.N)

        img = plt.imshow(grid, cmap=cmap, norm=norm)
        plt.gca().invert_yaxis()
        cbar = plt.colorbar(img, ticks=[-1, -0.5, 0, 0.5, 1, 1.5])
        cbar.set_ticklabels(['bombs', 'Risky Zones', 'Neutral', 'Agent', 'Safe Zones', 'Goal'])
        plt.show()

