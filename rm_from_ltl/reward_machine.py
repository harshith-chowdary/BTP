# reward_machine.py

import re
from time import sleep

class RewardMachine:
    def __init__(self, file="../bucchi_formal.txt"):
        self.states = {}  # Stores states and their transitions as parsed from the file
        self.current_state = None
        self.goal_reward = 1000
        self.acceptance_states = set()
        
        # Load automaton from the structured never claim format
        self._load_automaton(file)

        print("Reward Machine Initialized")

    def _load_automaton(self, file_path):
        """ Parse the structured automaton format from the provided file. """
        with open(file_path, 'r') as file:
            lines = file.readlines()
        
        current_state = None
        for line in lines:
            line = line.strip()

            # Detect start of a new state (e.g., "T0_init" or "accept_S1")
            if line.startswith("T") or line.startswith("accept"):
                state_name = line.split(":")[0]
                state_id_str = state_name.split("_")[1]
                
                # Map initial state "init" to state 0, otherwise convert to int
                if state_id_str == "init":
                    state_id = 0
                else:
                    state_id = int(state_id_str[1:])  # Skip the initial letter for numeric ID
                
                # Mark acceptance states
                if "accept" in state_name:
                    self.acceptance_states.add(state_id)
                self.states[state_id] = {'transitions': []}
                current_state = state_id

            elif line.startswith("::") and current_state is not None:
                # Extract the condition and target state using a flexible regex
                match = re.search(r":: (.*?) -> goto (T\d+_init|accept_S\d+|T\d+_S\d+)", line)
                if match:
                    condition, target = match.groups()
                    
                    # Parse target state ID
                    target_state_str = target.split("_")[1]
                    if target_state_str == "init":
                        target_state_id = 0
                    else:
                        target_state_id = int(target_state_str[1:])
                    
                    # Append transition with parsed condition and target state
                    self.states[current_state]['transitions'].append((self._parse_condition(condition), target_state_id))

        for state_id, state_data in self.states.items():
            print(f"State {state_id}")
            for condition, target_state in state_data['transitions']:
                print(f"  {condition} -> {target_state}")

        # Set initial state to be the first defined state (e.g., "T0_init" -> state 0)
        self.current_state = 0

    def _parse_condition(self, condition):
        """ Convert HOA transition condition to Python lambda. """
        # Replace keywords with `env_state` references
        condition = condition.replace("bomb", "env_state['bomb']")
        condition = condition.replace("risky", "env_state['risky']")
        condition = condition.replace("goal", "env_state['goal']")
        condition = condition.replace("safe", "env_state['safe']")
        condition = condition.replace("normal", "not env_state['risky'] and not env_state['bomb'] and not env_state['goal'] and not env_state['safe']")
        
        # Replace logical operators correctly
        condition = condition.replace("&&", " and ")
        condition = condition.replace("||", " or ")
        condition = condition.replace("!", " not ")
        
        # Return a lambda function that evaluates this condition against env_state
        return lambda env_state: eval(condition)

    def transition(self, env_state):
        """ Transition to the next state based on the environment state and the automaton's transition logic. """
        if self.current_state is None:
            raise ValueError("Automaton not initialized correctly.")

        print(f"Current State: {self.current_state}")  # Log current state
        # Find the appropriate transition based on conditions
        transitions = self.states[self.current_state]['transitions']
        for condition, target_state in transitions:
            if condition(env_state):
                self.current_state = target_state
                print(f"Transitioned to State: {self.current_state}")  # Log state transition
                return self._get_reward(env_state)

        print("No valid transition found")  # Log invalid transition
        return -100  # Default penalty if no valid transition found

    def _get_reward(self, env_state):
        print(f"Env State: {env_state}")
        """ Determine reward based on the current state and environment context. """
        if self.current_state in self.acceptance_states:
            return self.goal_reward if env_state['goal'] else 50 if env_state['safe'] else -1
        elif env_state['bomb']:
            return -1000
        elif env_state['risky']:
            return -200 if env_state['consecutive_risky'] else -30
        return -1

    def get_reward(self, env_state):
        """ Obtain the reward based on the current environment state using the Büchi automaton. """
        reward = self.transition(env_state)
        done = self.current_state in self.acceptance_states and env_state.get("goal", False)
        return reward, done

    def reset(self):
        """ Reset the reward machine to the initial state. """
        self.current_state = 0

# rm = RewardMachine()
# print(rm.states[0])
