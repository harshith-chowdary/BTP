import pickle

class TabularQModel:
    def __init__(self, Q, actions, q_init, epsilon, gamma, lr):
        self.Q       = Q
        self.actions = actions
        self.q_init  = q_init
        self.epsilon = epsilon
        self.gamma   = gamma
        self.lr      = lr
        # no RM state kept here; RewardMachineWrapper in env holds that

    def save(self, save_path):
        # you can also save any hyperparams if you like
        with open(save_path, 'wb') as f:
            pickle.dump({
                'Q':       self.Q,
                'actions': self.actions,
                'q_init':  self.q_init,
                'epsilon': self.epsilon,
                'gamma':   self.gamma,
                'lr':      self.lr
            }, f)
        print(f"TabularQModel saved to {save_path!r}")

    @classmethod
    def load(cls, save_path):
        with open(save_path, 'rb') as f:
            data = pickle.load(f)
        print(f"TabularQModel loaded from {save_path!r}")
        return cls(data['Q'], data['actions'], data['q_init'],
                   data['epsilon'], data['gamma'], data['lr'])

    def step(self, obs, S=None, M=None):
        """
        The baselines play loop will call `model.step(obs, S=rm_state, M=dones)`:
        - obs: raw env obs (e.g. a vector)
        - S: RM state(s); here assume M is a vector of zeros/ones.
        We just need to pick an action index from self.actions.
        """
        # convert obs to your Q-key format:
        s = tuple(obs)  if isinstance(obs, (list, np.ndarray)) else obs
        # if you also keyed on RM state in Q, unpack M (dones) to get rm_state
        # but in your implementation using RewardMachineWrapper the RM state is
        # part of the env.observation, so `s` already encodes it.
        #
        # just do greedy for play (epsilon=0)
        qrow = self.Q.get(s, None)
        if qrow is None:
            # unseen state → initialize
            qrow = {a: self.q_init for a in self.actions}
            self.Q[s] = qrow

        # pick best action
        best_a = max(self.actions, key=lambda a: qrow[a])
        return [best_a], None, S, None

