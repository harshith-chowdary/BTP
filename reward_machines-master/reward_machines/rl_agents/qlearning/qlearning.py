"""
Q-Learning based method
"""

import random, time
from baselines import logger
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



def get_qmax(Q,s,actions,q_init):
    if s not in Q:
        Q[s] = dict([(a,q_init) for a in actions])
    return max(Q[s].values())

def get_best_action(Q,s,actions,q_init):
    qmax = get_qmax(Q,s,actions,q_init)
    best = [a for a in actions if Q[s][a] == qmax]
    return random.choice(best)

def learn(env,
          network=None,
          seed=None,
          lr=0.1,
          total_timesteps=100000,
          epsilon=0.1,
          print_freq=10000,
          gamma=0.9,
          q_init=2.0,
          use_crm=False,
          use_rs=False):
    """Train a tabular q-learning model.

    Parameters
    -------
    env: gym.Env
        environment to train on
    network: string or a function
        This is just a placeholder to be consistent with the openai-baselines interface, but we don't really use state-approximation in tabular q-learning
    seed: int or None
        prng seed. The runs with the same seed "should" give the same results. If None, no seeding is used.
    lr: float
        learning rate
    total_timesteps: int
        number of env steps to optimizer for
    epsilon: float
        epsilon-greedy exploration
    print_freq: int
        how often to print out training progress
        set to None to disable printing
    gamma: float
        discount factor
    q_init: float
        initial q-value for unseen states
    use_crm: bool
        use counterfactual experience to train the policy
    use_rs: bool
        use reward shaping
    """

    # Running Q-Learning
    reward_total = 0
    step = 0
    num_episodes = 0
    Q = {}
    actions = list(range(env.action_space.n))

    while step < total_timesteps:
        s = tuple(env.reset())
        if s not in Q: Q[s] = dict([(a,q_init) for a in actions])
        while True:
            # Selecting and executing the action
            a = random.choice(actions) if random.random() < epsilon else get_best_action(Q,s,actions,q_init)
            sn, r, done, info = env.step(a)
            sn = tuple(sn)

            # Updating the q-values
            experiences = []
            if use_crm:
                # Adding counterfactual experience (this will alrady include shaped rewards if use_rs=True)
                for _s,_a,_r,_sn,_done in info["crm-experience"]:
                    experiences.append((tuple(_s),_a,_r,tuple(_sn),_done))
            elif use_rs:
                # Include only the current experince but shape the reward
                experiences = [(s,a,info["rs-reward"],sn,done)]
            else:
                # Include only the current experience (standard q-learning)
                experiences = [(s,a,r,sn,done)]

            for _s,_a,_r,_sn,_done in experiences:
                if _s not in Q: Q[_s] = dict([(b,q_init) for b in actions])
                if _done: _delta = _r - Q[_s][_a]
                else:     _delta = _r + gamma*get_qmax(Q,_sn,actions,q_init) - Q[_s][_a]
                Q[_s][_a] += lr*_delta

            # moving to the next state
            reward_total += r
            step += 1
            if step%print_freq == 0:
                logger.record_tabular("steps", step)
                logger.record_tabular("episodes", num_episodes)
                logger.record_tabular("total reward", reward_total)
                logger.dump_tabular()
                reward_total = 0
            if done:
                num_episodes += 1
                break
            s = sn

    # returning the model
    return TabularQModel(
        Q       = Q,
        actions = actions,
        q_init  = q_init,
        epsilon = epsilon,
        gamma   = gamma,
        lr      = lr
    )
