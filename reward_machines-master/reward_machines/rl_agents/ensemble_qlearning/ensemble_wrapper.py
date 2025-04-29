class EnsembleAgent:
    def __init__(self, rm1, agent1, rm2, agent2, alpha=0.5):
        self.rm1, self.agent1 = rm1, agent1
        self.rm2, self.agent2 = rm2, agent2
        self.alpha = alpha

    def reset(self, env):
        self.rm1.current_state = self.rm1.initial_state
        self.rm2.current_state = self.rm2.initial_state

    def select_action(self, s, available_actions):
        u1 = self.rm1.current_state
        u2 = self.rm2.current_state
        best_a, best_q = None, -float('inf')
        for a in available_actions:
            q1 = self.agent1.Q.get((tuple(s), u1, a), self.agent1.q_init)
            q2 = self.agent2.Q.get((tuple(s), u2, a), self.agent2.q_init)
            q_ens = (1-self.alpha)*q1 + self.alpha*q2
            if q_ens > best_q:
                best_q, best_a = q_ens, a
        return best_a

    def observe(self, s, a, s_next, event):
        # advance both RMs
        _, _ = self.rm1.step(event, a)
        _, _ = self.rm2.step(event, a)
