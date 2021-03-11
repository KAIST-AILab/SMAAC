import numpy as np
from grid2op.Reward.BaseReward import BaseReward


class LossReward(BaseReward):
    def __init__(self):
        BaseReward.__init__(self)
        self.reward_min = -1.0
        self.reward_illegal = -0.5
        self.reward_max = 1.0

    def initialize(self, env):
        pass

    def __call__(self, action, env, has_error, is_done, is_illegal, is_ambiguous):
        if has_error:
           if is_illegal or is_ambiguous:
               return self.reward_illegal
           elif is_done:
               return self.reward_min
        gen_p, *_ = env.backend.generators_info()
        load_p, *_ = env.backend.loads_info()
        reward = (load_p.sum() / gen_p.sum() * 10. - 9.) * 0.1 # avg ~ 0.01
        return reward