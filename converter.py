import numpy as np
import torch
import itertools


CASE_CONFIG = {
    # sub, gen, load, line
    5: (5, 2, 3, 8),
    14: (14, 5, 11, 20), # (14, 6, 11, 20)
    118: (118, 62, 99, 186),
    36: (36, 22, 37, 59)
}


class graphGoalConverter:
    def __init__(self, env, mask, mask_hi, danger, device, rule='c'):
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        self.mask = mask
        self.mask_hi = mask_hi
        self.danger = danger
        self.rule = rule
        self.device = device
        self.thermal_limit_under400 = torch.from_numpy(env._thermal_limit_a < 400)
        self.init_obs_converter()
        self.init_action_converter()

    def init_obs_converter(self):        
        self.idx = self.obs_space.shape
        self.pp = np.arange(sum(self.idx[:6]),sum(self.idx[:7])) 
        self.lp = np.arange(sum(self.idx[:9]),sum(self.idx[:10]))
        self.op = np.arange(sum(self.idx[:12]),sum(self.idx[:13]))
        self.ep = np.arange(sum(self.idx[:16]),sum(self.idx[:17]))
        self.rho = np.arange(sum(self.idx[:20]),sum(self.idx[:21]))
        self.topo = np.arange(sum(self.idx[:23]),sum(self.idx[:24]))
        self.main = np.arange(sum(self.idx[:26]),sum(self.idx[:27]))
        self.over = np.arange(sum(self.idx[:22]),sum(self.idx[:23]))
        
        # parse substation info
        self.subs = [{'e':[], 'o':[], 'g':[], 'l':[]} for _ in range(self.action_space.n_sub)]
        for gen_id, sub_id in enumerate(self.action_space.gen_to_subid):
            self.subs[sub_id]['g'].append(gen_id)
        for load_id, sub_id in enumerate(self.action_space.load_to_subid):
            self.subs[sub_id]['l'].append(load_id)
        for or_id, sub_id in enumerate(self.action_space.line_or_to_subid):
            self.subs[sub_id]['o'].append(or_id)
        for ex_id, sub_id in enumerate(self.action_space.line_ex_to_subid):
            self.subs[sub_id]['e'].append(ex_id)
        
        self.sub_to_topos = []  # [0]: [0, 1, 2], [1]: [3, 4, 5, 6, 7, 8]
        for sub_info in self.subs:
            a = []
            for i in sub_info['e']:
                a.append(self.action_space.line_ex_pos_topo_vect[i])
            for i in sub_info['o']:
                a.append(self.action_space.line_or_pos_topo_vect[i])
            for i in sub_info['g']:
                a.append(self.action_space.gen_pos_topo_vect[i])
            for i in sub_info['l']:
                a.append(self.action_space.load_pos_topo_vect[i])
            self.sub_to_topos.append(torch.LongTensor(a))

        # split topology over sub_id
        self.sub_to_topo_begin, self.sub_to_topo_end = [], []
        idx = 0
        for num_topo in self.action_space.sub_info:
            self.sub_to_topo_begin.append(idx)
            idx += num_topo
            self.sub_to_topo_end.append(idx)
        dim_topo = self.idx[-7]
        self.last_topo = np.ones(dim_topo, dtype=np.int32)
        self.n_feature = 5

    def convert_obs(self, o):
        # o.shape : (B, O)
        # output (Batch, Node, Feature)
        length = self.action_space.dim_topo # N
        
        # active power p
        p_ = torch.zeros(o.size(0), length).to(o.device)    # (B, N)
        p_[..., self.action_space.gen_pos_topo_vect] = o[...,  self.pp]
        p_[..., self.action_space.load_pos_topo_vect] = o[..., self.lp]
        p_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.op]
        p_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.ep]

        # rho (powerline usage ratio)
        rho_ = torch.zeros(o.size(0), length).to(o.device)
        rho_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.rho]
        rho_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.rho]

        # whether each line is in danger
        danger_ = torch.zeros(o.size(0), length).to(o.device)
        danger = ((o[...,self.rho] >= self.danger-0.05) & self.thermal_limit_under400.to(o.device)) | (o[...,self.rho] >= self.danger)
        danger_[..., self.action_space.line_or_pos_topo_vect] = danger.float()
        danger_[..., self.action_space.line_ex_pos_topo_vect] = danger.float()      

        # whether overflow occurs in each powerline
        over_ = torch.zeros(o.size(0), length).to(o.device)
        over_[..., self.action_space.line_or_pos_topo_vect] = o[..., self.over]/3
        over_[..., self.action_space.line_ex_pos_topo_vect] = o[..., self.over]/3

        # whether each powerline is in maintenance
        main_ = torch.zeros(o.size(0), length).to(o.device)
        temp = torch.zeros_like(o[..., self.main])
        temp[o[..., self.main]==0] = 1
        main_[..., self.action_space.line_or_pos_topo_vect] = temp
        main_[..., self.action_space.line_ex_pos_topo_vect] = temp

        # current bus assignment
        topo_ = torch.clamp(o[..., self.topo] - 1, -1)

        state = torch.stack([p_, rho_, danger_, over_, main_], dim=2) # B, N, F
        return state, topo_.unsqueeze(-1)
  
    def init_action_converter(self):
        self.sorted_sub = list(range(self.action_space.n_sub))
        self.sub_mask = []  # mask for parsing actionable topology
        self.psubs = []     # actionable substation IDs
        self.masked_sub_to_topo_begin = []
        self.masked_sub_to_topo_end = []
        idx = 0
        for i, num_topo in enumerate(self.action_space.sub_info):
            if num_topo > self.mask and num_topo < self.mask_hi:
                self.sub_mask.extend(
                    [j for j in range(self.sub_to_topo_begin[i]+1, self.sub_to_topo_end[i])])
                self.psubs.append(i)
                self.masked_sub_to_topo_begin.append(idx)
                idx += num_topo-1
                self.masked_sub_to_topo_end.append(idx)

            else:
                self.masked_sub_to_topo_begin.append(-1)
                self.masked_sub_to_topo_end.append(-1)
        self.n = len(self.sub_mask)

        if self.rule == 'f':
            if self.obs_space.n_sub == 5:
                self.masked_sorted_sub = [4, 0, 1, 3, 2]
            elif self.obs_space.n_sub == 14:
                self.masked_sorted_sub = [13, 5, 0, 12, 9, 6, 10, 1, 11, 3, 4, 7, 2]
            elif self.obs_space.n_sub == 36: # mask = 5
                self.masked_sorted_sub = [9, 33, 29, 7, 21, 1, 4, 23, 16, 26, 35]
                if self.mask == 4:
                    self.masked_sorted_sub = [35, 23, 9, 33, 4, 28, 1, 32, 13, 21, 26, 29, 16, 22, 7, 27]
        else:
            if self.obs_space.n_sub == 5:
                self.masked_sorted_sub = [0, 3, 2, 1, 4]
            elif self.obs_space.n_sub == 14:
                self.masked_sorted_sub = [5, 1, 3, 4, 2, 12, 0, 11, 13, 10, 9, 6, 7]
            elif self.obs_space.n_sub == 36: # mask = 5
                self.masked_sorted_sub = [16, 23, 21, 26, 33, 29, 35, 9, 7, 4, 1]
                if self.mask == 4:
                    self.masked_sorted_sub += [22, 27, 28, 32, 13]

        # powerlines which are not controllable by bus assignment action
        self.lonely_lines = set()
        for i in range(self.obs_space.n_line):
            if (self.obs_space.line_or_to_subid[i] not in self.psubs) \
               and (self.obs_space.line_ex_to_subid[i] not in self.psubs):
                self.lonely_lines.add(i)
        self.lonely_lines = list(self.lonely_lines) 
        print('Lonely line', len(self.lonely_lines), self.lonely_lines)
        print('Masked sorted topology', len(self.masked_sorted_sub), self.masked_sorted_sub)

    def inspect_act(self, sub_id, goal, topo_vect):
        # Correct illegal action collect original ids
        exs = self.subs[sub_id]['e']
        ors = self.subs[sub_id]['o']
        lines = exs + ors   # [line_id0, line_id1, line_id2, ...]
        
        # minimal prevention of isolation
        line_idx = len(lines) - 1 
        if (goal[:line_idx] == 1).all() * (goal[line_idx:] != 1).any():
            goal = torch.ones_like(goal)
        
        if torch.is_tensor(goal): goal = goal.numpy()
        beg = self.masked_sub_to_topo_begin[sub_id]
        end = self.masked_sub_to_topo_end[sub_id]
        already_same = np.all(goal == topo_vect[self.sub_mask][beg:end])
        return goal, already_same

    def convert_act(self, sub_id, new_topo, obs=None):
        new_topo = [1] + new_topo.tolist()
        act = self.action_space({'set_bus': {'substations_id': [(sub_id, new_topo)]}})
        return act

    def plan_act(self, goal, topo_vect, sub_order_score=None):
        # Compare obs.topo_vect and goal, then parse partial order from whole topological sort
        topo_vect = torch.LongTensor(topo_vect)
        topo_vect = topo_vect[self.sub_mask]
        targets = []
        goal = goal.squeeze(0).cpu() + 1

        if sub_order_score is None:
            sub_order = self.masked_sorted_sub
        else:
            sub_order = [i[0] for i in sorted(list(zip(self.masked_sorted_sub, sub_order_score.tolist())),
                        key=lambda x: -x[1])]

        for sub_id in sub_order:
            beg = self.masked_sub_to_topo_begin[sub_id]
            end = self.masked_sub_to_topo_end[sub_id]
            topo = topo_vect[beg:end]
            new_topo = goal[beg:end]
            if torch.any(new_topo != topo).item():
                targets.append((sub_id, new_topo))

        # Assign sequentially actions from the goal
        plan = [(sub_id, new_topo) for sub_id, new_topo in targets]
        return plan

    def heuristic_order(self, obs, low_actions):
        if len(low_actions) == 0:
            return []
        rhos = []
        for item in low_actions:
            sub_id = item[0]
            lines = self.subs[sub_id]['e'] + self.subs[sub_id]['o']
            rho = obs.rho[lines].copy()
            rho[rho==0] = 3
            rho_max = rho.max()
            rho_mean = rho.mean()
            rhos.append((rho_max, rho_mean))
        order = sorted(zip(low_actions, rhos), key=lambda x: (-x[1][0], -x[1][1]))
        return list(list(zip(*order))[0])
