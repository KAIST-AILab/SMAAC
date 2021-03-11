import os
import random
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from models import DoubleSoftQ, EncoderLayer, Actor
from util import ReplayBuffer
from converter import graphGoalConverter
from grid2op.Agent import BaseAgent


class Agent(BaseAgent):
    def __init__(self, env, **kwargs):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.obs_space = env.observation_space
        self.action_space = env.action_space
        super(Agent, self).__init__(env.action_space)
        mask = kwargs.get('mask', 2)
        mask_hi = kwargs.get('mask_hi', 19)
        self.rule = kwargs.get('rule', 'c')
        self.danger = kwargs.get('danger', 0.9)
        self.bus_thres = kwargs.get('threshold', 0.1)
        self.max_low_len = kwargs.get('max_low_len', 19)
        self.converter = graphGoalConverter(env, mask, mask_hi, self.danger, self.device, self.rule)
        self.thermal_limit = env._thermal_limit_a
        self.convert_obs = self.converter.convert_obs
        self.action_dim = self.converter.n
        self.order_dim = len(self.converter.masked_sorted_sub)
        self.node_num = env.dim_topo
        self.delay_step = 2
        self.update_step = 0
        self.k_step = 1
        self.nheads = kwargs.get('head_number', 8)
        self.target_update = kwargs.get('target_update', 1)
        self.hard_target = kwargs.get('hard_target', False)
        self.use_order = (self.rule == 'o')
        
        self.gamma = kwargs.get('gamma', 0.99)
        self.tau = kwargs.get('tau', 1e-3)
        self.dropout = kwargs.get('dropout', 0.)
        self.memlen = kwargs.get('memlen', int(1e5))
        self.batch_size = kwargs.get('batch_size', 128)
        self.update_start = self.batch_size * 8
        self.actor_lr = kwargs.get('actor_lr', 5e-5)
        self.critic_lr = kwargs.get('critic_lr', 5e-5)
        self.embed_lr = kwargs.get('embed_lr', 5e-5)
        self.alpha_lr = kwargs.get('alpha_lr', 5e-5)

        self.state_dim = kwargs.get('state_dim', 128)
        self.n_history = kwargs.get('n_history', 6)
        self.input_dim = self.converter.n_feature * self.n_history
        
        print(f'N: {self.node_num}, O: {self.input_dim}, S: {self.state_dim}, A: {self.action_dim}, ({self.order_dim})')
        print(kwargs)
        self.emb = EncoderLayer(self.input_dim, self.state_dim, self.nheads, self.node_num, self.dropout).to(self.device)
        self.temb = EncoderLayer(self.input_dim, self.state_dim, self.nheads, self.node_num, self.dropout).to(self.device)
        self.Q = DoubleSoftQ(self.state_dim, self.nheads, self.node_num, self.action_dim,
                        self.use_order, self.order_dim, self.dropout).to(self.device)
        self.tQ = DoubleSoftQ(self.state_dim, self.nheads, self.node_num, self.action_dim,
                        self.use_order, self.order_dim, self.dropout).to(self.device)
        self.actor = Actor(self.state_dim, self.nheads, self.node_num, self.action_dim,
                        self.use_order, self.order_dim, self.dropout).to(self.device)
        
        # copy parameters
        self.tQ.load_state_dict(self.Q.state_dict())
        self.temb.load_state_dict(self.emb.state_dict())

        # entropy
        self.target_entropy = -self.action_dim  * 3 if not self.use_order else -3 * (self.action_dim + self.order_dim)
        self.log_alpha = torch.FloatTensor([-3]).to(self.device)
        self.log_alpha.requires_grad = True

        # optimizers
        self.Q.optimizer = optim.Adam(self.Q.parameters(), lr=self.critic_lr)
        self.actor.optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.emb.optimizer = optim.Adam(self.emb.parameters(), lr=self.embed_lr)
        self.alpha_optim = optim.Adam([self.log_alpha], lr=self.alpha_lr)
        
        self.memory = ReplayBuffer(max_size=self.memlen)
        self.Q.eval()
        self.tQ.eval()
        self.emb.eval()
        self.temb.eval()
        self.actor.eval()
    
    def is_safe(self, obs):
        for ratio, limit in zip(obs.rho, self.thermal_limit):
            # Seperate big line and small line
            if (limit < 400.00 and ratio >= self.danger-0.05) or ratio >= self.danger:
                return False
        return True

    def load_mean_std(self, mean, std):
        self.state_mean = mean
        self.state_std = std.masked_fill(std < 1e-5, 1.)
        self.state_mean[0,sum(self.obs_space.shape[:20]):] = 0
        self.state_std[0,sum(self.action_space.shape[:20]):] = 1

    def state_normalize(self, s):
        s = (s - self.state_mean) / self.state_std
        return s

    def reset(self, obs):
        self.converter.last_topo = np.ones(self.node_num, dtype=int)
        self.topo = None
        self.goal = None
        self.goal_list = []
        self.low_len = -1
        self.adj = None
        self.stacked_obs = []
        self.low_actions = []
        self.save = False

    def cache_stat(self):
        cache = {
            'last_topo': self.converter.last_topo,
            'topo': self.topo,
            'goal': self.goal,
            'goal_list': self.goal_list,
            'low_len': self.low_len,
            'adj': self.adj,
            'stacked_obs': self.stacked_obs,
            'low_actions': self.low_actions,
            'save': self.save,
        }
        return cache

    def load_cache_stat(self, cache):
        self.converter.last_topo = cache['last_topo']
        self.topo = cache['topo']
        self.goal = cache['goal']
        self.goal_list = cache['goal_list']
        self.low_len = cache['low_len']
        self.adj = cache['adj']
        self.stacked_obs = cache['stacked_obs']
        self.low_actions = cache['low_actions']
        self.save = cache['save']

    def hash_goal(self, goal):
        hashed = ''
        for i in goal.view(-1):
            hashed += str(int(i.item()))
        return hashed

    def stack_obs(self, obs):
        obs_vect = obs.to_vect()
        obs_vect = torch.FloatTensor(obs_vect).unsqueeze(0)
        obs_vect, self.topo = self.convert_obs(self.state_normalize(obs_vect))
        if len(self.stacked_obs) == 0:
            for _ in range(self.n_history):                
                self.stacked_obs.append(obs_vect)
        else:
            self.stacked_obs.pop(0)
            self.stacked_obs.append(obs_vect)
        self.adj = (torch.FloatTensor(obs.connectivity_matrix()) + torch.eye(int(obs.dim_topo))).to(self.device)
        self.converter.last_topo = np.where(obs.topo_vect==-1, self.converter.last_topo, obs.topo_vect)

    def reconnect_line(self, obs):
        # if the agent can reconnect powerline not included in controllable substation, return action
        # otherwise, return None
        dislines = np.where(obs.line_status == False)[0]
        for i in dislines:
            act = None
            if obs.time_next_maintenance[i] != 0 and i in self.converter.lonely_lines:
                sub_or = self.action_space.line_or_to_subid[i]
                sub_ex = self.action_space.line_ex_to_subid[i]
                if obs.time_before_cooldown_sub[sub_or] == 0:
                    act = self.action_space({'set_bus': {'lines_or_id': [(i, 1)]}})
                if obs.time_before_cooldown_sub[sub_ex] == 0:
                    act = self.action_space({'set_bus': {'lines_ex_id': [(i, 1)]}})
                if obs.time_before_cooldown_line[i] == 0:
                    status = self.action_space.get_change_line_status_vect()
                    status[i] = True
                    act = self.action_space({'change_line_status': status})
                if act is not None:                          
                    return act
        return None

    def get_current_state(self):
        return torch.cat(self.stacked_obs + [self.topo], dim=-1)

    def act(self, obs, reward, done):
        sample = (reward is None)
        self.stack_obs(obs)
        is_safe = self.is_safe(obs)
        self.save = False        
        
        # reconnect powerline when the powerline in uncontrollable substations is disconnected
        if False in obs.line_status:
            act = self.reconnect_line(obs)
            if act is not None:
                return act
        
        # generate goal if it is initial or previous goal has been reached
        if self.goal is None or (not is_safe and self.low_len == -1):
            goal, bus_goal, low_actions, order, Q1, Q2 = self.generate_goal(sample, obs, not sample)
            if len(low_actions) == 0:
                act = self.action_space()
                if self.goal is None:
                    self.update_goal(goal, bus_goal, low_actions, order, Q1, Q2)
                return self.action_space()
            self.update_goal(goal, bus_goal, low_actions, order, Q1, Q2)

        act = self.pick_low_action(obs)
        return act
    
    def pick_low_action(self, obs):
        # Safe and there is no queued low actions, just do nothing
        if self.is_safe(obs) and self.low_len == -1:
            act = self.action_space()
            return act

        # optimize low actions every step
        self.low_actions = self.optimize_low_actions(obs, self.low_actions)
        self.low_len += 1

        # queue has been empty after optimization. just do nothing
        if len(self.low_actions) == 0:
            act = self.action_space()
            self.low_len = -1
        
        # normally execute low action from low actions queue
        else:
            sub_id, new_topo = self.low_actions.pop(0)[:2]
            act = self.converter.convert_act(sub_id, new_topo, obs.topo_vect)
                
        # When it meets maximum low action execution time, log and reset
        if self.max_low_len <= self.low_len:
            self.low_len = -1
        return act

    def high_act(self, stacked_state, adj, sample=True):
        order, Q1, Q2 = None, 0, 0
        with torch.no_grad():
            # stacked_state # B, N, F
            stacked_t, stacked_x = stacked_state[..., -1:], stacked_state[..., :-1]
            emb_input = stacked_x 
            state = self.emb(emb_input, adj).detach()
            actor_input = [state, stacked_t.squeeze(-1)]
            if sample:
                action, std = self.actor.sample(actor_input, adj)
                if self.use_order:
                    action, order = action
                critic_input = action
                Q1, Q2 = self.Q(state, critic_input, adj, order)
                Q1, Q2 = Q1.detach()[0].item(), Q2.detach()[0].item()
                if self.use_order:
                    std, order_std = std
            else:
                action = self.actor.mean(actor_input, adj)
                if self.use_order:
                    action, order = action
        if order is not None: order = order.detach().cpu()
        return action.detach().cpu(), order, Q1, Q2
                                                 
    def make_candidate_goal(self, stacked_state, adj, sample, obs):
        goal, order, Q1, Q2 = self.high_act(stacked_state, adj, sample)
        bus_goal = torch.zeros_like(goal).long()
        bus_goal[goal > self.bus_thres] = 1
        low_actions = self.converter.plan_act(bus_goal, obs.topo_vect, order[0] if order is not None else None)
        low_actions = self.optimize_low_actions(obs, low_actions)
        return goal, bus_goal, low_actions, order, Q1, Q2

    def generate_goal(self, sample, obs, nosave=False):
        stacked_state = self.get_current_state().to(self.device)
        adj = self.adj.unsqueeze(0)
        goal, bus_goal, low_actions, order, Q1, Q2 = self.make_candidate_goal(stacked_state, adj, sample, obs)
        return goal, bus_goal, low_actions, order, Q1, Q2
    
    def update_goal(self, goal, bus_goal, low_actions, order=None, Q1=0, Q2=0):
        self.order = order
        self.goal = goal
        self.bus_goal = bus_goal
        self.low_actions = low_actions
        self.low_len = 0
        self.save = True
        self.goal_list.append(self.hash_goal(bus_goal))

    def optimize_low_actions(self, obs, low_actions):
        # remove overlapped action
        optimized = []
        cooldown_list = obs.time_before_cooldown_sub
        if self.max_low_len != 1 and self.rule == 'c':
            low_actions = self.converter.heuristic_order(obs, low_actions)
        for low_act in low_actions:
            sub_id, sub_goal = low_act[:2]
            sub_goal, same = self.converter.inspect_act(sub_id, sub_goal, obs.topo_vect)
            if not same:
                optimized.append((sub_id, sub_goal, cooldown_list[sub_id]))
        
        # sort by cooldown_sub
        if self.max_low_len != 1 and self.rule != 'o':
            optimized = sorted(optimized, key=lambda x: x[2])
        
        # if current action has cooldown, then discard
        if len(optimized) > 0 and optimized[0][2] > 0:
            optimized = []
        return optimized
 
    def append_sample(self, s, m, a, r, s2, m2, d, order):
        if self.use_order:
            self.memory.append((s, m, a, r, s2, m2, int(d), order))
        else:
            self.memory.append((s, m, a, r, s2, m2, int(d)))

    def unpack_batch(self, batch):
        if self.use_order:
            states, adj, actions, rewards, states2, adj2, dones, orders = list(zip(*batch))
            orders = torch.cat(orders, 0)
        else:
            states, adj, actions, rewards, states2, adj2, dones = list(zip(*batch))
        states = torch.cat(states, 0)
        states2 = torch.cat(states2, 0)
        adj = torch.stack(adj, 0)
        adj2 = torch.stack(adj2, 0)
        actions = torch.cat(actions, 0)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        if self.use_order:
            return states.to(self.device), adj.to(self.device), actions.to(self.device), rewards.to(self.device), \
                states2.to(self.device), adj2.to(self.device), dones.to(self.device), orders.to(self.device)
        else:
            return states.to(self.device), adj.to(self.device), actions.to(self.device), \
                rewards.to(self.device), states2.to(self.device), adj2.to(self.device), dones.to(self.device)

    def update(self):
        self.update_step += 1
        batch = self.memory.sample(self.batch_size)
        orders = None
        if self.use_order:
            stacked_states, adj, actions, rewards, stacked_states2, adj2, dones, orders = self.unpack_batch(batch)
        else:
            stacked_states, adj, actions, rewards, stacked_states2, adj2, dones = self.unpack_batch(batch)
        
        self.Q.train()
        self.emb.train()
        self.actor.eval()
        
        # critic loss
        stacked_t, stacked_x = stacked_states[..., -1:], stacked_states[..., :-1]
        stacked2_t, stacked2_x = stacked_states2[..., -1:], stacked_states2[..., :-1]
        emb_input = stacked_x
        emb_input2 = stacked2_x
        states = self.emb(emb_input, adj)
        states2 = self.emb(emb_input2, adj2)
        actor_input2 = [states2, stacked2_t.squeeze(-1)]
        with torch.no_grad():
            tstates2 = self.temb(emb_input2, adj2).detach()
            action2, log_pi2 = self.actor.rsample(actor_input2, adj2)
            order2 = None
            if self.use_order:
                action2, order2 = action2
                log_pi2 = log_pi2[0] + log_pi2[1]
            critic_input2 = action2
            targets = self.tQ.min_Q(tstates2, critic_input2, adj2, order2) - self.log_alpha.exp() * log_pi2
        
        targets = rewards + (1-dones) * self.gamma * targets.detach()
        
        critic_input = actions
        predQ1, predQ2 = self.Q(states, critic_input, adj, orders)

        Q1_loss = F.mse_loss(predQ1, targets)
        Q2_loss = F.mse_loss(predQ2, targets)

        loss = Q1_loss + Q2_loss
        self.Q.optimizer.zero_grad()
        self.emb.optimizer.zero_grad()
        loss.backward()
        self.emb.optimizer.step()
        self.Q.optimizer.step()

        self.Q.eval()

        if self.update_step % self.delay_step == 0:
            # actor loss
            self.actor.train()
            states = self.emb(emb_input, adj)
            actor_input = [states, stacked_t.squeeze(-1)]
            action, log_pi = self.actor.rsample(actor_input, adj)
            order = None
            if self.use_order:
                action, order = action
                log_pi = log_pi[0] + log_pi[1]
            critic_input = action
            actor_loss = (self.log_alpha.exp() * log_pi - self.Q.min_Q(states, critic_input, adj, order)).mean()

            self.emb.optimizer.zero_grad()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.emb.optimizer.step()
            self.actor.optimizer.step()
        
            self.actor.eval()

            # target update
            if self.hard_target:
                self.tQ.load_state_dict(self.Q.state_dict())
                self.temb.load_state_dict(self.emb.state_dict())
            else:
                for tp, p in zip(self.tQ.parameters(), self.Q.parameters()):
                    tp.data.copy_(self.tau * p + (1-self.tau) * tp)
                for tp, p in zip(self.temb.parameters(), self.emb.parameters()):
                    tp.data.copy_(self.tau * p + (1-self.tau) * tp)
        
            # alpha loss
            alpha_loss = self.log_alpha * (-log_pi.detach() - self.target_entropy).mean()
            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()
        self.emb.eval()

        return predQ1.detach().mean().item(), predQ2.detach().mean().item()
    
    def save_model(self, path, name):
        torch.save(self.actor.state_dict(), os.path.join(path, f'{name}_actor.pt'))
        torch.save(self.emb.state_dict(), os.path.join(path, f'{name}_emb.pt'))
        torch.save(self.Q.state_dict(), os.path.join(path, f'{name}_Q.pt'))

    def load_model(self, path, name=None):
        head = ''
        if name is not None:
            head = name + '_'
        self.actor.load_state_dict(torch.load(os.path.join(path, f'{head}actor.pt'), map_location=self.device))
        self.emb.load_state_dict(torch.load(os.path.join(path, f'{head}emb.pt'), map_location=self.device))
        self.Q.load_state_dict(torch.load(os.path.join(path, f'{head}Q.pt'), map_location=self.device))