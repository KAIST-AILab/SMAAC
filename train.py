import os
import sys
import csv
import copy
import random
import numpy as np
import torch
from matplotlib import pyplot as plt


class TrainAgent(object):
    def __init__(self, agent, env, test_env, device,dn_json_path, dn_ffw, ep_infos):
        self.device = device
        self.agent = agent
        self.env = env
        self.test_env = test_env
        self.dn_json_path = dn_json_path
        self.dn_ffw = dn_ffw
        self.ep_infos = ep_infos

    def save_model(self, path, name):
        self.agent.save_model(path, name)

    # following competition evaluation script
    def compute_episode_score(self, chronic_id, agent_step, agent_reward, ffw=None):
        min_losses_ratio = 0.7
        ep_marginal_cost = self.env.gen_cost_per_MW.max()
        if ffw is None:
            ep_do_nothing_reward = self.ep_infos[chronic_id]["donothing_reward"]
            ep_do_nothing_nodisc_reward = self.ep_infos[chronic_id]["donothing_nodisc_reward"]
            ep_dn_played = self.ep_infos[chronic_id]['dn_played']
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])
        else:
            start_idx = 0 if ffw == 0 else ffw * 288 - 2
            end_idx = start_idx + 864
            ep_dn_played, ep_do_nothing_reward, ep_do_nothing_nodisc_reward = self.dn_ffw[(chronic_id, ffw)]
            ep_loads = np.array(self.ep_infos[chronic_id]["sum_loads"])[start_idx:end_idx]
            ep_losses = np.array(self.ep_infos[chronic_id]["losses"])[start_idx:end_idx]

        # Add cost of non delivered loads for blackout steps
        blackout_loads = ep_loads[agent_step:]
        if len(blackout_loads) > 0:
            blackout_reward = np.sum(blackout_loads) * ep_marginal_cost
            agent_reward += blackout_reward

        # Compute ranges
        worst_reward = np.sum(ep_loads) * ep_marginal_cost
        best_reward = np.sum(ep_losses) * min_losses_ratio
        zero_reward = ep_do_nothing_reward
        zero_blackout = ep_loads[ep_dn_played:]
        zero_reward += np.sum(zero_blackout) * ep_marginal_cost
        nodisc_reward = ep_do_nothing_nodisc_reward

        # Linear interp episode reward to codalab score
        if zero_reward != nodisc_reward:
            # DoNothing agent doesnt complete the scenario
            reward_range = [best_reward, nodisc_reward, zero_reward, worst_reward]
            score_range = [100.0, 80.0, 0.0, -100.0]
        else:
            # DoNothing agent can complete the scenario
            reward_range = [best_reward, zero_reward, worst_reward]
            score_range = [100.0, 0.0, -100.0]
            
        ep_score = np.interp(agent_reward, reward_range, score_range)
        return ep_score

    def interaction(self, obs, prev_act, cid, ffw, start_step):
        state = self.agent.get_current_state()
        adj = self.agent.adj.clone()
        action = self.agent.goal.clone()
        order = None if self.agent.order is None else self.agent.order.clone()
        reward, train_reward, step = 0, 0, 0
        while True:
            # prev_act is executed at first anyway
            if prev_act:
                act = prev_act
                prev_act = None
            else:
                act = self.agent.act(obs, None, None)
                if self.agent.save:
                    # pass this act to the next step.
                    prev_act = act
                    break
            # just step if action is okay or failed to find other action
            obs, rew, done, info = self.env.step(act)
            reward += rew
            new_reward = info['rewards']['loss']
            train_reward += new_reward
            step += 1
            if start_step + step == 864:
                done = True

            if done:
                break
        train_reward = np.clip(train_reward, -2, 10)
        next_state = self.agent.get_current_state()
        next_adj = self.agent.adj.clone()
        die = bool(done and info['exception'])
        transition = (state, adj, action, train_reward, next_state, next_adj, die, order)
        etcs = (step + start_step, prev_act, info)
        infos = (transition, etcs)
        return obs, reward, done, infos

    def multi_step_transition(self, temp_memory):
        transitions = []
        running_reward = 0
        final_state, final_adj, final_die = temp_memory[-1][4:7]
        for tran in reversed(temp_memory):
            (state, adj, action, train_reward, _,_,_, order) = tran
            running_reward += train_reward
            new_tran = (state, adj, action, running_reward, final_state, final_adj, final_die, order)
            transitions.append(new_tran)

        return transitions
    
    # compute weight for chronic sampling
    def chronic_priority(self, cid, ffw, step):
        m = 864
        scale = 2.
        diff_coef = 0.05
        d = self.dn_ffw[(cid, ffw)][0]
        progress = 1 - np.sqrt(step/m)
        difficulty = 1 - np.sqrt(d/m)
        score = (progress + diff_coef * difficulty) * scale
        return score

    def train(self, seed, nb_frame, test_step, train_chronics, valid_chronics, output_dir, model_path, max_ffw):
        best_score = -100

        # initialize training chronic sampling weights
        train_chronics_ffw = [(cid, fw) for cid in train_chronics for fw in range(max_ffw)]
        total_chronic_num = len(train_chronics_ffw)
        chronic_records = [0] * total_chronic_num
        chronic_step_records = [0] * total_chronic_num

        for i in chronic_records:
            cid, fw = train_chronics_ffw[i]
            chronic_records[i] = self.chronic_priority(cid, fw, 1)

        # training loop
        while self.agent.update_step < nb_frame:

            # sample training chronic
            dist = torch.distributions.categorical.Categorical(logits=torch.Tensor(chronic_records))
            record_idx = dist.sample().item()
            chronic_id, ffw = train_chronics_ffw[record_idx]
            self.env.set_id(chronic_id)
            self.env.seed(seed)
            obs = self.env.reset()
            if ffw > 0:
                self.env.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self.env.step(self.env.action_space())
            done = False
            alive_frame = 0
            total_reward = 0
            train_reward = 0

            self.agent.reset(obs)
            prev_act = self.agent.act(obs, None, None)
            temp_memory = []
            while not done:
                obs, reward, done, info = self.interaction(obs, prev_act, chronic_id, ffw, alive_frame)
                alive_frame, prev_act = info[1][:2]
                total_reward += reward
                train_reward += info[0][3]
                temp_memory.append(list(map(lambda x: x.cpu() if torch.is_tensor(x) else x, info[0])))
                if len(temp_memory) == self.agent.k_step or done:
                    for transition in self.multi_step_transition(temp_memory):
                        self.agent.append_sample(*transition)
                    temp_memory.clear()
                
                if len(self.agent.memory) > self.agent.update_start:
                    self.agent.update()
                    if self.agent.update_step % test_step == 0:
                        eval_iter = self.agent.update_step // test_step
                        cache = self.agent.cache_stat()
                        result, stats, scores, steps = self.test(valid_chronics, max_ffw)
                        self.agent.load_cache_stat(cache)
                        print(f"[{eval_iter:4d}] Valid: score {stats['score']} | step {stats['step']}")
                        
                        # log and save model
                        with open(os.path.join(model_path, 'score.csv'), 'a', newline='') as cf:
                            csv.writer(cf).writerow(scores)
                        with open(os.path.join(model_path, 'step.csv'), 'a', newline='') as cf:
                            csv.writer(cf).writerow(steps)
                        if best_score < stats['score']:
                            best_score = stats['score']
                            self.agent.save_model(model_path, 'best')
                if self.agent.update_step > nb_frame :
                    break
            
            # update chronic sampling weight
            chronic_records[record_idx] = self.chronic_priority(chronic_id, ffw, alive_frame)
            chronic_step_records[record_idx] = alive_frame

    def test(self, chronics, max_ffw, f=None, verbose=False):
        result = {}
        steps, scores = [], []

        if max_ffw == 5:
            chronics = chronics * 5
        for idx, i in enumerate(chronics):
            if max_ffw == 5:
                ffw = idx
            else:
                ffw = int(np.argmin([self.dn_ffw[(i, fw)][0] for fw in range(max_ffw) if (i, fw) in self.dn_ffw and self.dn_ffw[(i, fw)][0] >= 10]))

            dn_step = self.dn_ffw[(i, ffw)][0]
            self.test_env.seed(59)
            self.test_env.set_id(i)
            obs = self.test_env.reset()
            self.agent.reset(obs)
            
            if ffw > 0:
                self.test_env.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self.test_env.step(self.test_env.action_space())

            total_reward = 0
            alive_frame = 0
            done = False
            result[(i, ffw)] = {}
            while not done:
                act = self.agent.act(obs, 0, 0)
                obs, reward, done, info = self.test_env.step(act)
                total_reward += reward
                alive_frame += 1
                if alive_frame == 864:
                    done = True
            
            l2rpn_score = float(self.compute_episode_score(i, alive_frame, total_reward, ffw))
            print(f'[Test Ch{i:4d}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) Score: {l2rpn_score:9.4f} ')
            scores.append(l2rpn_score)
            steps.append(alive_frame)

            result[(i, ffw)]["real_reward"] = total_reward
            result[(i, ffw)]["reward"] = l2rpn_score
            result[(i, ffw)]["step"] = alive_frame

        val_step = val_score = val_rew = 0
        for key in result:
            val_step += result[key]['step']
            val_score += result[key]['reward']
            val_rew += result[key]['real_reward']
        stats = {
            'step': val_step / len(chronics),
            'score': val_score / len(chronics),
            'reward': val_rew / len(chronics),
            'alpha': self.agent.log_alpha.exp().item()
        }
        return result, stats, scores, steps

    def evaluate(self, chronics, max_ffw, fig_path, mode='best', plot_topo=False):
        if plot_topo:
            from grid2op.PlotGrid import PlotMatplot
            plot_helper = PlotMatplot(self.test_env.observation_space, width=1280, height=1280,
                                    sub_radius=7.5, gen_radius=2.5, load_radius=2.5)
            self.test_env.attach_renderer()
        result = {}
        steps, scores = [], []

        if max_ffw == 5:
            chronics = chronics * 5
        for idx, i in enumerate(chronics):
            if max_ffw == 5:
                ffw = idx
            else:
                ffw = int(np.argmin([self.dn_ffw[(i, fw)][0] for fw in range(max_ffw) if (i, fw) in self.dn_ffw and self.dn_ffw[(i, fw)][0] >= 10]))

            dn_step = self.dn_ffw[(i, ffw)][0]
            self.test_env.seed(59)
            self.test_env.set_id(i)
            obs = self.test_env.reset()
            self.agent.reset(obs)
            
            if ffw > 0:
                self.test_env.fast_forward_chronics(ffw * 288 - 3)
                obs, *_ = self.test_env.step(self.test_env.action_space())

            total_reward = 0
            alive_frame = 0
            done = False
            topo_dist = []

            result[(i, ffw)] = {}
            bus_goal = None
            while not done:
                if plot_topo:
                    danger = not self.agent.is_safe(obs)
                    if self.agent.save and danger:
                        temp_acts = []
                        temp_obs = [obs]
                        bus_goal = self.agent.bus_goal.numpy() + 1
                        prev_topo = obs.topo_vect[self.agent.converter.sub_mask]
                        prev_step = alive_frame
                    topo_dist.append(float((obs.topo_vect==2).sum()))
                act = self.agent.act(obs, 0, 0)
                obs, reward, done, info = self.test_env.step(act)
                total_reward += reward
                alive_frame += 1
                if plot_topo:
                    if bus_goal is not None:
                        temp_acts.append(act)
                        temp_obs.append(obs)
                        if self.agent.is_safe(obs) and len(self.agent.low_actions)==0:
                            if (np.sum([a == self.test_env.action_space() for a in temp_acts]) < len(temp_acts) -1) and alive_frame - prev_step > 1:
                                temp_topo = obs.topo_vect[self.agent.converter.sub_mask]
                                print('Prev:', prev_topo)
                                print('Goal:', bus_goal)
                                print('Topo:', temp_topo)
                                for i in range(3):
                                    fig = plot_helper.plot_obs(temp_obs[i], line_info="rho", load_info=None, gen_info=None)
                                    fig.savefig(f'{idx}_{alive_frame}_obs{i}.pdf')
                                print(prev_step, alive_frame - prev_step, (prev_topo != temp_topo).sum())
                            bus_goal = None
                            temp_acts = []

                if alive_frame == 864:
                    done = True
            
            l2rpn_score = float(self.compute_episode_score(i, alive_frame, total_reward, ffw))

            print(f'[Test Ch{i:4d}({ffw:2d})] {alive_frame:3d}/864 ({dn_step:3d}) Score: {l2rpn_score:9.4f}')
            scores.append(l2rpn_score)
            steps.append(alive_frame)

            result[(i, ffw)]["real_reward"] = total_reward
            result[(i, ffw)]["reward"] = l2rpn_score
            result[(i, ffw)]["step"] = alive_frame

            # plot topo dist
            if plot_topo:
                plt.figure(figsize=(8, 6))
                plt.plot(np.arange(len(topo_dist)), topo_dist)
                plt.savefig(os.path.join(fig_path, f'{mode}_{idx}_topo.png'))
                np.save(os.path.join(fig_path, f'{mode}_{idx}_topo.npy'), np.array(topo_dist))

        val_step = val_score = val_rew = 0
        for key in result:
            val_step += result[key]['step']
            val_score += result[key]['reward']
            val_rew += result[key]['real_reward']
            
        stats = {
            'step': val_step / len(chronics),
            'score': val_score / len(chronics),
            'reward': val_rew / len(chronics)
        }
        if plot_topo:
            with open(os.path.join(fig_path, f"{mode}_{stats['score']:.3f}.txt"), 'w') as f:
                f.write(str(stats))
                f.write(str(result))
        return stats, scores, steps