import os
import csv
import json
import random
from datetime import datetime
from argparse import ArgumentParser
import numpy as np
import torch
import grid2op
from lightsim2grid import LightSimBackend
from grid2op.Reward import L2RPNSandBoxScore
from custom_reward import *
from agent import Agent
from train import TrainAgent
import matplotlib.cbook
import warnings
warnings.filterwarnings("ignore", category=matplotlib.cbook.mplDeprecation)


ENV_CASE = {
    '5': 'rte_case5_example',
    'sand': 'l2rpn_case14_sandbox',
    'wcci': 'l2rpn_wcci_2020'
}

DATA_SPLIT = {
    '5': ([i for i in range(20) if i not in [17, 19]], [17], [19]),
    'sand': (list(range(0, 40*26, 40)), list(range(1, 100*10+1, 100)), []),# list(range(2, 100*10+2, 100))),
    'wcci': ([17, 240, 494, 737, 990, 1452, 1717, 1942, 2204, 2403, 19, 242, 496, 739, 992, 1454, 1719, 1944, 2206, 2405, 230, 301, 704, 952, 1008, 1306, 1550, 1751, 2110, 2341, 2443, 2689],
            list(range(2880, 2890)), [])
}

MAX_FFW = {
    '5': 5,
    'sand': 26,
    'wcci': 26
}


def cli():
    parser = ArgumentParser()
    parser.add_argument('-s', '--seed', type=int, default=0)
    parser.add_argument('-c', '--case', type=str, default='wcci', choices=['sand', 'wcci', '5'])
    parser.add_argument('-gpu', '--gpuid', type=int, default=0)

    parser.add_argument('-ml', '--memlen', type=int, default=50000)
    parser.add_argument('-nf', '--nb_frame', type=int, default=100000,
                        help='the total number of interactions')
    parser.add_argument('-ts', '--test_step', type=int, default=1000,
                        help='the interaction number for next evaluation')
    parser.add_argument('-hn', '--head_number', type=int, default=8,
                        help='the number of head for attention')
    parser.add_argument('-sd', '--state_dim', type=int, default=128,
                        help='dimension of hidden state for GNN')
    parser.add_argument('-nh', '--n_history', type=int, default=6,
                        help='length of frame stack')
    parser.add_argument('-do', '--dropout', type=float, default=0.)
    parser.add_argument('-r', '--rule', type=str, default='c', choices=['c', 'd', 'o', 'f'],
                        help='low-level rule (capa, desc, opti, fixed)')
    parser.add_argument('-thr', '--threshold', type=float, default=0.1,
                        help='[-1, thr) -> bus 1 / [thr, 1] -> bus 2')
    parser.add_argument('-dg', '--danger', type=float, default=0.9,
                        help='the powerline with rho over danger is regarded as hazardous')
    parser.add_argument('-m', '--mask', type=int, default=5,
                        help='this agent manages the substations containing topology elements over "mask"')
    parser.add_argument('-tu', '--target_update', type=int, default=1,
                        help='period of target update')
    parser.add_argument('--tau', type=float, default=1e-3,
                        help='the weight of soft target update')
    parser.add_argument('-bs', '--batch_size', type=int, default=128)
    parser.add_argument('-lr', '--lr', type=float, default=5e-5)
    parser.add_argument('--gamma', type=float, default=0.995)
    parser.add_argument('-n', '--name', type=str, default='untitled')

    args = parser.parse_args()
    args.actor_lr = args.critic_lr = args.embed_lr = args.alpha_lr = args.lr
    return args

def log_params(args, path):
    f = open(os.path.join(path, "param.txt"), 'w')
    for key, val in args.__dict__.items():
        f.write(key + ': ' + str(val) + "\n")
    f.close()
    with open(os.path.join(path, 'param.json'), 'w', encoding='utf-8') as f:
        json.dump(vars(args), f)

def read_ffw_json(path, chronics, case):
    res = {}
    for i in chronics:
        for j in range(MAX_FFW[case]):
            with open(os.path.join(path, f'{i}_{j}.json'), 'r', encoding='utf-8') as f:
                a = json.load(f)
                res[(i,j)] = (a['dn_played'], a['donothing_reward'], a['donothing_nodisc_reward'])
            if i >= 2880: break
    return res

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)
    
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


if __name__ == '__main__':
    args = cli()
    seed_everything(args.seed)

    # settings
    model_name = f'{args.name}_{args.seed}'
    print('model name: ', model_name)

    OUTPUT_DIR = './result'
    DATA_DIR = './data'
    output_result_dir = os.path.join(OUTPUT_DIR, model_name)
    model_path = os.path.join(output_result_dir, 'model')
        
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpuid)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    env_name = ENV_CASE[args.case]
    env_path = os.path.join(DATA_DIR, env_name)
    chronics_path = os.path.join(env_path, 'chronics')
    train_chronics, valid_chronics, test_chronics = DATA_SPLIT[args.case]
    dn_json_path = os.path.join(env_path, 'json')
    
    # select chronics
    dn_ffw = read_ffw_json(dn_json_path, train_chronics + valid_chronics, args.case)

    ep_infos = {}
    if os.path.exists(dn_json_path):
        for i in list(set(train_chronics+valid_chronics)):
            with open(os.path.join(dn_json_path, f'{i}.json'), 'r', encoding='utf-8') as f:
                ep_infos[i] = json.load(f)

    env = grid2op.make(env_path, test=True, reward_class=L2RPNSandBoxScore, backend=LightSimBackend(),
                other_rewards={'loss': LossReward})
    test_env = grid2op.make(env_path, test=True, reward_class=L2RPNSandBoxScore, backend=LightSimBackend(),
                other_rewards={'loss': LossReward})
    env.deactivate_forecast()
    test_env.deactivate_forecast()
    env.seed(args.seed)
    test_env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = env.parameters.NB_TIMESTEP_OVERFLOW_ALLOWED = 3
    test_env.parameters.NB_TIMESTEP_RECONNECTION = env.parameters.NB_TIMESTEP_RECONNECTION = 12
    test_env.parameters.NB_TIMESTEP_COOLDOWN_LINE = env.parameters.NB_TIMESTEP_COOLDOWN_LINE = 3
    test_env.parameters.NB_TIMESTEP_COOLDOWN_SUB = env.parameters.NB_TIMESTEP_COOLDOWN_SUB = 3
    test_env.parameters.HARD_OVERFLOW_THRESHOLD = env.parameters.HARD_OVERFLOW_THRESHOLD = 200.0
    test_env.seed(59)
    chronic_num = len(test_chronics)
        
    print(env.parameters.__dict__)            
    # specify agent
    my_agent = Agent(env, **vars(args))
    mean = torch.load(os.path.join(env_path, 'mean.pt'))
    std = torch.load(os.path.join(env_path, 'std.pt'))
    my_agent.load_mean_std(mean, std)
    
    trainer = TrainAgent(my_agent, env, test_env, device, dn_json_path, dn_ffw, ep_infos)

    if not os.path.exists(output_result_dir):
        os.makedirs(output_result_dir)
        os.makedirs(model_path)
        log_params(args, output_result_dir)

    trainer.train(
        args.seed, args.nb_frame, args.test_step,
        train_chronics, valid_chronics, output_result_dir, model_path, MAX_FFW[args.case])
    trainer.agent.save_model(model_path, 'last')