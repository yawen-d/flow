"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""
import argparse
import json
import os
import sys
from time import strftime
from copy import deepcopy

from flow.core.rewards import REWARD_REGISTRY
from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

#from ray.rllib.agents.callbacks import DefaultCallbacks

import numpy as np
import wandb

# Callbacks
# Custom state can be stored for the episode in the info["episode"].user_data dict
# Custom scalar metrics reported by saving values to the info["episode"].custom_metrics dict
def on_episode_start(info):
    episode = info["episode"]
    episode.user_data["true_reward"] = []

def on_episode_step(info):
    episode = info["episode"]
    environment = info["env"]
    env = environment.vector_env.envs[0]
    actions = episode.last_action_for()

    rew = 0
    vel = np.array([env.k.vehicle.get_speed(veh_id) for veh_id in env.k.vehicle.get_ids()])
    if all(vel > -100):
        rew += REWARD_REGISTRY['desired_vel'](env, actions)
        rew += 0.1 * REWARD_REGISTRY['accel'](env, actions)
        rew += 0.1 * REWARD_REGISTRY['headway'](env, actions)
        #rew += REWARD_REGISTRY['desired_vel'](env, actions)
        #rew += 0.1 * REWARD_REGISTRY['forward'](env, actions)
        #rew += REWARD_REGISTRY['lane_bool'](env, actions)

    # reward average velocity
    episode.user_data["true_reward"].append(rew)


def on_episode_end(info):
    episode = info["episode"]
    sum_rew = np.sum(episode.user_data["true_reward"])
    episode.custom_metrics["true_reward"] = sum_rew


def parse_args(args):
    """Parse training options user can specify in command line.

    Returns
    -------
    argparse.Namespace
        the output parser object
    """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG")

    # required input parameters
    parser.add_argument(
        'exp_config', type=str,
        help='Name of the experiment configuration file, as located in '
             'exp_configs/rl/singleagent or exp_configs/rl/multiagent.')
    parser.add_argument(
        '--multi', action='store_true', help='Run multiagent experiment')
    parser.add_argument(
        '--test', action='store_true', help='No wandb')
    parser.add_argument(
        '--n_cpus', type=int, default=1,
        help='How many CPUs to use')
    parser.add_argument(
        '--num_steps', type=int, default=5000,
        help='How many total steps to perform learning over')
    parser.add_argument(
        '--rollout_size', type=int, default=20,
        help='How many rollouts performed each episode.')
    parser.add_argument(
        '--horizon', type=int, default=300,
        help='How many steps in each epsiode.')
    parser.add_argument(
        '--checkpoint', type=int, default=50,
        help='How frequently to checkpoint model.')
    parser.add_argument(
        '--checkpoint_path', type=str, default=None,
        help='Directory with checkpoint to restore training from.')
    return parser.parse_known_args(args)[0]


def setup_exps_rllib(flow_params,
                     n_rollouts,
                     n_cpus,
                     reward_specification=None,
                     policy_graphs=None,
                     policy_mapping_fn=None,
                     policies_to_train=None):
    """Return the relevant components of an RLlib experiment.

    Parameters
    ----------
    flow_params : dict
        flow-specific parameters (see flow/utils/registry.py)
    n_rollouts : int
        number of rollouts per training iteration
    n_cpus : int
        number of cpus 
    policy_graphs : dict, optional
        TODO
    policy_mapping_fn : function, optional
        TODO
    policies_to_train : list of str, optional
        TODO

    Returns
    -------
    str
        name of the training algorithm
    str
        name of the gym environment to be trained
    dict
        training configuration parameters
    """
    from ray import tune
    from ray.tune.registry import register_env
    try:
        from ray.rllib.agents.agent import get_agent_class
    except ImportError:
        from ray.rllib.agents.registry import get_agent_class

    horizon = flow_params['env'].horizon
    
    alg_run = "PPO"

    agent_cls = get_agent_class(alg_run)
    config = deepcopy(agent_cls._default_config)

    config["seed"] = 17
    config["num_workers"] = n_cpus - 1
    config["train_batch_size"] = horizon * n_rollouts
    config["gamma"] = 0.999  # discount rate
    print(sys.argv)
    fcnet_hiddens = [int(sys.argv[5])] * int(sys.argv[6])
    config["model"].update({"fcnet_hiddens": fcnet_hiddens}) 
    config["sgd_minibatch_size"] = min(16*1024, config["train_batch_size"])
    config["use_gae"] = True
    config["lambda"] = 0.97
    config["kl_target"] = 0.02
    config["vf_clip_param"] = 10000
    config["num_sgd_iter"] = 10
    config["vf_loss_coeff"] = 0.5
    config["horizon"] = horizon

    step = [
            [0, 0.0001],
            [1400*749, 0.0001],
            [1400*750, 0.00005],
            [1400*1499, 0.00005],
            [1400*1500, 0.00001],
            [1400*2999, 0.00001],
            [1400*3000, 0.000005],
            [1400*4999, 0.000005],
            [1400*5000, 0.000001],
        ]

    rise = [[0, 0.0001]]
    drop3 = 0.000001
    drop2 = 0.000005
    drop1 = 0.00001
    lo=0.00005
    hi=0.0005
    cyclic = []
    cycle_len = 1400
    cycle_stop = 360
    for i in range(cycle_stop):
        cyclic.append([cycle_len*2*i, hi])
        cyclic.append([cycle_len*(2*i+1), lo])
    cyclic.append([2*cycle_len*cycle_stop, drop1])
    cyclic.append([4*cycle_len*cycle_stop-1, drop1])
    cyclic.append([4*cycle_len*cycle_stop, drop2])
    cyclic.append([8*cycle_len*cycle_stop-1, drop2])
    cyclic.append([8*cycle_len*cycle_stop, drop3])
    test = [[0, 0.0001], [1400*5-1, 0.0001], [1400*5, 0.00001], [1400*10, 0.00001]]
    space1 = [[0, 0.0001], [1400*749, 0.0001], [1400*750, 0.00001], [1400*7499, 0.00001], [1400*7500, 0.000001], [1400*5000000, 0.000001]]
    space2 = [[0, 0.0001], [1400*1499, 0.0001], [1400*1500, 0.000001], [1400*2999, 0.000001], [1400*3000, 0.0000001], [1400*5000000, 0.0000001]]

    linear = [[0, 0.000001], [1400*5000000, 0.000001]]

    config["lr_schedule"] = space2
    #config["simple_optimizer"] = True
    #config["framework"] = "torch"

    # save the flow params for replay
    flow_json = json.dumps(
        flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
    config['env_config']['flow_params'] = flow_json
    config['env_config']['run'] = alg_run

    # multiagent configuration
    if policy_graphs is not None:
        config['multiagent'].update({'policies': policy_graphs})
    if policy_mapping_fn is not None:
        config['multiagent'].update(
            {'policy_mapping_fn': tune.function(policy_mapping_fn)})
    if policies_to_train is not None:
        config['multiagent'].update({'policies_to_train': policies_to_train})

    config['callbacks'] = {
                    "on_episode_start": on_episode_start,
                    "on_episode_step": on_episode_step,
                    "on_episode_end": on_episode_end,
                }
    #config['callbacks'] = RewardCallback
    create_env, gym_name = make_create_env(params=flow_params, reward_specification=reward_specification)

    # Register as rllib env
    register_env(gym_name, create_env)
    return alg_run, gym_name, config


def train(flags):
    """Train policies using the PPO algorithm in RLlib."""
    import ray
    from ray.tune import run_experiments
    from ray.tune.experiment import convert_to_experiment_list

    # Import relevant information from the exp_config script.
    if flags.multi:
        module = __import__(
            "flow_cfg.exp_configs.rl.multiagent", fromlist=[flags.exp_config])
    else:
        module = __import__(
            "flow_cfg.exp_configs.rl.singleagent", fromlist=[flags.exp_config])

    submodule = getattr(module, flags.exp_config)
    flow_params = submodule.flow_params
    flow_params["exp_tag"] = sys.argv[2]
    flow_params["env"].horizon = flags.horizon

    rewards = sys.argv[3].split(",")
    weights = sys.argv[4].split(",")
    assert len(rewards) == len(weights)
    reward_specification = [(r, float(w)) for r, w in zip(rewards, weights)]
    
    n_rollouts = flags.rollout_size
    n_cpus = flags.n_cpus
    policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
    policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
    policies_to_train = getattr(submodule, "policies_to_train", None)

    #ray.init()
    ray.init(address=os.environ["ip_head"])
    
    alg_run, gym_name, config = setup_exps_rllib(
        flow_params, n_rollouts, n_cpus, reward_specification,
        policy_graphs, policy_mapping_fn, policies_to_train)

    print(f"Epochs: {flags.num_steps}")
    exp_config = {
        "run": alg_run,
        "env": gym_name,
        "config": {
            **config
        },
        "checkpoint_freq": flags.checkpoint,
        "checkpoint_at_end": True,
        "max_failures": 999,
        "stop": {
            "training_iteration": flags.num_steps,
        },
    }

    if flags.checkpoint_path is not None:
        exp_config['restore'] = flags.checkpoint_path

    run_experiments({flow_params["exp_tag"]: exp_config})


def main(args):
    """Perform the training operations."""
    # Parse script-level arguments (not including package arguments).
    #np.seterr(all='raise')
    flags = parse_args(args)
    if flags.test:
        wandb.init(entity="aypan17", project="test-space", sync_tensorboard=True)
    else:
        wandb.init(entity="aypan17", project="value-learning", group="traffic-merge", sync_tensorboard=True)
    train(flags)


if __name__ == "__main__":
    print(sys.argv)
    main(sys.argv[1:])
