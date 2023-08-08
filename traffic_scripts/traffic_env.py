"""Runner script for single and multi-agent reinforcement learning experiments.

This script performs an RL experiment using the PPO algorithm. Choice of
hyperparameters can be seen and adjusted from the code below.

Usage
    python train.py EXP_CONFIG
"""

# The goal of this file:
# 1. get relevant arguments from the command line
# 2. get the relevant experiment configuration from flow_cfg
# 3. build the environment and interact with it using StableBaselines3


import argparse
import json
from pathlib import Path
import time
import os
import os.path as osp
from copy import deepcopy
from typing import Union
import numpy as np
import gym
import wandb
from wandb.integration.sb3 import WandbCallback
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from flow.core.rewards import REWARD_REGISTRY
from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.registry import make_create_env
from flow.environment import make_rollout_venv

from flow.data import wrappers

from flow.utils.rllib import FlowParamsEncoder, get_flow_params


def relative_symlink(target: Union[Path, str], destination: Union[Path, str]):
    """Create a symlink pointing to ``target`` from ``location``.
    Args:
        target: The target of the symlink (the file/directory that is pointed to)
        destination: The location of the symlink itself.
    """
    target = Path(target)
    destination = Path(destination)
    target_dir = destination.parent
    target_dir.mkdir(exist_ok=True, parents=True)
    relative_source = os.path.relpath(target, target_dir)
    dir_fd = os.open(str(target_dir.absolute()), os.O_RDONLY)
    print(f"{relative_source} -> {destination.name} in {target_dir}")
    try:
        os.symlink(relative_source, destination.name, dir_fd=dir_fd)
    finally:
        os.close(dir_fd)

def change_permissions_recursive(path, mode):
    for root, dirs, files in os.walk(path, topdown=False):
        for dir in [osp.join(root, d) for d in dirs]:
            os.chmod(dir, mode)
        for file in [osp.join(root, f) for f in files]:
            os.chmod(file, mode)

# Callbacks
# class CustomCallback(BaseCallback):
#     def on_episode_start(self, **kwargs):
#         self.locals["info"]["episode"].user_data["true_reward"] = []

#     def _on_step(self, **kwargs):
#         episode = self.locals["info"]["episode"]
#         environment = self.locals["info"]["env"]
#         env = environment.vector_env.envs[0]
#         actions = episode.last_action_for()

#         rew = 0
#         vel = np.array(
#             [env.k.vehicle.get_speed(veh_id) for veh_id in env.k.vehicle.get_ids()]
#         )
#         if all(vel > -100):
#             rew += REWARD_REGISTRY["desired_vel"](env, actions)
#             rew += 0.1 * REWARD_REGISTRY["accel"](env, actions)
#             rew += 0.1 * REWARD_REGISTRY["headway"](env, actions)

#         # reward average velocity
#         episode.user_data["true_reward"].append(rew)

#     def on_episode_end(self, **kwargs):
#         episode = self.locals["info"]["episode"]
#         sum_rew = np.sum(episode.user_data["true_reward"])
#         episode.custom_metrics["true_reward"] = sum_rew


# Rest of the code from parsing arguments and setup functions remain the same
def main(flags):
    """Perform the training operations."""
    # 1. Get experiment config
    module = __import__(
        "flow_cfg.exp_configs.rl.singleagent", fromlist=[flags.exp_config]
    )
    submodule = getattr(module, flags.exp_config)
    config_json = deepcopy(submodule.flow_params)
    config_json["exp_tag"] = flags.name
    config_json["env"].horizon = flags.horizon
    config_json_str = json.dumps(config_json, cls=FlowParamsEncoder, sort_keys=True, indent=4)

    # 2. Produce the parameters of reward misspecification
    rewards = flags.reward_fn.split(",")
    weights = flags.reward_weighting.split(",")
    assert len(rewards) == len(weights)
    reward_specification = [(r, float(w)) for r, w in zip(rewards, weights)] if not flags.test else None
    print("Reward specification: ", reward_specification)
    config_json["rewards"] = flags.reward_fn
    config_json["weights"] = flags.reward_weighting

    # 3. Create a log dir in flow/data/... and save configs 
    # make sure to add a timestamp to the name so that it's unique
    timestamp = time.strftime("%Y-%m-%d_%H-%M-%S", time.gmtime())
    data_dir = osp.join(os.getcwd(), "data")
    log_dir = osp.join(data_dir, flags.name, timestamp)
    ensure_dir(log_dir)
    with open(osp.join(log_dir, "configs.json"), "w") as f:
        f.write(config_json_str)
    
    # 4. Init a Weights and Biases run
    run = wandb.init(
        project="redteam_reward_models",
        config=config_json,
        dir=data_dir,
        sync_tensorboard=True,
        group=None if flags.test else "MERGE",
        monitor_gym=True,  # auto-upload the videos of agents playing the game
    )

    # Create a symlink from run.dir to wandb directory in log_dir
    wandb_dir = osp.join(log_dir, "wandb")
    parent_dir = os.path.dirname(run.dir)
    relative_symlink(parent_dir, wandb_dir)
    # TODO: We cannot modify the generated experiment results because of the file permissions
    change_permissions_recursive(log_dir, 0o777)

    # # 4. Create Flow environment
    create_env, env_id = make_create_env(
        params=config_json, reward_specification=reward_specification
    )
    # env = DummyVecEnv([create_env])
    # env = create_env()
    env = make_rollout_venv(
        gym_id=env_id,
        num_vec=flags.n_cpus,
        parallel=True,
        max_episode_steps=flags.horizon,
        # env_make_kwargs: Mapping[str, Any],
    )
    # env = wrappers.RolloutInfoWrapper(env)

    # Use DummyVecEnv CartPole-v1 for testing
    # env = gym.make("CartPole-v1")

    # 5. Use a PPO agent to collect data
    model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)
    
    # breakpoint()

    # custom_callback = CustomCallback()
    # Train the agent
    # model.learn(total_timesteps=flags.num_steps, callback=custom_callback)
    model.learn(
        total_timesteps=flags.num_steps,
        callback=WandbCallback(
            model_save_path=osp.join(log_dir, "models"),
            verbose=2,
        ),
    )
    run.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="Parse argument used when running a Flow simulation.",
        epilog="python train.py EXP_CONFIG",
    )
    parser.add_argument(
        "exp_config",
        type=str,
        help="Name of the experiment configuration file, as located in "
        "exp_configs/rl/singleagent or exp_configs/rl/multiagent.",
    )
    parser.add_argument("--name", type=str, help="Name of the experiment.")
    parser.add_argument("--reward_fn", type=str, help="Reward function.")
    parser.add_argument("--reward_weighting", type=str, help="Reward weighting.")
    parser.add_argument("--n_cpus", type=int, default=8, help="How many CPUs to use")
    parser.add_argument("--test", action="store_true", help="Run in test mode.")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=100000,
        help="How many total steps to perform learning over",
    )
    parser.add_argument(
        "--horizon", type=int, default=300, help="How many steps in each epsiode."
    )
    parser.add_argument(
        "--checkpoint", type=int, default=50, help="How frequently to checkpoint model."
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Directory with checkpoint to restore training from.",
    )
    flags = parser.parse_args()
    main(flags)
