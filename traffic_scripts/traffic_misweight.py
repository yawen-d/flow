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

from flow.core.util import ensure_dir
from flow.utils.registry import env_constructor
from flow.utils.rllib import FlowParamsEncoder, get_flow_params
from flow.utils.registry import make_create_env

import numpy as np
import wandb

os.environ.setdefault("TUNE_GLOBAL_CHECKPOINT_S", str(sys.maxsize))

# Callbacks
# Custom state can be stored for the episode in the info["episode"].user_data dict
# Custom scalar metrics reported by saving values to the info["episode"].custom_metrics dict
def on_episode_start(info):
	episode = info["episode"]
	episode.user_data["true_reward"] = []

def on_episode_step(info):
	episode = info["episode"]
	env = info["env"]

	kernel = env.vector_env.envs[0].k
	vel = np.array([
			kernel.vehicle.get_speed(veh_id)
			for veh_id in kernel.vehicle.get_ids()
		])

	# reward average velocity
	eta_2 = 4.
	true_reward = eta_2 * np.mean(vel) / 20

	# punish accelerations (should lead to reduced stop-and-go waves)
	eta = 4  # 0.25
	mean_actions = np.mean(np.abs(np.array(episode.last_action_for())))
	accel_threshold = 0

	if mean_actions > accel_threshold:
		true_reward += eta * (accel_threshold - mean_actions)
	episode.user_data["true_reward"].append(true_reward)


def on_episode_step_multi(info):
	episode = info["episode"]
	env = info["env"]

	kernel = env.envs[0].k
	vel = np.array([
			kernel.vehicle.get_speed(veh_id)
			for veh_id in kernel.vehicle.get_ids()
		])

	# reward average velocity
	eta_2 = 4.
	true_reward = eta_2 * np.mean(vel) / 20

	# punish accelerations (should lead to reduced stop-and-go waves)
	eta = 4  # 0.25
	mean_actions = np.mean(np.abs(np.array(episode.last_action_for())))
	accel_threshold = 0

	if mean_actions > accel_threshold:
		true_reward += eta * (accel_threshold - mean_actions)
	episode.user_data["true_reward"].append(true_reward)

def on_episode_end(info):
	episode = info["episode"]
	mean_rew = np.sum(episode.user_data["true_reward"])
	print(mean_rew)
	episode.custom_metrics["true_reward"] = mean_rew

def on_train_result(info):
	pass

def on_postprocess_traj(info):
	pass


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

	# optional input parameters
	parser.add_argument(
		'--rl_trainer', type=str, default="rllib",
		help='the RL trainer to use. either rllib or Stable-Baselines')

	parser.add_argument(
		'--num_cpus', type=int, default=1,
		help='How many CPUs to use')
	parser.add_argument(
		'--num_steps', type=int, default=5000,
		help='How many total steps to perform learning over')
	parser.add_argument(
		'--rollout_size', type=int, default=20,
		help='How many steps are in a training batch.')
	parser.add_argument(
		'--horizon', type=int, default=400,
		help='Number of steps in each episode.')
	parser.add_argument(
		'--checkpoint', type=int, default=50,
		help='How frequently to checkpoint model.')
	parser.add_argument(
		'--checkpoint_path', type=str, default=None,
		help='Directory with checkpoint to restore training from.')

	parser.add_argument('-p', '--metric', type=str, default='reward')
	parser.add_argument('-g', '--goal', type=str, default='maximize')
	parser.add_argument('-n', '--name', type=str, default='untitled')
	parser.add_argument('-gpu', '--gpuid', type=int, default=0)
	parser.add_argument('--sweep', action='store_true', help='run hparam sweep')
	parser.add_argument('--replicas', type=bool, default=False, 
		help='number of different hparam settings to try')

	parser.add_argument('--optim', type=str, default='adam')
	parser.add_argument('--batch_size', type=int, default=128)
	parser.add_argument('--lr', type=float, default=5e-5)
	parser.add_argument('--epochs', type=int, default=1000)
	parser.add_argument('--d_model', type=int, default=128)
	parser.add_argument('--dropout', type=float, default=0.2)

	return parser.parse_known_args(args)[0]


def run_model_stablebaseline(flow_params,
							 num_cpus=1,
							 rollout_size=50,
							 num_steps=50):
	"""Run the model for num_steps if provided.

	Parameters
	----------
	flow_params : dict
		flow-specific parameters
	num_cpus : int
		number of CPUs used during training
	rollout_size : int
		length of a single rollout
	num_steps : int
		total number of training steps
	The total rollout length is rollout_size.

	Returns
	-------
	stable_baselines.*
		the trained model
	"""
	from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
	from stable_baselines import PPO2

	if num_cpus == 1:
		constructor = env_constructor(params=flow_params, version=0)()
		# The algorithms require a vectorized environment to run
		env = DummyVecEnv([lambda: constructor])
	else:
		env = SubprocVecEnv([env_constructor(params=flow_params, version=i)
							 for i in range(num_cpus)])

	train_model = PPO2('MlpPolicy', env, verbose=1, n_steps=rollout_size)
	train_model.learn(total_timesteps=num_steps)
	return train_model


def setup_exps_rllib(flow_params,
					 n_cpus,
					 n_rollouts,
					 policy_graphs=None,
					 policy_mapping_fn=None,
					 policies_to_train=None):
	"""Return the relevant components of an RLlib experiment.

	Parameters
	----------
	flow_params : dict
		flow-specific parameters (see flow/utils/registry.py)
	n_cpus : int
		number of CPUs to run the experiment over
	n_rollouts : int
		number of rollouts per training iteration
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
	
	cfg = wandb.config
	horizon = flow_params['env'].horizon
	print(f"Horizon: {horizon}")
	alg_run = "PPO"

	agent_cls = get_agent_class(alg_run)
	config = deepcopy(agent_cls._default_config)
	
	config["seed"] = 17
	
	config["num_workers"] = n_cpus - 1
	config["train_batch_size"] = horizon * n_rollouts
	config["sgd_minibatch_size"] = min(16 * 1024, config["train_batch_size"])
	config["gamma"] = 0.999  # discount rate
	fcnet = [int(sys.argv[4])] * int(sys.argv[5]) #[32, 32, 32]
	config["model"].update({"fcnet_hiddens": fcnet})
	config["use_gae"] = True
	config["lambda"] = 0.97
	config["kl_target"] = 0.02
	config["vf_clip_param"] = 10000
	config["num_sgd_iter"] = 10
	config["horizon"] = horizon
	
	# save the flow params for replay
	flow_json = json.dumps(
		flow_params, cls=FlowParamsEncoder, sort_keys=True, indent=4)
	config['env_config']['flow_params'] = flow_json
	config['env_config']['run'] = alg_run

	# multiagent configuration
	if policy_graphs is not None:
		print("policy_graphs", policy_graphs)
		config['multiagent'].update({'policies': policy_graphs})
	if policy_mapping_fn is not None:
		config['multiagent'].update(
			{'policy_mapping_fn': tune.function(policy_mapping_fn)})
	if policies_to_train is not None:
		config['multiagent'].update({'policies_to_train': policies_to_train})

	config['callbacks'] = {
					"on_episode_start": on_episode_start,
					"on_episode_step": on_episode_step_multi if cfg.multi else on_episode_step,
					"on_episode_end": on_episode_end,
				}

	create_env, gym_name = make_create_env(params=flow_params)

	# Register as rllib env
	register_env(gym_name, create_env)
	return alg_run, gym_name, config


def train_rllib(submodule, flags):
	"""Train policies using the PPO algorithm in RLlib."""
	import ray
	from ray.tune import run_experiments

	flow_params = submodule.flow_params
	flow_params["exp_tag"] = sys.argv[2]
	flow_params["env"].additional_params["eta"] = float(sys.argv[3])
	flow_params["env"].horizon = flags.horizon
	n_cpus = int(sys.argv[6])
	n_rollouts = flags.rollout_size
	policy_graphs = getattr(submodule, "POLICY_GRAPHS", None)
	policy_mapping_fn = getattr(submodule, "policy_mapping_fn", None)
	policies_to_train = getattr(submodule, "policies_to_train", None)

	alg_run, gym_name, config = setup_exps_rllib(
		flow_params, n_cpus, n_rollouts,
		policy_graphs, policy_mapping_fn, policies_to_train)

	ray.init(address=os.environ["ip_head"])
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


def train_h_baselines(env_name, args, multiagent):
	"""Train policies using SAC and TD3 with h-baselines."""
	from hbaselines.algorithms import OffPolicyRLAlgorithm
	from hbaselines.utils.train import parse_options, get_hyperparameters

	# Get the command-line arguments that are relevant here
	args = parse_options(description="", example_usage="", args=args)

	# the base directory that the logged data will be stored in
	base_dir = "training_data"

	for i in range(args.n_training):
		# value of the next seed
		seed = args.seed + i

		# The time when the current experiment started.
		now = strftime("%Y-%m-%d-%H:%M:%S")

		# Create a save directory folder (if it doesn't exist).
		dir_name = os.path.join(base_dir, '{}/{}'.format(args.env_name, now))
		ensure_dir(dir_name)

		# Get the policy class.
		if args.alg == "TD3":
			if multiagent:
				from hbaselines.multi_fcnet.td3 import MultiFeedForwardPolicy
				policy = MultiFeedForwardPolicy
			else:
				from hbaselines.fcnet.td3 import FeedForwardPolicy
				policy = FeedForwardPolicy
		elif args.alg == "SAC":
			if multiagent:
				from hbaselines.multi_fcnet.sac import MultiFeedForwardPolicy
				policy = MultiFeedForwardPolicy
			else:
				from hbaselines.fcnet.sac import FeedForwardPolicy
				policy = FeedForwardPolicy
		else:
			raise ValueError("Unknown algorithm: {}".format(args.alg))

		# Get the hyperparameters.
		hp = get_hyperparameters(args, policy)

		# Add the seed for logging purposes.
		params_with_extra = hp.copy()
		params_with_extra['seed'] = seed
		params_with_extra['env_name'] = args.env_name
		params_with_extra['policy_name'] = policy.__name__
		params_with_extra['algorithm'] = args.alg
		params_with_extra['date/time'] = now

		# Add the hyperparameters to the folder.
		with open(os.path.join(dir_name, 'hyperparameters.json'), 'w') as f:
			json.dump(params_with_extra, f, sort_keys=True, indent=4)

		# Create the algorithm object.
		alg = OffPolicyRLAlgorithm(
			policy=policy,
			env="flow:{}".format(env_name),
			eval_env="flow:{}".format(env_name) if args.evaluate else None,
			**hp
		)

		# Perform training.
		alg.learn(
			total_steps=args.total_steps,
			log_dir=dir_name,
			log_interval=args.log_interval,
			eval_interval=args.eval_interval,
			save_interval=args.save_interval,
			initial_exploration_steps=args.initial_exploration_steps,
			seed=seed,
		)


def train_stable_baselines(submodule, flags):
	"""Train policies using the PPO algorithm in stable-baselines."""
	from stable_baselines.common.vec_env import DummyVecEnv
	from stable_baselines import PPO2

	flow_params = submodule.flow_params
	# Path to the saved files
	exp_tag = flow_params['exp_tag']
	result_name = '{}/{}'.format(exp_tag, strftime("%Y-%m-%d-%H:%M:%S"))

	# Perform training.
	print('Beginning training.')
	model = run_model_stablebaseline(
		flow_params, flags.num_cpus, flags.rollout_size, flags.num_steps)

	# Save the model to a desired folder and then delete it to demonstrate
	# loading.
	print('Saving the trained model!')
	path = os.path.realpath(os.path.expanduser('~/baseline_results'))
	ensure_dir(path)
	save_path = os.path.join(path, result_name)
	model.save(save_path)

	# dump the flow params
	with open(os.path.join(path, result_name) + '.json', 'w') as outfile:
		json.dump(flow_params, outfile,
				  cls=FlowParamsEncoder, sort_keys=True, indent=4)

	# Replay the result by loading the model
	print('Loading the trained model and testing it out!')
	model = PPO2.load(save_path)
	flow_params = get_flow_params(os.path.join(path, result_name) + '.json')
	flow_params['sim'].render = True
	env = env_constructor(params=flow_params, version=0)()
	# The algorithms require a vectorized environment to run
	eval_env = DummyVecEnv([lambda: env])
	obs = eval_env.reset()
	reward = 0
	for _ in range(flow_params['env'].horizon):
		action, _states = model.predict(obs)
		obs, rewards, dones, info = eval_env.step(action)
		reward += rewards
	print('the final reward is {}'.format(reward))


def main(args):
	"""Perform the training operations."""
	# Parse script-level arguments (not including package arguments).
	flags = parse_args(args)

	# Import the sub-module containing the specified exp_config and determine
	# whether the environment is single agent or multi-agent.
	if flags.multi:
		assert flags.rl_trainer.lower() in ["rllib", "h-baselines"], \
			"Currently, multiagent experiments are only supported through "\
			"RLlib. Try running this experiment using RLlib: " \
			"'python train.py EXP_CONFIG'"

	if flags.sweep:
		flags.__dict__.update(SWEEP_CONFIG)
		config = flags.__dict__
		sweep_id = wandb.sweep(config, entity="aypan17", project="value-learning", group="traffic", sync_tensorboard=True)
		wandb.agent(sweep_id, train, count=flags.replicas)
	else:
		flags.__dict__.update(DEFAULT_CONFIG)
		config = flags.__dict__
		wandb.init(entity="aypan17", project="value-learning", group="traffic", config=config, sync_tensorboard=True)
		train()

def train():
	config = wandb.config

	# Import relevant information from the exp_config script.
	if config.multi:
		module = __import__(
			"flow_cfg.exp_configs.rl.multiagent", fromlist=[config.exp_config])
	else:
		module = __import__(
			"flow_cfg.exp_configs.rl.singleagent", fromlist=[config.exp_config])

	submodule = getattr(module, config.exp_config)

	# Perform the training operation.
	if config.rl_trainer.lower() == "rllib":
		train_rllib(submodule, config)
	elif config.rl_trainer.lower() == "stable-baselines":
		train_stable_baselines(submodule, config)
	elif config.rl_trainer.lower() == "h-baselines":
		train_h_baselines(config.exp_config, args, config.multi)
	else:
		raise ValueError("rl_trainer should be either 'rllib', 'h-baselines', "
						 "or 'stable-baselines'.")

SWEEP_CONFIG = {
			'method': 'random', #'grid'
			'parameters': {
				'epochs': {
					'values': [2, 5, 10]
				},
				'batch_size': {
					'values': [256, 128, 64, 32]
				},
				'dropout': {
					'values': [0.3, 0.4, 0.5]
				},
				'lr': {
					'values': [1e-2, 1e-3, 1e-4, 3e-4, 3e-5, 1e-5]
				},
				'd_model':{
					'values':[128,256,512]
				},
				'optimizer': {
					'values': ['adam', 'sgd']
				},
			}
		}

DEFAULT_CONFIG = {}

if __name__ == "__main__":
	print(sys.argv)
	np.seterr(all='raise')	
	main(sys.argv[1:])
