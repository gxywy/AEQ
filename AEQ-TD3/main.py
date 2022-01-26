import numpy as np
import torch
import gym
import argparse
import os

from torch import cuda

import utils
import TD3
from rl_plotter.logger import Logger

# Runs policy for X episodes and returns average reward
# A fixed seed is used for the eval environment
def eval_policy(policy, env_name, seed, eval_episodes=10):
	eval_env = gym.make(env_name)
	eval_env.seed(seed + 100)

	all_reward = []
	for _ in range(eval_episodes):
		ep_reward = 0.
		state, done = eval_env.reset(), False
		while not done:
			action = policy.select_action(np.array(state))
			state, reward, done, _ = eval_env.step(action)
			ep_reward += reward
		all_reward.append(ep_reward)

	print("---------------------------------------")
	print(f"Evaluation over {eval_episodes} episodes: {np.mean(all_reward):.3f}")
	print("---------------------------------------")
	return all_reward

def set_seed(epoch, seed):
	seed_shift = epoch * 9999
	mod_value = 999999
	env_seed = (seed + seed_shift) % mod_value
	eval_env_seed = (seed + 10000 + seed_shift) % mod_value
	q_eval_env_seed = (seed + 20000 + seed_shift) % mod_value
	torch.manual_seed(env_seed)
	np.random.seed(env_seed)
	return env_seed, eval_env_seed, q_eval_env_seed

if __name__ == "__main__":
	
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="HalfCheetah-v2")          # OpenAI gym environment name
	parser.add_argument("--seed", default=0, type=int)              # Sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--start_timesteps", default=25e3, type=int)# Time steps initial random policy is used
	parser.add_argument("--eval_freq", default=5e3, type=int)       # How often (time steps) we evaluate
	parser.add_argument("--max_timesteps", default=2e6, type=int)   # Max time steps to run environment
	parser.add_argument("--expl_noise", default=0.1)                # Std of Gaussian exploration noise
	parser.add_argument("--batch_size", default=256, type=int)      # Batch size for both actor and critic
	parser.add_argument("--discount", default=0.99)                 # Discount factor
	parser.add_argument("--tau", default=0.005)                     # Target network update rate
	parser.add_argument("--policy_noise", default=0.2)              # Noise added to target policy during critic update
	parser.add_argument("--noise_clip", default=0.5)                # Range to clip target policy noise
	parser.add_argument("--policy_freq", default=2, type=int)       # Frequency of delayed policy updates
	# Additional arguments
	parser.add_argument("--num_q", default=10, type=int)
	parser.add_argument("--base_beta", default=0.5, type=float)
	parser.add_argument("--scale_beta", default=0.5, type=float)
	parser.add_argument("--prior_net", action="store_true")
	parser.add_argument("--dual_actor", action="store_true")
	parser.add_argument("--tune_beta", action="store_true")
	parser.add_argument("--tune_beta_reversed", action="store_true")
	parser.add_argument("--rand_beta", action="store_true")
	parser.add_argument("--ep_seed", action="store_true")
	parser.add_argument("--q_target", default="mean_std")
	parser.add_argument("--log_q", action="store_true")
	parser.add_argument("--log_beta", action="store_true")
	parser.add_argument("--log_parm", action="store_true")
	parser.add_argument("--log_ufq", action="store_true")
	parser.add_argument("--debug", action="store_true")
	args = parser.parse_args()

	if args.q_target == "redq":
		exp_name = f"REDQ_{args.num_q}(TD3)"
	elif args.q_target == "mean_std":
		exp_name = f"UFQ_{args.num_q}(TD3)"
	else:
		exp_name = f"TD3_{args.num_q}({args.q_target})"

	file_name = f"{exp_name}_{args.env}_{args.seed}"
	print("---------------------------------------")
	print(f"Policy: {exp_name}, Env: {args.env}, Seed: {args.seed}")
	print("---------------------------------------")

	if args.prior_net:
		exp_name += "+pn"
	if args.dual_actor:
		exp_name += "+da"
	if args.tune_beta:
		exp_name += "+tb"
	if args.tune_beta_reversed:
		exp_name += "+tbr"
	if args.rand_beta:
		exp_name += "+rb"
	if args.ep_seed:
		exp_name += "+eps"

	logger = Logger(log_dir="./results", exp_name=exp_name, env_name=args.env, seed=args.seed, config=locals(), debug=args.debug)
	if args.log_q:
		q_logger = logger.new_custom_logger(filename="q.csv", fieldnames=["target_q", "real_q", "bias", "bias_norm", "bias_std"])
	if args.log_parm:
		parm_logger = logger.new_custom_logger(filename="q_parm.csv", fieldnames=[f"q{i}_parm_{j}" for i in range(10) for j in range(2)])
	if args.log_beta:
		fields = []
		for i in range(5):
			fields += [f"beta{i}", f"count{i}", f"q_mean{i}", f"q_std{i}", f"q_max{i}", f"q_min{i}"]
		beta_logger = logger.new_custom_logger(filename="beta.csv", fieldnames=fields)
	if args.log_ufq:
		ufq_logger = logger.new_custom_logger(filename="ufq.csv", fieldnames=["uf_item", "std_q_mean", "std_q_max", "std_q_min", "mean_q_mean", "beta", "beta_max", "beta_min", "count_mean", "count_max"])

	env = gym.make(args.env)

	# Set seeds
	env_seed, eval_env_seed, q_eval_env_seed = args.seed, args.seed, args.seed
	if args.ep_seed:
		env_seed, eval_env_seed, q_eval_env_seed = set_seed(0, args.seed)
		env.seed(env_seed)
		env.action_space.seed(env_seed)
	else:
		env.seed(env_seed)
		env.action_space.seed(env_seed)
		torch.manual_seed(args.seed)
		np.random.seed(args.seed)
	
	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.shape[0] 
	max_action = float(env.action_space.high[0])

	kwargs = {
		"state_dim": state_dim,
		"action_dim": action_dim,
		"max_action": max_action,
		"discount": args.discount,
		"tau": args.tau,
		"policy_noise": args.policy_noise * max_action,
		"noise_clip": args.noise_clip * max_action,
		"policy_freq": args.policy_freq,

		"num_q": args.num_q,
		"base_beta": args.base_beta,
		"scale_beta": args.scale_beta,
		"prior_net": args.prior_net,
		"dual_actor": args.dual_actor,
		"tune_beta": args.tune_beta,
		"tune_beta_reversed": args.tune_beta_reversed,
		"rand_beta": args.rand_beta,
		"q_target": args.q_target,
	}

	policy = TD3.TD3(**kwargs)

	replay_buffer = utils.ReplayBuffer(state_dim, action_dim)
	
	# Evaluate untrained policy
	logger.update(eval_policy(policy, args.env, eval_env_seed), 0)

	state, done = env.reset(), False
	episode_reward = 0
	episode_timesteps = 0
	episode_num = 0
	init_q_estimate_flag = True

	for t in range(int(args.max_timesteps)):
		
		episode_timesteps += 1

		# mjstate = env.sim.get_state()

		# Select action randomly or according to policy
		if t < args.start_timesteps:
			action = env.action_space.sample()
		else:
			action = (
				policy.select_action(np.array(state))
				+ np.random.normal(0, max_action * args.expl_noise, size=action_dim)
			).clip(-max_action, max_action)

		# Perform action
		next_state, reward, done, _ = env.step(action) 
		done_bool = float(done) if episode_timesteps < env._max_episode_steps else 0
		
		# Store data in replay buffer
		mjstate = env.sim.get_state()
		replay_buffer.add(state, action, next_state, reward, done_bool, mjstate)

		state = next_state
		episode_reward += reward

		# Train agent after collecting sufficient data
		if t >= args.start_timesteps:
			policy.train_one_group(replay_buffer, args.batch_size)

		if (t > 1000 and init_q_estimate_flag and args.log_q) or ((t + 1) % 50000 == 0 and args.log_q):
			policy.log_q_estimate(replay_buffer, q_logger, args.env, q_eval_env_seed, t)
			init_q_estimate_flag = False

		if done: 
			# +1 to account for 0 indexing. +0 on ep_timesteps since it will increment +1 even if done=True
			print(f"Total T: {t+1} Episode Num: {episode_num+1} Episode T: {episode_timesteps} Reward: {episode_reward:.3f}")
			# Reset environment
			state, done = env.reset(), False
			episode_reward = 0
			episode_timesteps = 0
			episode_num += 1

			if args.ep_seed:
				env_seed, eval_env_seed, q_eval_env_seed = set_seed(episode_num, args.seed)
				env.seed(env_seed)
				env.action_space.seed(env_seed)

		# Evaluate episode
		if (t + 1) % args.eval_freq == 0:
			all_reward = eval_policy(policy, args.env, eval_env_seed)
			logger.update(all_reward, t + 1)

			if args.log_parm:
				q_parm_list = []
				for q_net in policy.q_net_list:
					q_parm_list.append(q_net.get_params()[0])
					q_parm_list.append(q_net.get_params()[1])
				parm_logger.update(q_parm_list, t + 1)

		if (t + 1) % 10000 == 0 and args.log_beta:
			beta_list = []
			for i in range(5):
				if (t + 1) >= i * 100000:
					count = replay_buffer.count[i * 100000].item()
					state_beta = torch.FloatTensor(replay_buffer.state[i * 100000]).unsqueeze(0).cuda()
					action_beta = torch.FloatTensor(replay_buffer.action[i * 100000]).unsqueeze(0).cuda()
					beta = (count - 1) / replay_buffer.max_count if replay_buffer.max_count != 0 else 0
					q_list = []
					for critic in policy.q_net_list:
						q_list.append(critic(state_beta, action_beta).detach())
					q_list_cat = torch.cat(q_list, dim=1)
					min_q, _ = torch.min(q_list_cat, dim=1, keepdim=True)
					max_q, _ = torch.max(q_list_cat, dim=1, keepdim=True)
					mean_q = torch.mean(q_list_cat, dim=1, keepdim=True)
					std_q = torch.std(q_list_cat, dim=1, keepdim=True)
					beta_list += [beta, count, mean_q.cpu().item(), std_q.cpu().item(), max_q.cpu().item(), min_q.cpu().item()]
				else:
					beta_list += [0, 0, 0, 0, 0, 0]
			beta_logger.update(beta_list, t + 1)
		
		if (t + 1) % 10000 == 0 and args.log_ufq:
			ufq_logger.update([policy.uf_item, policy.std_q_mean, policy.std_q_max, policy.std_q_min, policy.mean_q_mean, policy.beta_mean, policy.beta_max, policy.beta_min, replay_buffer.mean_count, replay_buffer.max_count], total_steps=(t + 1))
