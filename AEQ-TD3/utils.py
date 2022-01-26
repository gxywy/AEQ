import numpy as np
import torch
import gym
import scipy.signal

class ReplayBuffer(object):
	def __init__(self, state_dim, action_dim, max_size=int(1e6)):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0

		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.not_done = np.zeros((max_size, 1))
		self.mjstate = [None] * max_size ##
		self.count = np.zeros((max_size, 1))  ##
		#self.total_count = 0 ##
		self.max_count = 0 ##
		self.mean_count = 0 ##

		self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


	def add(self, state, action, next_state, reward, done, mjstate):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.not_done[self.ptr] = 1. - done
		self.mjstate[self.ptr] = mjstate ##
		self.count[self.ptr] = 0 ##

		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)


	def sample(self, batch_size):
		ind = np.random.randint(0, self.size, size=batch_size)
		self.count[ind] += 1 ##
		#self.total_count += batch_size ##
		self.max_count = max(self.max_count, self.count.max()) ##
		self.mean_count = self.count.mean() ##

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.not_done[ind]).to(self.device),
			[self.mjstate[i] for i in ind],
			torch.FloatTensor(self.count[ind]).to(self.device),
		)

def get_mc_return(env_name, mjstates, policy, seed):
	real_qs = []
	print("Evaluating real q-value (mc_return)...")
	for mjstate in mjstates:
		real_q = eval_policy(policy=policy, env_name=env_name, saved_state=mjstate, seed=seed)
		real_qs.append(real_q)
	return real_qs

def eval_policy(policy, env_name, saved_state, seed):
	eval_env = gym.make(env_name)
	eval_env.reset()
	eval_env.sim.set_state(saved_state)
	eval_env.seed(seed + 100)

	all_reward = []
	state, done = eval_env.env._get_obs(), False
	while not done:
		action = policy.select_action(np.array(state))
		state, reward, done, _ = eval_env.step(action)
		all_reward.append(reward)
	mc_returns = discount_cumsum(all_reward)
	return mc_returns[0]

def discount_cumsum(x, discount=0.99):
	"""
	magic from rllab for computing discounted cumulative sums of vectors.

	input:
		vector x,
		[x0,
			x1,
			x2]

	output:
		[x0 + discount * x1 + discount^2 * x2,
			x1 + discount * x2,
			x2]
	"""
	return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]