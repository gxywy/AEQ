import copy
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def to_numpy(var):
	return var.cpu().data.numpy()

# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477

class Actor(nn.Module):
	def __init__(self, state_dim, action_dim, max_action):
		super().__init__()

		self.l1 = nn.Linear(state_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, action_dim)
		
		self.max_action = max_action
		

	def forward(self, state):
		a = F.relu(self.l1(state))
		a = F.relu(self.l2(a))
		return self.max_action * torch.tanh(self.l3(a))


class Critic(nn.Module):
	def __init__(self, state_dim, action_dim):
		super().__init__()

		self.l1 = nn.Linear(state_dim + action_dim, 256)
		self.l2 = nn.Linear(256, 256)
		self.l3 = nn.Linear(256, 1)


	def forward(self, state, action):
		sa = torch.cat([state, action], 1)

		q1 = F.relu(self.l1(sa))
		q1 = F.relu(self.l2(q1))
		q1 = self.l3(q1)

		return q1
	
	def get_params(self):
		"""
		Returns parameters of the actor
		"""
		return copy.deepcopy(np.hstack([to_numpy(v).flatten() for v in
								   self.parameters()]))

class TD3(object):
	def __init__(
		self,
		state_dim,
		action_dim,
		max_action,
		discount=0.99,
		tau=0.005,
		policy_noise=0.2,
		noise_clip=0.5,
		policy_freq=2,
		num_q=10,
		base_beta=0.5,
		scale_beta=0.5,
		prior_net=False,
		dual_actor=False,
		tune_beta=False,
		tune_beta_reversed=False,
		rand_beta=False,
		q_target="mean_std",
	):
		self.num_actor = 2 if dual_actor else 1
		self.actor_list, self.actor_target_list, self.actor_optimizer_list = [], [], []
		for _ in range(self.num_actor):
			new_actor = Actor(state_dim, action_dim, max_action).to(device)
			self.actor_list.append(new_actor)
			new_actor_target = copy.deepcopy(new_actor)
			self.actor_target_list.append(new_actor_target)
			self.actor_optimizer_list.append(torch.optim.Adam(new_actor.parameters(), lr=3e-4))

		self.num_q = num_q
		self.group_num_q = self.num_q
		self.q_net_list, self.q_target_net_list, self.q_optimizer_list = [], [], []
		for _ in range(self.num_q):
			new_q_net = Critic(state_dim, action_dim).to(device)
			self.q_net_list.append(new_q_net)
			new_q_target_net = copy.deepcopy(new_q_net)
			self.q_target_net_list.append(new_q_target_net)
			self.q_optimizer_list.append(torch.optim.Adam(new_q_net.parameters(), lr=3e-4))
		
		if prior_net:
			self.prior_net_list, self.prior_target_net_list = [], []
			for _ in range(2):
				new_prior_net = Critic(state_dim, action_dim).to(device)
				self.prior_net_list.append(new_prior_net)
				new_prior_target_net = copy.deepcopy(new_prior_net)
				self.prior_target_net_list.append(new_prior_target_net)
		
		self.max_action = max_action
		self.discount = discount
		self.tau = tau
		self.policy_noise = policy_noise
		self.noise_clip = noise_clip
		self.policy_freq = policy_freq

		self.prior_net = prior_net
		self.dual_actor = dual_actor
		self.tune_beta = tune_beta
		self.tune_beta_reversed = tune_beta_reversed
		self.rand_beta = rand_beta
		self.q_target = q_target
		print(f"prior_net:{prior_net}, dual_actor:{dual_actor}, tune_beta:{tune_beta}, tune_beta_reversed:{tune_beta_reversed}, rand_beta:{rand_beta}, q_target:{q_target}")

		self.base_beta = base_beta
		self.scale_beta = scale_beta
		self.beta = 0

		self.beta_mean = 0
		self.beta_max = 0
		self.beta_min = 0
		self.std_q_mean = 0
		self.std_q_max = 0
		self.std_q_min = 0
		self.mean_q_mean = 0
		self.uf_item = 0

		self.total_it = 0


	def select_action(self, state, cuda=False):
		if not cuda:
			state = torch.FloatTensor(state.reshape(1, -1)).to(device)

		if self.dual_actor:
			action1 = self.actor_list[0](state)
			action2 = self.actor_list[1](state)

			action_q_list1 = []
			for critic in self.q_net_list[:5]:
				action_q1 = critic(state, action1)
				action_q_list1.append(action_q1)
			action_q_cat1 = torch.cat(action_q_list1, dim=1)

			action_q_list2 = []
			for critic in self.q_net_list[5:]:
				action_q2 = critic(state, action2)
				action_q_list2.append(action_q2)
			action_q_cat2 = torch.cat(action_q_list2, dim=1)
			
			action = action1 if torch.max(action_q_cat1).cpu().data > torch.max(action_q_cat2).cpu().data else action2
			if not cuda:
				return action.cpu().data.numpy().flatten()
			else:
				return action
		else:
			if not cuda:
				return self.actor_list[0](state).cpu().data.numpy().flatten()
			else:
				return self.actor_list[0](state)


	def train_one_group(self, replay_buffer, batch_size=256):
		if self.dual_actor:
			self.group_num_q = 5
			self.policy_freq = 1
			self.train(replay_buffer, batch_size, slice(0, 5), 0)
			self.train(replay_buffer, batch_size, slice(5, 10), 1)
		else:
			self.train(replay_buffer, batch_size, slice(0, 10), 0)


	def train(self, replay_buffer, batch_size=256, q_net_slice=slice(0,10), actor_index=0):
		self.total_it += 1

		# Sample replay buffer 
		state, action, next_state, reward, not_done, mjstate, count = replay_buffer.sample(batch_size)

		with torch.no_grad():
			# Select action according to policy and add clipped noise
			noise = (
				torch.randn_like(action) * self.policy_noise
			).clamp(-self.noise_clip, self.noise_clip)
			
			next_action = (
				self.actor_target_list[actor_index](next_state) + noise
			).clamp(-self.max_action, self.max_action)
			
			if self.q_target == "redq":
				q_target_prediction_list = []
				for index in np.random.choice(self.num_q, 2, replace=False):
					if self.prior_net:
						#q_target_prediction_list.append(self.q_target_net_list[q_net_slice][index](next_state, next_action) + self.prior_target_net_list[q_net_slice][index](next_state, next_action))
						q_target_prediction_list.append(self.q_target_net_list[index](next_state, next_action) + self.prior_target_net_list[0 if index < 5 else 1](next_state, next_action)) # 与dual_actor无关
					else:
						q_target_prediction_list.append(self.q_target_net_list[index](next_state, next_action))
				q_target_prediction_cat = torch.cat(q_target_prediction_list, dim=1)
				random_min_q, min_indices = torch.min(q_target_prediction_cat, dim=1, keepdim=True)
				target_q_one = reward + not_done * self.discount * random_min_q
			else:
				q_target_prediction_list = []
				if self.prior_net:
					# for q_target_net, prior_target_net in zip(self.q_target_net_list[q_net_slice], self.prior_target_net_list[q_net_slice]):
					# 	q_target_prediction_list.append(q_target_net(next_state, next_action) + prior_target_net(next_state, next_action))
					for index, q_target_net in enumerate(self.q_target_net_list):
						q_target_prediction_list.append(q_target_net(next_state, next_action) + self.prior_target_net_list[0 if index < 5 else 1](next_state, next_action)) # 与dual_actor无关
				else:
					for q_target_net in self.q_target_net_list:
						q_target_prediction_list.append(q_target_net(next_state, next_action))
				q_target_prediction_cat = torch.cat(q_target_prediction_list, dim=1)

				min_q, _ = torch.min(q_target_prediction_cat, dim=1, keepdim=True)
				max_q, _ = torch.max(q_target_prediction_cat, dim=1, keepdim=True)
				mean_q = torch.mean(q_target_prediction_cat, dim=1, keepdim=True)
				std_q = torch.std(q_target_prediction_cat, dim=1, keepdim=True)

				#beta = np.random.randint(0, 30) / 100
				self.beta = (count - 1) / replay_buffer.max_count #(count - 1) * (replay_buffer.total_count / replay_buffer.size)
				#print(beta2.cpu().mean().item())
				if self.tune_beta:
					self.beta = self.base_beta + self.scale_beta * self.beta
				if self.tune_beta_reversed:
					self.beta = self.base_beta - self.scale_beta * self.beta
				if self.rand_beta:
					self.beta = torch.Tensor(np.random.randint(0, 100, size=[batch_size, 1]) / 100).cuda()#self.base_beta + self.scale_beta * torch.Tensor(np.random.randint(0, 100, size=[batch_size, 1]) / 100).cuda()
				
				self.beta_mean = self.beta.cpu().mean().item()
				self.beta_max = self.beta.cpu().max().item()
				self.beta_min = self.beta.cpu().min().item()
				self.uf_item = (self.beta * std_q).cpu().mean().item()
				self.std_q_mean = std_q.cpu().mean().item()
				self.std_q_max = std_q.cpu().max().item()
				self.std_q_min = std_q.cpu().min().item()
				self.mean_q_mean = mean_q.cpu().mean().item()

				if self.q_target == "mean_std":
					target_q_one = reward + not_done * self.discount * (mean_q - self.beta * std_q)
				elif self.q_target == "min":
					target_q_one = reward + not_done * self.discount * min_q
				elif self.q_target == "max":
					target_q_one = reward + not_done * self.discount * max_q
				elif self.q_target == "mean":
					target_q_one = reward + not_done * self.discount * mean_q
				elif self.q_target == "std":
					target_q_one = reward + not_done * self.discount * (mean_q - self.base_beta * std_q)
			target_q = target_q_one.expand((-1, self.group_num_q))
		
		# Get current Q estimates
		q_prediction_list = []
		if self.prior_net:
			# for q_net, prior_net in zip(self.q_net_list[q_net_slice], self.prior_net_list[q_net_slice]):
			# 	q_prediction_list.append(q_net(state, action) + prior_net(state, action).detach())
			for index, q_net in enumerate(self.q_net_list[q_net_slice]):
				if self.dual_actor:
					q_prediction_list.append(q_net(state, action) + self.prior_net_list[actor_index](state, action).detach())
				else:
					q_prediction_list.append(q_net(state, action) + self.prior_net_list[0 if index < 5 else 1](state, action).detach())
		else:
			for q_net in self.q_net_list[q_net_slice]:
				q_prediction_list.append(q_net(state, action))
		q_prediction_cat = torch.cat(q_prediction_list, dim=1)

		critic_loss = F.mse_loss(q_prediction_cat, target_q)
		
		# Optimize the critic
		for optimizer in self.q_optimizer_list:
			optimizer.zero_grad()
		critic_loss.backward()

		for optimizer in self.q_optimizer_list:
			optimizer.step()

		# Delayed policy updates
		if self.total_it % self.policy_freq == 0:

			# Compute actor losse
			q_actor_prediction_list = []
			if self.prior_net:
				for prior_net in self.prior_net_list:
					prior_net.requires_grad_(False)
				for index, q_net in enumerate(self.q_net_list[q_net_slice]):
					q_net.requires_grad_(False)	
					if self.dual_actor:
						q_actor_prediction_list.append(q_net(state, self.actor_list[actor_index](state)) + self.prior_net_list[actor_index](state, action).detach())
					else:
						q_actor_prediction_list.append(q_net(state, self.actor_list[actor_index](state)) + self.prior_net_list[0 if index < 5 else 1](state, self.actor_list[actor_index](state)))
			else:
				for q_net in self.q_net_list[q_net_slice]:
					q_net.requires_grad_(False)
					q_actor_prediction_list.append(q_net(state, self.actor_list[actor_index](state)))
			q_actor_prediction_list = torch.cat(q_actor_prediction_list, dim=1)
			avg_q_actor = torch.mean(q_actor_prediction_list, dim=1, keepdim=True)
			actor_loss = -avg_q_actor.mean()
			
			# Optimize the actor 
			self.actor_optimizer_list[actor_index].zero_grad()
			actor_loss.backward()
			self.actor_optimizer_list[actor_index].step()

			# Update the frozen target models
			for q_net, q_target_net in zip(self.q_net_list[q_net_slice], self.q_target_net_list[q_net_slice]):
				for param, target_param in zip(q_net.parameters(), q_target_net.parameters()):
					target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

			for param, target_param in zip(self.actor_list[actor_index].parameters(), self.actor_target_list[actor_index].parameters()):
				target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
		
		for q_net in self.q_net_list[q_net_slice]:
			q_net.requires_grad_(True)
		
		if self.prior_net:
			for prior_net in self.prior_net_list:
				prior_net.requires_grad_(True)
	

	def log_q_estimate(self, replay_buffer, logger=None, env_name=None, seed=0, t = 0):
		state, _, _, _, _, mjstate, _ = replay_buffer.sample(1000)

		mc_returns = utils.get_mc_return(env_name, mjstate, self, seed)

		q_prediction_list = []
		action = self.select_action(state, cuda=True)
		if self.prior_net:
			for index, q_net in enumerate(self.q_net_list):
				q_prediction_list.append(q_net(state, action) + self.prior_net_list[0 if index < 5 else 1](state, action).detach()) # 与dual_actor无关
		else:
			for q_net in self.q_net_list:
				q_prediction_list.append(q_net(state, action))
		q_prediction_cat = torch.cat(q_prediction_list, dim=1)

		mean_eval_q = torch.mean(q_prediction_cat, dim=1, keepdim=True)
		bias = mean_eval_q.detach().cpu().numpy() - mc_returns

		logger.update(fieldvalues=[mean_eval_q.cpu().mean().item(), np.mean(mc_returns), np.abs(np.mean(bias)), np.abs(np.mean(bias / mc_returns)), np.std(bias)], total_steps=(t + 1))