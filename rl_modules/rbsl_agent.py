import torch
import torch.nn.functional as F
import os
from datetime import datetime
import numpy as np
from mpi4py import MPI
from mpi_utils.mpi_utils import sync_networks, sync_grads
from rl_modules.base_agent_bi import BaseAgent
from rl_modules.models import actor, critic, LagrangianPIDController
from rl_modules.discriminator import Discriminator

"""
GCSL (MPI-version)

"""
class RBSL(BaseAgent):
    def __init__(self, args, env, env_params):
        super().__init__(args, env, env_params) 
        # create the network
        self.actor_network = actor(env_params)
        self.critic_network = critic(env_params)
        self.safety_actor_network = actor(env_params)
        self.safety_critic_network = critic(env_params)
        self.safety_cost_critic_network = critic(env_params)
        # sync the networks across the cpus
        sync_networks(self.actor_network)
        sync_networks(self.critic_network)
        sync_networks(self.safety_actor_network)
        sync_networks(self.safety_critic_network)
        sync_networks(self.safety_cost_critic_network)
        # build up the target network
        self.actor_target_network = actor(env_params)
        self.critic_target_network = critic(env_params)
        self.safety_actor_target_network = actor(env_params)
        self.safety_critic_target_network = critic(env_params)
        self.safety_cost_critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.actor_target_network.load_state_dict(self.actor_network.state_dict())
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        self.safety_actor_target_network.load_state_dict(self.safety_actor_network.state_dict())
        self.safety_critic_target_network.load_state_dict(self.safety_critic_network.state_dict())
        self.safety_cost_critic_target_network.load_state_dict(self.safety_cost_critic_network.state_dict())

        self.q_thres = 1.5 * (1 - self.args.gamma**50) / (1 - self.args.gamma) / 50

        # if use gpu
        if self.args.cuda:
            self.actor_network.cuda()
            self.critic_network.cuda()
            self.actor_target_network.cuda()
            self.critic_target_network.cuda()
            self.safety_actor_network.cuda()
            self.safety_actor_target_network.cuda()
            self.safety_critic_network.cuda()
            self.safety_critic_target_network.cuda()
            self.safety_cost_critic_network.cuda()
            self.safety_cost_critic_target_network.cuda()

        # create the optimizer
        self.actor_optim = torch.optim.Adam(self.actor_network.parameters(), lr=self.args.lr_actor)
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr_critic)
        self.safety_actor_optim = torch.optim.Adam(self.safety_actor_network.parameters(), lr=self.args.lr_safety_actor)
        self.safety_critic_optim = torch.optim.Adam(self.safety_critic_network.parameters(), lr=self.args.lr_safety_critic)
        self.safety_cost_critic_optim = torch.optim.Adam(self.safety_cost_critic_network.parameters(), lr=self.args.lr_safety_critic)

    # soft update
    def _soft_update(self):
        self._soft_update_target_network(self.actor_target_network, self.actor_network)
        self._soft_update_target_network(self.critic_target_network, self.critic_network)
        self._soft_update_target_network(self.safety_actor_target_network, self.safety_actor_network)
        self._soft_update_target_network(self.safety_critic_target_network, self.safety_critic_network)
        self._soft_update_target_network(self.safety_cost_critic_target_network, self.safety_cost_critic_network)

    # this function will choose action for the agent and do the exploration
    def _stochastic_actions(self, input_tensor):
        pi = self.actor_network(input_tensor)
        action = pi.cpu().numpy().squeeze()

        # add the gaussian
        action += self.args.noise_eps * self.env_params['action_max'] * np.random.randn(*action.shape)
        action = np.clip(action, -self.env_params['action_max'], self.env_params['action_max'])
        # random actions...
        random_actions = np.random.uniform(low=-self.env_params['action_max'], high=self.env_params['action_max'], \
                                            size=self.env_params['action'])
        # choose if use the random actions
        action += np.random.binomial(1, self.args.random_eps, 1)[0] * (random_actions - action)
        return action
    
    def _deterministic_action(self, input_tensor):
        action = self.actor_network(input_tensor)
        recovery_action = self.safety_actor_network(input_tensor)
        if self.args.cuda:
            input_tensor = input_tensor.cuda()
        qc = self.safety_cost_critic_network(input_tensor, action)
        if qc > self.q_thres:
            action = recovery_action
        return action
    
    # update the safety network
    def _update_safety_network(self, mutip, future_p=None):
        # sample the episodes
        sample_batch = self.sample_batch_safety(future_p=future_p)
        transitions = sample_batch['transitions'] 

        self.multiplier = mutip

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        g_norm = self.g_norm.normalize(transitions['g'])
        
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 
        c_tensor = torch.tensor(transitions['costs'], dtype=torch.float32) 

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()
            c_tensor = c_tensor.cuda()

        # calculate the target Q value function
        with torch.no_grad():
            # do the normalization
            # concatenate the stuffs
            actions_next = self.safety_actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.safety_critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * (q_next_value <= self.q_thres) * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)
        # the q loss
        real_q_value = self.safety_critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # add AM penalty
        num_random_actions = 10
        random_actions_tensor = torch.FloatTensor(q_next_value.shape[0] * num_random_actions, actions_tensor.shape[-1]).uniform_(-1, 1).to(actions_tensor.device)
        inputs_norm_tensor_repeat = inputs_norm_tensor.repeat_interleave(num_random_actions, axis=0)

        q_random_actions = self.safety_critic_network(inputs_norm_tensor_repeat, random_actions_tensor)
        q_random_actions = q_random_actions.reshape(q_next_value.shape[0], -1)

        # sample according to exp(Q)
        sampled_random_actions = torch.distributions.Categorical(logits=q_random_actions.detach()).sample()
        critic_loss_AM = q_random_actions[torch.arange(q_random_actions.shape[0]), sampled_random_actions].mean()
        critic_loss += critic_loss_AM

        with torch.no_grad():
            qc_next_value = self.safety_cost_critic_target_network(inputs_next_norm_tensor, actions_next)
            qc_next_value = qc_next_value.detach()
            target_qc_value = c_tensor + self.args.gamma * qc_next_value
            target_qc_value = target_qc_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_qc_value = torch.clamp(target_qc_value, 0, clip_return)

        # qc loss
        real_qc_value = self.safety_cost_critic_network(inputs_norm_tensor, actions_tensor)
        cost_critic_loss = (target_qc_value - real_qc_value).pow(2).mean()

        # the actor loss
        actions_real = self.safety_actor_network(inputs_norm_tensor)
        qc_val = self.safety_cost_critic_network(inputs_norm_tensor, actions_real)
        # actor_loss = -((qc_val <= self.q_thres) * self.critic_network(inputs_norm_tensor, actions_real)).mean()
        qc_penalty = ((qc_val - self.q_thres) * self.multiplier).mean()
        actor_loss = -(self.safety_critic_network(inputs_norm_tensor, actions_real)).mean() + qc_penalty
        
        # start to update the network
        self.safety_actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.safety_actor_network)
        self.safety_actor_optim.step()

        # update the critic_network
        self.safety_critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.safety_critic_network)
        self.safety_critic_optim.step()

        self.safety_cost_critic_optim.zero_grad()
        cost_critic_loss.backward()
        sync_grads(self.safety_cost_critic_network)
        self.safety_cost_critic_optim.step()

        results = {'Train/safety_critic_loss': critic_loss, 
                   'Train/safety_actor_loss': actor_loss,
                   'Train/safety_cost_critic_loss': cost_critic_loss,
                   'Train/real_q_value': (real_q_value).mean(),
                   'Train/real_qc_value': (real_qc_value).mean()}

        return results
    
    # update the network
    def _update_network(self, future_p=None):
        # sample the episodes
        sample_batch = self.sample_batch(future_p=future_p)
        transitions = sample_batch['transitions'] 

        # start to do the update
        obs_norm = self.o_norm.normalize(transitions['obs'])
        ag_norm = self.g_norm.normalize(transitions['ag'])
        g_norm = self.g_norm.normalize(transitions['g'])
        
        inputs_norm = np.concatenate([obs_norm, g_norm], axis=1)
        obs_next_norm = self.o_norm.normalize(transitions['obs_next'])
        ag_next_norm = self.g_norm.normalize(transitions['ag_next'])
        g_next_norm = self.g_norm.normalize(transitions['g_next'])
        inputs_next_norm = np.concatenate([obs_next_norm, g_next_norm], axis=1)
        
        # transfer them into the tensor
        inputs_norm_tensor = torch.tensor(inputs_norm, dtype=torch.float32)
        inputs_next_norm_tensor = torch.tensor(inputs_next_norm, dtype=torch.float32)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.float32)
        r_tensor = torch.tensor(transitions['r'], dtype=torch.float32) 

        if self.args.reward_type == 'positive':
            r_tensor = r_tensor + 1.
        elif self.args.reward_type == 'square':
            # Question: does it make sense to do this here?
            r_tensor = - torch.tensor(np.linalg.norm(ag_next_norm-g_norm, axis=1) ** 2, dtype=torch.float32).unsqueeze(1)

        if self.args.cuda:
            inputs_norm_tensor = inputs_norm_tensor.cuda()
            inputs_next_norm_tensor = inputs_next_norm_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            r_tensor = r_tensor.cuda()

        # Compute the actions
        actions_real = self.actor_network(inputs_norm_tensor)

        # calculate the target Q value function
        offset = sample_batch['future_offset']
        weights = pow(self.args.gamma, offset)  
        weights = torch.tensor(weights[:, None]).to(actions_tensor.device)
        with torch.no_grad():
            actions_next = self.actor_target_network(inputs_next_norm_tensor)
            q_next_value = self.critic_target_network(inputs_next_norm_tensor, actions_next)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, -clip_return, 0)

        # the q loss
        real_q_value = self.critic_network(inputs_norm_tensor, actions_tensor)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()

        # print(self.args.expert_percent == 0.5 or self.args.expert_percent == 1)

        if (self.args.expert_percent == 0.5 or self.args.expert_percent == 1):
            # add AM penalty
            num_random_actions = 10
            random_actions_tensor = torch.FloatTensor(q_next_value.shape[0] * num_random_actions, actions_tensor.shape[-1]).uniform_(-1, 1).to(actions_tensor.device)
            inputs_norm_tensor_repeat = inputs_norm_tensor.repeat_interleave(num_random_actions, axis=0)

            q_random_actions = self.safety_critic_network(inputs_norm_tensor_repeat, random_actions_tensor)
            q_random_actions = q_random_actions.reshape(q_next_value.shape[0], -1)

            # sample according to exp(Q)
            sampled_random_actions = torch.distributions.Categorical(logits=q_random_actions.detach()).sample()
            critic_loss_AM = q_random_actions[torch.arange(q_random_actions.shape[0]), sampled_random_actions].mean()
            critic_loss += critic_loss_AM

            # print(1)

        # Compute the advantage weighting
        with torch.no_grad():
            v = self.critic_network(inputs_norm_tensor, actions_real)
            v = torch.clamp(v, -clip_return, 0)
            # print(v.shape)
            adv = target_q_value - v
            adv = torch.clamp(torch.exp(adv.detach()), 0, 10)
            # print(adv.shape)
        weights = weights * adv
        # assert 0

        if self.args.env == 'FetchPickAndPlaceObstacle':
            if (self.args.expert_percent == 0.5 or self.args.expert_percent == 1):
                self.alpha = 1
            else:
                self.alpha = 0

        if self.args.env == 'FetchReachObstacle':
            if (self.args.expert_percent == 0.1 or self.args.expert_percent == 0):
                self.alpha = 1
            else:
                self.alpha = 0

        # print(self.args.expert_percent == 0.5)

        if self.args.env == 'FetchPushObstacle' or self.args.env == 'PandaPush':
            if self.args.expert_percent == 1:
                self.alpha = 5
            elif self.args.expert_percent == 0.5:
                self.alpha = 1
            else:
                self.alpha = 0

        if (self.args.env == 'FetchSlideObstacle'):
            if (self.args.expert_percent == 0.5 or self.args.expert_percent == 1):
                self.alpha = 2
            else:
                self.alpha = 0
        
        # print(self.alpha)

        real_q = self.critic_network(inputs_norm_tensor, actions_real)
        lmbda = self.alpha/real_q.abs().mean().detach()
        actor_loss = torch.mean(weights * torch.square(actions_real - actions_tensor)) - lmbda * (self.critic_network(inputs_norm_tensor, actions_real)).mean()
        # actor_loss = -(self.critic_network(inputs_norm_tensor, actions_real)).mean() 

        # update the actor network
        self.actor_optim.zero_grad()
        actor_loss.backward()
        sync_grads(self.actor_network)
        self.actor_optim.step()

        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        sync_grads(self.critic_network)
        self.critic_optim.step()

        results = {                   
            'Train/actor_loss': actor_loss,
            'Train/critic_loss': critic_loss,
            }

        return results
