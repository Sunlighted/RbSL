import numpy as np 
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal, Categorical


"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""

# Inverse tanh torch function
def atanh(z):
    return 0.5 * (torch.log(1 + z) - torch.log(1 - z))

# Initialize Policy weights
def weights_init_(m):
    if isinstance(m, nn.Linear):
        torch.nn.init.xavier_uniform_(m.weight, gain=1)
        torch.nn.init.constant_(m.bias, 0)

# define the actor network
class actor(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions
    
class actorg(nn.Module):
    def __init__(self, env_params):
        super(actor, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        
        self.apply(weights_init_)

    def forward(self, x, act):
        x = F.relu(self.fc1(torch.cat[x, act]))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = 0.05 * self.max_action * torch.tanh(self.action_out(x))
        a = (act + actions).clamp(-self.max_action, self.max_action)

        return a

# define the actor network
class actor_inverse(nn.Module):
    def __init__(self, env_params):
        super(actor_inverse, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(3*env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.max_action * torch.tanh(self.action_out(x))

        return actions
    
class actorgauss(nn.Module):
    def __init__(self, env_params):
        super(actorgauss, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['action'])

        self.lfc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.lfc2 = nn.Linear(256, 256)
        self.lfc3 = nn.Linear(256, 256)
        self.log_std_out = nn.Linear(256, env_params['action'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x1 = x
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        mean = self.action_out(x)
        actions = self.max_action * torch.tanh(mean)

        x1 = F.relu(self.lfc1(x1))
        x1 = F.relu(self.lfc2(x1))
        x1 = F.relu(self.lfc3(x1))
        log_std = torch.clamp(self.log_std_out(x1), -20, 2)
        std = torch.exp(log_std)
        action_distribution = Normal(mean, std)

        return actions, action_distribution

# define the planner network 
class planner(nn.Module):
    def __init__(self, env_params):
        super(planner, self).__init__()
        self.fc1 = nn.Linear(2 * env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.action_out = nn.Linear(256, env_params['goal'])
        
        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        actions = self.action_out(x)

        return actions

class critic(nn.Module):
    def __init__(self, env_params, activation=None):
        super(critic, self).__init__()
        self.activation = activation
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        if self.activation == 'sigmoid':
            q_value = torch.sigmoid(q_value)
        return q_value

class doublecritic(nn.Module):
    def __init__(self, env_params):
        super(doublecritic, self).__init__()
        self.max_action = env_params['action_max']
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q1_out = nn.Linear(256, 1)

        self.fc4 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], 256)
        self.fc5 = nn.Linear(256, 256)
        self.fc6 = nn.Linear(256, 256)
        self.q2_out = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, x, actions):
        x = torch.cat([x, actions / self.max_action], dim=1)
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        q1_value = self.q1_out(x1)

        x2 = F.relu(self.fc4(x))
        x2 = F.relu(self.fc5(x2))
        x2 = F.relu(self.fc6(x2))
        q2_value = self.q2_out(x2)

        return q1_value, q2_value
    
    def Q1(self, x, action):
        x = torch.cat([x, action / self.max_action], dim=1)
        x1 = F.relu(self.fc1(x))
        x1 = F.relu(self.fc2(x1))
        x1 = F.relu(self.fc3(x1))
        q1_value = self.q1_out(x1)
        
        return q1_value
    
    def Q_min(self, x, action):
        q1, q2 = self.forward(x, action)
        return torch.min(q1, q2)

class value(nn.Module):
    def __init__(self, env_params):
        super(value, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, 1)

        self.apply(weights_init_)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)

        return q_value
    
class VAE(nn.Module):
    """
    Variational Auto-Encoder
    
    Args:
        obs_dim (int): The dimension of the observation space.
        act_dim (int): The dimension of the action space.
        hidden_size (int): The number of hidden units in the encoder and decoder networks.
        latent_dim (int): The dimensionality of the latent space.
        act_lim (float): The upper limit of the action space.
        device (str): The device to use for computation (cpu or cuda).
    """

    def __init__(self, env_params, hidden_size = 256):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['action'], hidden_size)
        self.e2 = nn.Linear(hidden_size, hidden_size)

        self.mean = nn.Linear(hidden_size, 2 * env_params['action'])
        self.log_std = nn.Linear(hidden_size, 2 * env_params['action'])

        self.d1 = nn.Linear(env_params['obs'] + env_params['goal'] + 2 * env_params['action'], hidden_size)
        self.d2 = nn.Linear(hidden_size, hidden_size)
        self.d3 = nn.Linear(hidden_size, env_params['action'])

        self.act_lim = env_params['action']
        self.env_params = env_params

    def forward(self, obs, act):
        z = F.relu(self.e1(torch.cat([obs, act], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)

        u = self.decode(obs, z)
        return u, mean, std

    def decode(self, obs, z=None):
        if z is None:
            z = torch.randn((obs.shape[0], 2 * self.env_params['action'])).clamp(-0.5, 0.5)

        a = F.relu(self.d1(torch.cat([obs, z], 1)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a))

    # for BEARL only
    def decode_multiple(self, obs, z=None, num_decode=10):
        if z is None:
            z = torch.randn(
                (obs.shape[0], num_decode, 2 * self.env_params['action'])).clamp(-0.5, 0.5)

        a = F.relu(
            self.d1(
                torch.cat(
                    [obs.unsqueeze(0).repeat(num_decode, 1, 1).permute(1, 0, 2), z], 2)))
        a = F.relu(self.d2(a))
        return torch.tanh(self.d3(a)), self.d3(a)

class LagrangianPIDController:
    '''
    Lagrangian multiplier controller
    
    Args:
        KP (float): The proportional gain.
        KI (float): The integral gain.
        KD (float): The derivative gain.
        thres (float): The setpoint for the controller.
    '''

    def __init__(self, KP, KI, KD, thres) -> None:
        super().__init__()
        self.KP = KP
        self.KI = KI
        self.KD = KD
        self.thres = thres
        self.error_old = 0
        self.error_integral = 0

    def control(self, qc):
        '''
        @param qc [batch,]
        '''
        error_new = torch.mean(qc - self.thres)  # [batch]
        error_diff = F.relu(error_new - self.error_old)
        self.error_integral = torch.mean(F.relu(self.error_integral + error_new))
        self.error_old = error_new

        multiplier = F.relu(self.KP * F.relu(error_new) + self.KI * self.error_integral +
                            self.KD * error_diff)
        return torch.mean(multiplier)