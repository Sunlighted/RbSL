import threading
import numpy as np
import pickle

"""
the replay buffer here is basically from the openai baselines code

"""
class replay_buffer:
    def __init__(self, env_params, buffer_size, sample_func):
        self.env_params = env_params
        self.T = env_params['max_timesteps']
        self.size = buffer_size // self.T
        # memory management
        self.current_size = 0
        self.n_transitions_stored = 0
        self.sample_func = sample_func
        # create the buffer to store info
        self.buffers = {'obs': np.empty([self.size, self.T, self.env_params['obs']]),
                        'ag': np.empty([self.size, self.T, self.env_params['goal']]),
                        'g': np.empty([self.size, self.T - 1, self.env_params['goal']]),
                        'actions': np.empty([self.size, self.T - 1, self.env_params['action']]),
                        'r': np.empty([self.size, self.T - 1, 1]),
                        'costs': np.empty([self.size, self.T - 1, 1]),
                        }
        # thread lock
        self.key_map = {'o': 'obs', 'ag': 'ag', 'g': 'g', 'u':'actions', 'r':'r', 'c':'costs'}
        self.lock = threading.Lock()
    
    # store the episode
    def store_episode(self, episode_batch):
        mb_obs, mb_ag, mb_g, mb_actions, mb_r, mb_c = episode_batch
        batch_size = mb_obs.shape[0]
        with self.lock:
            idxs = self._get_storage_idx(inc=batch_size)
            # store the informations
            self.buffers['obs'][idxs] = mb_obs
            self.buffers['ag'][idxs] = mb_ag
            self.buffers['g'][idxs] = mb_g
            self.buffers['actions'][idxs] = mb_actions
            self.buffers['r'][idxs] = mb_r
            self.buffers['costs'][idxs] = mb_c
            self.n_transitions_stored += self.T * batch_size

    def shuffle_goals(self):
        np.random.shuffle(self.buffers['g'][:self.current_size])
    
    # sample the data from the replay buffer
    def sample(self, batch_size, future_p=None):
        temp_buffers = {}
        with self.lock:
            for key in self.buffers.keys():
                temp_buffers[key] = self.buffers[key][:self.current_size]
        temp_buffers['initial_obs'] = temp_buffers['obs'][:, :1, :]
        temp_buffers['obs_next'] = temp_buffers['obs'][:, 1:, :]
        temp_buffers['ag_next'] = temp_buffers['ag'][:, 1:, :]
        # sample transitions
        transitions = self.sample_func(temp_buffers, batch_size, future_p=future_p)
        return transitions

    def _get_storage_idx(self, inc=None):
        inc = inc or 1
        if self.current_size+inc <= self.size:
            idx = np.arange(self.current_size, self.current_size+inc)
        elif self.current_size < self.size:
            overflow = inc - (self.size - self.current_size)
            idx_a = np.arange(self.current_size, self.size)
            idx_b = np.random.randint(0, self.current_size, overflow)
            idx = np.concatenate([idx_a, idx_b])
        else:
            idx = np.random.randint(0, self.size, inc)
        self.current_size = min(self.size, self.current_size+inc)
        if inc == 1:
            idx = idx[0]
        return idx

    def save(self, path):
        with open(path, "wb") as fp:
            data = {} 
            for key in self.key_map:
                data[key] = self.buffers[self.key_map[key]][:self.n_transitions_stored]
            pickle.dump(data, fp)

    def load(self, path, percent=1.0):
        with open(path, "rb") as fp:  
            data = pickle.load(fp)
            size = data['o'].shape[0]
            self.current_size = int(size * percent)
            # if size > self.size:
            #     self.buffers = {key: np.empty([size, *shape]) for key, shape in self.buffer_shapes.items()}
            #     self.size = size

            for key in data.keys():
                self.buffers[self.key_map[key]][:self.current_size] = data[key][:self.current_size]

    def load_mixture(self, path_expert, path_random, expert_percent=0.1, random_percent=0.9, args=None):
        # 0 <= expert_percent <= 1, same for random_percent
        
        with open(path_expert, "rb") as fp_expert:  
            with open(path_random, "rb") as fp_random:  
                data_expert = pickle.load(fp_expert)  
                data_random = pickle.load(fp_random)  
                size_expert = data_expert['o'].shape[0]
                size_random = data_random['o'].shape[0]
                assert(size_expert == size_random)
                self.current_size = int(size_expert*expert_percent + size_random*random_percent)
                size = self.current_size
                split_point = int(size_expert*expert_percent)
                # if size > self.size:
                #     self.buffers = {key: np.empty([size, *shape]) for key, shape in self.buffer_shapes.items()}
                #     self.size = size
                    
                for key in data_expert.keys():
                    self.buffers[self.key_map[key]][:split_point] = data_expert[key][:split_point]
                    self.buffers[self.key_map[key]][split_point:size] = data_random[key][:size - split_point]

    def load_filter(self, path, percent=1.0):
        with open(path, "rb") as fp:  
            data = pickle.load(fp)
            size = data['o'].shape[0]
            self.current_size = int(size * percent)
            data1 = {'o': data['o'][:self.current_size],
                        'ag': data['ag'][:self.current_size],
                        'g': data['g'][:self.current_size],
                        'u': data['u'][:self.current_size],
                        'r': data['r'][:self.current_size],
                        'c': data['c'][:self.current_size],
                        }
            filter_data = {}
            
            size = 0
            x = np.zeros(self.T - 1).reshape(self.T - 1,1)
            for i in range(data1['c'].shape[0]):
                if not (data1['c'][i] == x).all():
                    size += 1
            for key in data.keys():
                filter_data[key] = np.empty([size, data1[key].shape[1], data1[key].shape[2]])

            index=0
            for i in range(data1['c'].shape[0]):
                if not (data1['c'][i] == x).all():
                    for key in data1.keys():
                        filter_data[key][index] = data1[key][i]
                    index += 1

            self.current_size = filter_data['o'].shape[0]

            for key in data.keys():
                self.buffers[self.key_map[key]] = filter_data[key]
            