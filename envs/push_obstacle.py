import os
from gym import utils
from gym.envs.robotics import rotations, robot_env, fetch_env
import gym.envs.robotics.utils as robot_utils
import numpy as np
import gym


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'push_obstacle.xml')

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)

class FetchPushObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
            'obstacle0:joint': [1.25, 0.53, 0.5, 1., 0., 0., 0.]
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=True, n_substeps=20,
            gripper_extra_height=0.0, target_in_the_air=False, target_offset=0.0,
            obj_range=0.15, target_range=0.27, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)
        self.penaltize_height = False

        self.pos_obstacle = self.sim.model.geom_pos[self.sim.model.geom_name2id('obstacle0')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('obstacle0')]

        self.cost_threshold = 0.05


    # def compute_reward(self, achieved_goal, goal, info):
    #     r = fetch_env.FetchEnv.compute_reward(self, achieved_goal, goal, info)
    #     if self.hack_obstacle:
    #         r += -0.1 * info['is_blocked']
    #     return r

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)

        # Randomize start position of object.
        if self.has_object:
            object_xpos = self.initial_gripper_xpos[:2]
            
            while np.linalg.norm(object_xpos - self.initial_gripper_xpos[:2]) < 0.1:
                object_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,self.obj_range, size=2)
            object_qpos = self.sim.data.get_joint_qpos('object0:joint')
            assert object_qpos.shape == (7,)
            object_qpos[:2] = object_xpos

            self.sim.data.set_joint_qpos('object0:joint', object_qpos)
            self.object_qpos = object_qpos

        obstacle_xpos = self.initial_gripper_xpos[:2]
        while np.linalg.norm(obstacle_xpos - self.initial_gripper_xpos[:2]) < 0.1 or (self.has_object and np.linalg.norm(object_xpos - obstacle_xpos) < 0.15):
            obstacle_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,self.obj_range, size=2)
        obstacle_qpos = self.sim.data.get_joint_qpos('obstacle0:joint')
        assert obstacle_qpos.shape == (7,)
        obstacle_qpos[:2] = obstacle_xpos
        obstacle_qpos[2] = self.height_offset
        self.sim.data.set_joint_qpos('obstacle0:joint', obstacle_qpos)
        self.obstacle_qpos = obstacle_qpos

        self.sim.forward()
        return True

    def _sample_goal(self):
        if self.has_object:
            object_xpos = self.sim.data.get_joint_qpos('object0:joint')[:2]
            obstacle_xpos = self.sim.data.get_joint_qpos('obstacle0:joint')[:2]
            obstacle_xpos_x = obstacle_xpos[0]
            obstacle_xpos_y = obstacle_xpos[1]
            object_xpos_x = object_xpos[0]
            object_xpos_y = object_xpos[1]
            vector_from_obstacle_to_object_x = (object_xpos_x - obstacle_xpos_x)
            vector_from_obstacle_to_object_y = (object_xpos_y - obstacle_xpos_y)

            goal = self.initial_gripper_xpos[:3]
            while np.linalg.norm(goal[:2] - self.initial_gripper_xpos[:2]) < 0.05 or np.linalg.norm(goal[:2] - object_xpos) < 0.1 or \
                np.linalg.norm(obstacle_xpos - goal[:2]) < 0.15 or vector_from_obstacle_to_object_x*(goal[0] - obstacle_xpos_x) > 0 or vector_from_obstacle_to_object_y*(goal[1] - obstacle_xpos_y)>0:
            # while vector_from_obstacle_to_object_x*(goal[0] - obstacle_xpos_x) > 0 or vector_from_obstacle_to_object_y*(goal[1] - obstacle_xpos_y)>0:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()
    
    # def _sample_subgoal(self):
    #     choice = np.random.randint(2)
    #     subgoal = np.array(self.sub_target[int(choice)])
    #     subgoal[2] = self.height_offset
    #     if self.target_in_the_air and self.np_random.uniform() < 0.5:
    #         subgoal[2] += self.np_random.uniform(0, 0.45)
    #     return subgoal.copy()
    
    def compute_reward(self, achieved_goal, goal, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        # d1 = goal_distance(achieved_goal, goal[1])
        if self.reward_type == 'sparse':
            return -(d > self.distance_threshold).astype(np.float32) # - (d1 > self.distance_threshold).astype(np.float32)
        else:
            return -d
        
    def compute_cost(self, obs, k, info):
        # Compute distance between goal and the achieved goal.
        c = ((obs[3] < obs[25] + 0.06 + self.size_obstacle[0] / 2 + self.size_object[0] / 2 + k) and (obs[3] > obs[25] + 0.06 - self.size_obstacle[0] / 2 - self.size_object[0] / 2 - k)\
             and (obs[4] < obs[26] + self.size_obstacle[1] / 2 + self.size_object[1] / 2 + k) and (obs[4] > obs[26] - self.size_obstacle[1] / 2 - self.size_object[1] / 2 - k))
        # d1 = goal_distance(achieved_goal, goal[1])
        if c:
            cost = 1
        else:
            cost = 0
        return np.array(cost)
        
    # def reset(self):
    #     # Attempt to reset the simulator. Since we randomize initial conditions, it
    #     # is possible to get into a state with numerical issues (e.g. due to penetration or
    #     # Gimbel lock) or we may not achieve an initial condition (e.g. an object is within the hand).
    #     # In this case, we just keep randomizing until we eventually achieve a valid initial
    #     # configuration.
    #     super(FetchPushObstacleEnv, self).reset()
    #     did_reset_sim = False
    #     while not did_reset_sim:
    #         did_reset_sim = self._reset_sim()
    #     self.goal = self._sample_goal().copy()
    #     obs = self._get_obs()
    #     return obs
    
    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self._set_action(action)
        self.sim.step()
        self._step_callback()
        obs = self._get_obs()

        done = False
        info = {
            'is_success': self._is_success(obs['achieved_goal'], self.goal),
        }
        reward = self.compute_reward(obs['achieved_goal'], self.goal, info)
        cost = self.compute_cost(obs['observation'], self.cost_threshold, info)
        return obs, reward, cost, done, info
    
    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # rotations
            object_rot = rotations.mat2euler(self.sim.data.get_site_xmat('object0'))
            # velocities
            object_velp = self.sim.data.get_site_xvelp('object0') * dt
            object_velr = self.sim.data.get_site_xvelr('object0') * dt
            # gripper state
            object_rel_pos = object_pos - grip_pos
            object_velp -= grip_velp
        else:
            object_pos = object_rot = object_velp = object_velr = object_rel_pos = np.zeros(0)
        gripper_state = robot_qpos[-2:]
        gripper_vel = robot_qvel[-2:] * dt  # change to a scalar if the gripper is made symmetric

        if not self.has_object:
            achieved_goal = grip_pos.copy()
        else:
            achieved_goal = np.squeeze(object_pos.copy())
        obs = np.concatenate([
            grip_pos, object_pos.ravel(), object_rel_pos.ravel(), gripper_state, object_rot.ravel(),
            object_velp.ravel(), object_velr.ravel(), grip_velp, gripper_vel,
        ])

        obstacle_pos = self.sim.data.get_site_xpos('obstacle0')
        # print(1, obstacle_pos)
        obstacle_rot = rotations.mat2euler(self.sim.data.get_site_xmat('obstacle0'))
        # gripper state
        obstacle_grip_rel_pos = obstacle_pos - grip_pos
        # object state
        obstacle_obj_rel_pos = obstacle_pos - object_pos
        # obs = np.concatenate([obs, obstacle_pos.ravel(
        # ), obstacle_grip_rel_pos.ravel(), obstacle_obj_rel_pos.ravel()])
        obs = np.concatenate([obs, obstacle_pos.ravel(), obstacle_rot.ravel(), obstacle_grip_rel_pos.ravel(), obstacle_obj_rel_pos.ravel()])


        return {
            'observation': obs.copy(),
            'achieved_goal': achieved_goal.copy(),
            'desired_goal': self.goal.copy(),
        }
    
    # def _render_callback(self):
    #     # Visualize target.
    #     sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()
    #     site_id = self.sim.model.site_name2id('target0')
    #     self.sim.model.site_pos[site_id] = self.goal - sites_offset[0]
    #     # site_id1 = self.sim.model.site_name2id('subtarget0')
    #     # self.sim.model.site_pos[site_id1] = self.subgoal - sites_offset[0]
    #     self.sim.forward()


    def goal2observation(self, goal):
        '''
        generate an observation that starts from the goal.
        '''
        obs = self._get_obs()
        assert isinstance(obs, dict)
        # object_pos
        obs['observation'][3:6] = goal
        # object_rel_pos
        obs['observation'][6:9] = obs['observation'][3:6] - obs['observation'][0:3]
        return obs.copy()
    
    
    

# env = FetchPushObstacleEnv()
# # env = gym.make('FetchPush-v1')
# # print(env.initial_gripper_xpos, env.initial_state)
# for _ in range(1):
#     obs = env.reset()
# # print(env.initial_gripper_xpos)
# print(env.goal)
# print(env.obstacle_qpos)
# for t in range(1000):
#     action = env.action_space.sample()
#     obs, reward, cost, done, info = env.step(action)
#     env.render()
#     if done:
#         print("Episode finished after {} timesteps".format(t+1))
#         break
# env.close()