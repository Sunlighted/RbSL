import os
from gym import utils
from gym.envs.robotics import rotations, robot_env, fetch_env
import gym.envs.robotics.utils as robot_utils
import numpy as np


# Ensure we get the path separator correct on windows
MODEL_XML_PATH = os.path.join(os.path.dirname(__file__), 'assets', 'fetch', 'pick_and_place_obstacle.xml')

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.linalg.norm(goal_a - goal_b, axis=-1)


class FetchPickAndPlaceObstacleEnv(fetch_env.FetchEnv, utils.EzPickle):
    def __init__(self, reward_type='sparse'):
        initial_qpos = {
            'robot0:slide0': 0.405,
            'robot0:slide1': 0.48,
            'robot0:slide2': 0.0,
            'object0:joint': [1.25, 0.53, 0.4, 1., 0., 0., 0.],
        }
        fetch_env.FetchEnv.__init__(
            self, MODEL_XML_PATH, has_object=True, block_gripper=False, n_substeps=20,
            gripper_extra_height=0.2, target_in_the_air=True, target_offset=0.0,
            obj_range=0.15, target_range=0.23, distance_threshold=0.05,
            initial_qpos=initial_qpos, reward_type=reward_type)
        utils.EzPickle.__init__(self)

        self.pos_obstacle = self.sim.model.geom_pos[self.sim.model.geom_name2id('obstacle0')]
        self.size_object = self.sim.model.geom_size[self.sim.model.geom_name2id('object0')]
        self.size_obstacle = self.sim.model.geom_size[self.sim.model.geom_name2id('obstacle0')]
        self.cost_threshold = 0.07

    def _reset_sim(self):
        self.sim.set_state(self.initial_state)
        self.initial_gripper_xpos[:3] = [1.34193226,0.74910037,0.555]

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
        while np.linalg.norm(obstacle_xpos - self.initial_gripper_xpos[:2]) < 0.1 or (self.has_object and np.linalg.norm(object_xpos - obstacle_xpos) < 0.1):
            obstacle_xpos = self.initial_gripper_xpos[:2] + self.np_random.uniform(-self.obj_range,self.obj_range, size=2)
        obstacle_qpos = self.sim.data.get_site_xpos('obstacle0')
        # assert obstacle_qpos.shape == (7,)
        # obstacle_pos = obstacle_qpos
        obstacle_qpos[:2] = obstacle_xpos
        obstacle_qpos[2] = self.height_offset + self.np_random.uniform(0.03, 0.2)
        # obstacle_qpos = self.sim.data.get_joint_qpos('obstacle0:joint')
        # print(obstacle_qpos)
        # self.sim.data.set_joint_qpos('object0:joint', object_qpos)
        # assert 0
        # print(self.sim.model.site_pos)
        
        # site_idx = self.sim.model.site_name2id('obstacle0')
        # print(site_idx)
        # obstacle_spos = obstacle_qpos.copy()
        # obstacle_spos[0] = obstacle_qpos[0].copy() - 1.3
        # obstacle_spos[1] = obstacle_qpos[1].copy() - 0.75
        # obstacle_spos[2] = obstacle_qpos[2].copy() - 0.42469975

        # self.sim.model.site_pos[self.sim.model.site_name2id('obstacle0')] = obstacle_spos
        self.sim.model.body_pos[self.sim.model.body_name2id('obstacle0')] = obstacle_qpos
        # print(self.sim.model.body_pos)
        # assert 0

        self.sim.forward()
        return True
    
    def compute_cost(self, achieved_goal, goal, obs, k, info):
        # Compute distance between goal and the achieved goal.
        d = goal_distance(achieved_goal, goal)
        c = ((obs[3] < obs[25] + (self.size_obstacle[0]) / 2 + self.size_object[0] / 2 + k) and (obs[3] > obs[25] - (self.size_obstacle[0]) / 2 - self.size_object[0] / 2 - k)\
             and (obs[4] < obs[26] + self.size_obstacle[1] / 2 + self.size_object[1] / 2 + k) and (obs[4] > obs[26] - self.size_obstacle[1] / 2 - self.size_object[1] / 2 - k)\
             and (obs[5] < obs[27] + 0.03 + self.size_obstacle[2] / 2 + self.size_object[2] / 2 + k) and (obs[5] > obs[27] + 0.03 - self.size_obstacle[2] / 2 - self.size_object[2] / 2 - k)) and ((d > self.distance_threshold).astype(np.float32) == 1)
        if c:
            cost = 1
        else:
            cost = 0
        return np.array(cost)
    
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
        cost = self.compute_cost(obs['achieved_goal'], self.goal, obs['observation'], self.cost_threshold, info)
        return obs, reward, cost, done, info
    
    def _sample_goal(self):
        if self.has_object:
            object_xpos = self.sim.data.get_joint_qpos('object0:joint')[:2]
            obstacle_xpos = self.sim.data.get_body_xpos('obstacle0')[:2]
            # print(1,obstacle_xpos)
            obstacle_xpos_x = obstacle_xpos[0]
            obstacle_xpos_y = obstacle_xpos[1]
            obstacle_xpos_z = self.sim.data.get_body_xpos('obstacle0')[2]
            # print(1,obstacle_xpos_z)
            object_xpos_x = object_xpos[0]
            object_xpos_y = object_xpos[1]
            object_xpos_z = self.sim.data.get_joint_qpos('object0:joint')[2]
            # print(1,object_xpos_z)
            # assert 0
            vector_from_obstacle_to_object_x = (object_xpos_x - obstacle_xpos_x)
            vector_from_obstacle_to_object_y = (object_xpos_y - obstacle_xpos_y)
            vector_from_obstacle_to_object_z = (object_xpos_z - obstacle_xpos_z)

            goal = self.initial_gripper_xpos[:3]
            while np.linalg.norm(goal[:2] - self.initial_gripper_xpos[:2]) < 0.05 or np.linalg.norm(goal[:2] - object_xpos) < 0.05 or \
                np.linalg.norm(obstacle_xpos - goal[:2]) < 0.1 or vector_from_obstacle_to_object_x*(goal[0] - obstacle_xpos_x) > 0 or vector_from_obstacle_to_object_y*(goal[1] - obstacle_xpos_y)>0:
            # while vector_from_obstacle_to_object_x*(goal[0] - obstacle_xpos_x) > 0 or vector_from_obstacle_to_object_y*(goal[1] - obstacle_xpos_y)>0:
                goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
            goal += self.target_offset
            goal[2] = self.height_offset
            if self.target_in_the_air and self.np_random.uniform() < 0.5:
                #while vector_from_obstacle_to_object_z*(goal[2] - obstacle_xpos_z) > 0 :
                goal[2] += self.np_random.uniform(0, 0.45)
        else:
            goal = self.initial_gripper_xpos[:3] + self.np_random.uniform(-self.target_range, self.target_range, size=3)
        return goal.copy()
    
    def _get_obs(self):
        # positions
        grip_pos = self.sim.data.get_site_xpos('robot0:grip')
        dt = self.sim.nsubsteps * self.sim.model.opt.timestep
        grip_velp = self.sim.data.get_site_xvelp('robot0:grip') * dt
        robot_qpos, robot_qvel = robot_utils.robot_get_obs(self.sim)
        if self.has_object:
            object_pos = self.sim.data.get_site_xpos('object0')
            # print(2, object_pos)
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

        obstacle_pos = self.sim.data.get_body_xpos('obstacle0')
        obstacle_pos[2] = obstacle_pos[2] - 0.03
        # print(1, obstacle_pos)
        obstacle_rot = rotations.mat2euler(self.sim.data.get_body_xmat('obstacle0'))
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
    
# env = FetchPickAndPlaceObstacleEnv()
# # env = gym.make('FetchPush-v1')
# # print(env.initial_gripper_xpos, env.initial_state)
# for _ in range(1):
#     obs = env.reset()
# # print(env.initial_gripper_xpos)
# # print(env.goal)
# # print(env.object_qpos)
# for t in range(1000):
#     action = env.action_space.sample()
#     obs, reward, cost, done, info = env.step(action)
#     env.render()
#     if done:
#         print("Episode finished after {} timesteps".format(t+1))
#         break
# env.close()

