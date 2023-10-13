import gym
from gym.envs.registration import register

def register_envs():
    
    register(
        id='FetchPushObstacle-v0',
        entry_point='envs.push_obstacle:FetchPushObstacleEnv',
        max_episode_steps=50,)
        
    register(id='FetchPickAndPlaceObstacle-v0',
            entry_point='envs.pick_and_place_obstacle:FetchPickAndPlaceObstacleEnv',
            max_episode_steps=50,)

    register(id='FetchSlideObstacle-v0',
        entry_point='envs.slide_obstacle:FetchSlideObstacleEnv',
        max_episode_steps=50,)
    
    register(id='FetchReachObstacle-v0',
        entry_point='envs.reach_obstacle:FetchReachObstacleEnv',
        max_episode_steps=50,)