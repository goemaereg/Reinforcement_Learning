"""Code for starting a training session.

Usage:
>>> python play.py
"""

# The six following lines aims to ignore the numerous warnings of tensorflow. They can be removed.
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import warnings
warnings.simplefilter(action='ignore')
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import gym
import numpy as np

import baselines.her.her as her
from baselines import logger
from baselines.common.cmd_util import make_env

from baselines.her.experiment.config import DEFAULT_PARAMS, DEFAULT_ENV_PARAMS, prepare_params, configure_dims, configure_ddpg
from baselines.common import tf_util

import panda_gym

def load_policy(network, env_name, load_path):
    # Prepare params.
    params = DEFAULT_PARAMS
    # env_name = env.spec.id
    params['env_name'] = env_name
    if env_name in DEFAULT_ENV_PARAMS:
        params.update(DEFAULT_ENV_PARAMS[env_name])  # merge env-specific parameters in

    params = prepare_params(params)

    dims = configure_dims(params)
    policy = configure_ddpg(dims=dims, params=params)

    tf_util.load_variables(load_path)

    return policy


def play(load_path, env_name, env_kwargs):
    # load the environment
    logger.log("Loading the environment")
    env = make_env(env_name, env_type='robotics', flatten_dict_observations=False, env_kwargs=env_kwargs)

    # load the model
    logger.log("Loading the model")
    model = load_policy(env_name=env_name, network='mlp', load_path=load_path)

    # Running the model
    logger.log("Running the loaded model")
    while True:
        obs = env.reset()
        episode_rew = 0
        done = False
        while not done:
            action, _, _, _ = model.step(obs)
            obs, rew, done, _ = env.step(action)
            episode_rew += rew
            env.render()
            if done:
                print('episode_rew={}'.format(episode_rew))
                episode_rew = 0
    env.close()

if __name__=='__main__':
    load_path = 'policy_PandaPickAndPlace'
    env_name='PandaPickAndPlace-v0'
    play(load_path, env_name, env_kwargs={'render':True})