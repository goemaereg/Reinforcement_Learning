"""Code for starting a training session.

Usage:
>>> mpirun -np 8 python main.py
"""

# The six following lines aims to ignore the numerous warnings of tensorflow. They can be removed.
import panda_gym
import baselines.her.her as her
from baselines import logger
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import make_vec_env
import numpy as np
import gym
import tensorflow as tf
import warnings
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.simplefilter(action='ignore')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


try:
    from mpi4py import MPI
except ImportError:
    MPI = None


def train(seed, log_path, save_path, env_id, replay_strategy, total_timesteps, num_env):
    rank = 0 if MPI is None else MPI.COMM_WORLD.Get_rank()

    # configure log
    if rank == 0:
        logger.configure(log_path)
    else:
        logger.configure(log_path, format_strs=[])

    # config tf
    config = tf.ConfigProto(allow_soft_placement=True,
                            intra_op_parallelism_threads=1,
                            inter_op_parallelism_threads=1)
    config.gpu_options.allow_growth = True
    get_session(config=config)

    # vectorize the environnement
    env = make_vec_env(env_id, 'robotics', num_env, seed,
                       flatten_dict_observations=False)

    # learn
    model = her.learn(
        env=env,
        seed=seed,
        total_timesteps=total_timesteps,
        network='mlp', replay_strategy=replay_strategy
    )

    # save the model
    if save_path is not None and rank == 0:
        save_path = os.path.expanduser(save_path)
        model.save(save_path)


if __name__ == '__main__':
    seed = 0
    log_path = '~/logs/PandaPickAndPlace_{}/'.format(seed)
    save_path = 'policy_PandaPickAndPlace'
    env_id = 'PandaPickAndPlace-v0'
    replay_strategy = 'future'
    total_timesteps = 200000
    num_env = 1
    train(seed, log_path, save_path, env_id,
          replay_strategy, total_timesteps, num_env)
