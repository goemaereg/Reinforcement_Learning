"""Code for starting a training session.

Usage:
$ mpirun -np 8 python train.py PandaPickAndPlace-v1 0 500000
"""

import os

import baselines.her.her as her
import baselines.ddpg.ddpg as ddpg
from baselines import logger
from baselines.common.tf_util import get_session
from baselines.common.cmd_util import make_vec_env
import tensorflow as tf
from mpi4py import MPI

import panda_gym


def train(seed, log_path, save_path, env_id, replay_strategy, timesteps, num_env):
    rank = 0 if MPI is None else MPI.COMM_WORLD.Get_rank()

    # configure log
    if rank == 0:
        logger.configure(log_path)
    else:
        logger.configure(log_path, format_strs=[])

    # config tf
    config = tf.ConfigProto(
        allow_soft_placement=True,
        intra_op_parallelism_threads=1,
        inter_op_parallelism_threads=1,
    )
    config.gpu_options.allow_growth = True
    get_session(config=config)

    ddpg_and_her = False
    #ddpg_and_her = True

    # vectorize the environnement
    flatten_dict_observations = not ddpg_and_her
    env = make_vec_env(
        env_id, "robotics", num_env, seed, flatten_dict_observations=flatten_dict_observations
    )

    # learn
    if ddpg_and_her:
        model = her.learn(
            env=env,
            seed=seed,
            total_timesteps=timesteps,
            network="mlp",
            replay_strategy=replay_strategy,
        #    override_params={'n_cycles':10},
        )
    else:
        model = ddpg.learn(
            env=env,
            seed=seed,
            total_timesteps=timesteps,
            network="mlp"
        #    override_params={'n_cycles':10},
        )

    # save the model
    if save_path is not None and rank == 0:
        save_path = os.path.expanduser(save_path)
        model.save(save_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("env_id")
    parser.add_argument("seed")
    parser.add_argument("timesteps")
    args = parser.parse_args()
    env_id = str(args.env_id)
    seed = int(args.seed)
    log_path = "results/{}/{}/".format(env_id, seed)
    save_path = "results/{}/policy_{}".format(env_id, env_id)
    replay_strategy = "future"
    # n_epoch * 50 cycles * 50 timestep per episode (à multiplier par le nombre de workers)
    timesteps = int(args.timesteps) // 8  # 8 MPI workers
    num_env = 1
    train(seed, log_path, save_path, env_id, replay_strategy, timesteps, num_env)
