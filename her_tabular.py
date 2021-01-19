import gym
import gym_additions
import json
from tabular.agents import *
from tabular.hierarchical import *
from utils import save_plot

#env_name = 'FourRoomsGoalBig-v0'
env_name = 'FourRoomsGoal-v0'
env = gym.make(env_name)
shapes = (tuple([s.n for s in env.observation_space]), env.action_space.n)
d = {
    'env_shapes': shapes,
    'explo_horizon': 1,
    'learn_rate': 0.05,
    'explo_steps': 10,
    'gamma': 0.9,
    'lambda': 0.9,
    'n': 10
}

n_episodes = 3000
n_steps = 150000 # virually never
#n_steps = 120 # episode horizon
# subtrajectory length
n_subtraject_steps = 8

# plot scales
n_plot_xscale = (0, 50000)
n_plot_yscale = (0, 150)

# replay buffer size
n_replaybuffer_size = 1*1024
# number of optimization cycles within an episode
n_minibatch_cycles = 2 
# sample batch size for each optimization cycle
n_batchsize = 8 


from collections import deque


class ReplayBuffer(object):
    def __init__(self, capacity):
        """
        Parameters
        ----------
        capacity : int
            the length of the buffer
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Parameters
        ----------
        batch_size : int
           the number of samples 
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)


def env_state(obj):
    return (obj[0], obj[1])


def env_goal(obj):
    return (obj[2], obj[3])


def test_agent(agent, env, n_episodes, n_steps):
    """ Returns the steps_history of the agent"""
    replay_buffer = ReplayBuffer(n_replaybuffer_size)
    # Training phase
    steps_history = np.empty(n_episodes)
    xaxis = np.empty(n_episodes)
    for ep in range(n_episodes):
        if (ep%(n_episodes//5)==0):
            print("Episode {}/{}".format(ep+1, n_episodes))
        # sample initial state and goal
        obs = env.reset()
        transition_history = []
        T=16
        # log an episode
        for step in range(n_steps):
            # sample action a using behavioral policy from agent
            action = agent.act(obs)
            # execute action and observe new state
            new_obs, reward, done, _ = env.step(action)
            # use a binary and sparse reward from the environment: int(done)
            # store transition in replay buffer
            transition = (obs, action, reward, new_obs, done)
            replay_buffer.push(*transition)
            transition_history.append(transition)
            # update obs
            obs = new_obs
            if done:
                break
                if np.random.rand()<0.01: print("Step {}".format(step))

            steps_history[ep] = step

        index = 0
        for step in range(len(transition_history)):
            # sample a set of additional goals for replay in this sub trajectory: # S(so, s1, ..., sT) = m(sT) = sT
            goal_index = (((step // n_subtraject_steps) + 1) * n_subtraject_steps) - 1
            goal_index = min(goal_index, len(transition_history) - 1)
            _, _, _, goal_obs, _ = transition_history[goal_index]
            sampled_goal = env_state(goal_obs)
            # current transition step
            obs, action, _, new_obs, _ = transition_history[step]
            state = env_state(obs)
            new_state = env_state(new_obs)
            # use a binary and sparse reward if new goal reached
            done = (state == sampled_goal)
            #reward = - int(done == False)
            reward = int(done)
            # store transition in replay buffer
            transition = ((*state, *sampled_goal), action, reward, (*new_state, *sampled_goal), done)
            replay_buffer.push(*transition)

        # learn on N cycles of minibatches of size B
        if ep > 0:
            xaxis[ep] = xaxis[ep - 1]
        else:
            xaxis[ep] = 0
        for i in range(n_minibatch_cycles):
            # Sample a minibatch B from the replay buffer
            n_samples = min(n_batchsize, len(replay_buffer))
            transitions = replay_buffer.sample(n_samples)
            # Perform one step of learning using the agent and the minibatch
            for transition in transitions:
                agent.learn(*transition)

            xaxis[ep] += n_samples 

        if ep==0:
            print("First trial in {} steps".format(step))
    env.close()
    return xaxis[1:], steps_history[1:] # first is purely random


def smooth(perf, smooth_avg):
    perf_smooth = []
    perf_smooth += [np.mean(perf[i-smooth_avg:i+smooth_avg])
                    for i in range(smooth_avg, len(perf)-smooth_avg)]
    return perf_smooth

agents = [
    QLearning(**d)
    ]
if len(agents) == 1:
    agent = agents[0]
    xaxis, perf = test_agent(agent, env, n_episodes, n_steps)
    print("Final performance: {}".format(perf[-1]))
    # plotting
    launch_specs = 'perf{}'.format(env.roomsize)
    file_name = "tabular_her/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
    suptitle = "{} HER performance on {}{}".format(agent.name, env_name[:-3], env.roomsize)
    title = agent.tell_specs()
    xlabel = 'Optimisation steps'
    ylabel = "Performance at {}".format(env_name)
    save_plot(perf, file_name, suptitle, title, xlabel, ylabel,
              xaxis=xaxis, interval_xaxis=n_plot_xscale, interval_yaxis=n_plot_yscale,
              smooth_avg=n_episodes//100, only_avg=True)

else:
    perfs = []
    for agent in agents:
        np.random.seed(0)
        random.seed(0)
        _, perf = test_agent(agent, env, n_episodes, n_steps)
        # The smoothing is done here as there are multiple curves
        perfs.append(smooth(perf, n_episodes//100))
        print("Final performance: {}".format(perf[-1]))

    perfs = np.array(perfs)
    print(perfs.shape)
    # plotting
    launch_specs = 'comp'
    #file_name = "tabular/perf_plots/{}/{}/{}".format(env_name, agent.name, launch_specs)
    file_name = "tabular_her/perf_plots/{}/{}".format(env_name, launch_specs)
    suptitle = "Agent HER Performance Comparison on {}{}".format(env_name[:-3], env.roomsize)
    title = agent.tell_specs()
    xlabel = 'Episode'
    ylabel = "Performance at {}".format(env_name)
    save_plot(perfs, file_name, suptitle, title, xlabel, ylabel,
              smooth_avg=0, labels=[agent.name for agent in agents])
