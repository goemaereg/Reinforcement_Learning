import numpy as np
from utils import save_plot
import time
import gym
import gym_additions
from agents_core import ValueIterationV2
np.set_printoptions(threshold=500, linewidth=500, edgeitems=250)

def test(agent, env, n_episodes):
    print("Now testing:")
    reward_record = np.empty(n_episodes)
    for episode in range(n_episodes):
        # if (episode%(n_episodes//5)==0): print("\tEpisode {}/{}".format(episode, n_episodes))
        state = env.reset()
        cumulate_reward = 0
        while True:
            env.render()
            action = agent.act(state)
            print("Actions is {} in state {}".format(["Up", "Right", "Left"][action], state))
            state, reward, done, _ = env.step(action)
            print("Ended up in state {} with r {}".format(state, reward))
            cumulate_reward += reward
            if done:
                env.render()
                break

        reward_record[episode] = cumulate_reward
    return reward_record.mean()

def perf_plot(env, agent):
    v_record = []
    v_record.append(agent.V.copy())
    theta = 1e-100
    #print("Test performance initial {}".format(test(agent, env, 500)))
    n_sweeps = 1000 # number of sweeps (loops) over the full state space
    for sweep in range(n_sweeps):
        delta = 0
        if (sweep%(n_sweeps//5)==0): print("Sweep {}/{}".format(sweep, n_sweeps))
        for s in env.non_terminal_states():
            v = agent.V[s]
            agent.learn(s)
            delta += abs(v-agent.V[s])

        if delta < theta:
            pass
            print("Exiting through delta={} < {} after {} sweeps".format(delta, theta, sweep))
            break

        v_record.append(agent.V.copy())

    #    print("Test performance at sweep {}: {}".format(sweep, test(agent, env, 500)))
    v_record = np.array(v_record)
    save_plot(v_record, 'VI_value_coinflip_{}'.format(env.p_h), "Value Iteration final V at CoinFlip p_h={}".format(env.p_h),
              xlabel='State', ylabel='Value estimate', labels=[str(i) for i in  range(len(v_record))])

    policy = np.array([agent.act(s) for s in env.non_terminal_states()])
    save_plot(policy, 'VI_policy_coinflip_{}'.format(env.p_h), "Value Iteration final Policy at CoinFlip p_h={}".format(env.p_h),
              xlabel='State', ylabel='Action (stake)')

def value_function(env, agent):
    """ Simply converges and outputs the value function of this environment """
    theta = 1e-100
    n_sweeps = 1000 # number of sweeps (loops) over the full state space
    for sweep in range(n_sweeps):
        delta = 0
        if (sweep%(n_sweeps//5)==0): print("Sweep {}/{}".format(sweep, n_sweeps))
        for s in env.non_terminal_states():
            v = agent.V[s]
            agent.learn(s)
            delta += abs(v-agent.V[s])

        if delta < theta:
            pass
            print("Exiting through delta={} < {} after {} sweeps".format(delta, theta, sweep))
            break
    return agent.V.copy()


if __name__ == '__main__':
    env = gym.make("Tractor-v0")
    agent = ValueIterationV2(env)
    learn = False

    if learn:
        value_function(env, agent)
        np.save("tractor_optimal_v", agent.V)

    V = np.load("tractor_optimal_v.npy")
    agent.V = V.copy()
    print(test(agent, env, 1))
