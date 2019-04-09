# Reinforcement_SuttonBarto
Implementations of the algorithms and methods learnt throughout the book "RL : an Introduction" by Sutton and Barto.

## Installing
The table-based algorithms implemented there use small Gridworld environments.
The render-supported versions I use are implemented on [my forked repository](https://github.com/Louis-Bagot/gym-gridworlds) of [Ondřej Podsztavek's work](https://github.com/podondra/gym-gridworlds).
The repository can be simply cloned and added to the python path in order to instantiate a gym-like environment as reported in the original repository :
```
$ import gym
$ import gym_gridworlds
$ env = gym.make('Gridworld-v0')  # substitute environment's name
```

This Reinforcement_SuttonBarto repository can then be simply cloned and used.
