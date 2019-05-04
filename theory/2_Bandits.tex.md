# Multi-armed bandits

## Definition
A $k$-armed bandit problem is one where the agent gets to choose one of $k$ actions, all resulting in some reward; and the goal is to maximize the *expected total reward*.

Each action has a *mean* or *expected* reward, denoted $q_*$, i.e. $ q_* \left( a \right) = \mathbb{E} \left\{ R_t | At = a \right\} $.

Knowing these values $\forall a$ would solve the problem, as we could just maximize over them. Therefore, we can try to approximate $q_*$ by an estimator $Q_t$.

## Exploration Exploitation (XX) dilemma
With some estimates $Q_t\left(a\right)\,\, \forall a$, we have the possibility to *exploit* this current knowledge by acting *greedily* w.r.t $Q_t$ by simply taking the most interesting (maximum valued) action. This is, at a given time-step, the optimal way of accumulating the most reward for the next step.

We can, instead, *explore* different actions to gather knowledge about their values. This would enable us to perform a better action later on, so this is a more far-sighted strategy that only makes sense if we have many steps available.

As we cannot perform both at once, we need an XX strategy.

## Action-value
An *action-value function*, in this context, estimates the value of each action, like $Q_t$ recently introduced.

The most natural way of estimating is by simply averaging over the rewards obtained for each action. This *sample-average* method converges to $q_*$ according to the law of large numbers - that is, provided infinite visits of each action.

This can be ensured by an $\epsilon$-greedy strategy, acting greedily most of the time, but randomly under a small probability $\epsilon$. Obviously, $\epsilon$-greedy and sample-average comes with no warranty of sample-efficiency.
