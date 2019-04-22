#Multi-armed bandits

## Definition
A $k$-armed bandit problem is one where the agent gets to choose one of $k$ actions, all resulting in some reward; and the goal is to maximize the *expected total reward*.
Each action has a *mean* or *expected* reward, denoted $q_*$, i.e. $ q_* \left( a \right) = \mathbb{E} \left\{ R_t | At = a \right\} $
Knowing these values $\forall a$ would solve the problem, as we could just maximize over them. Therefore, we can try to approximate $q_*$ by an estimator $Q_t$.

## Exploration Exploitation dilemma
With some estimates $Q_t\left(a\right)\,\, \forall a$, we have the possibility to *exploit* this current knowledge by acting *greedily* w.r.t $Q_t$ by simply taking the most interesting (maximum valued) action.
We can, instead, *explore* different actions to gather knowledge about their values.
