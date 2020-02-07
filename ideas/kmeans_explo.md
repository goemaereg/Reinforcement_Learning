# K-means intrinsic reward scheme
This is a *memory-based*, *non-parametric* intrinsic reward scheme.

## Core concept/algorithm
The idea is to use memory of previous states (in all episodes' history) to assess the novelty of a new incoming state, in order to attribute an intrinsic reward.
The method is to store a *set M of K tuples* M={(m_i, n_i)} where *m_i is a mean (center) state* and *n_i is the visit count* i.e. the number of times the mean was used (will be clarified).
For example, it can be some random selection of the beginning play.

When a new state s is experienced:
* the reward is function of the distance to the closest mean:
  r = f(d), where d = min_i ||s - m_i||²
  or some other function of the distances to the center?
* if not CREATE_CONDITION:
    the closest mean is updated: m_i += m_i*n_i/(n_i + 1) + s/(n_i + 1)
    n_i += 1
* if CREATE_CONDITION:
    s becomes a new mean, K is incremented.

CREATE_CONDITION can be multiple things:
* creating a new mean every T steps
  * T could be extended over time (takes longer to create states)
  * Some additional conditions (see next) could be added.
* when a state is too far away from the rest:
  * if d>delta (hyperparameter)
  * if d>diameter(M) (i.e. the longest distance in the set for now.)

A proper online K-means algorithm probably takes all of this into account.

## Advantages:
* no need to understand the dynamics
* non-parametric -> low computational cost (especially if solution 1)
* control over memory size K
* probably easy to implement

## Problems:
* stochasticity
* one of the means might take all the states (e.g. rightmost, if env goes right).
* we need a technique to spread the means apart. Agent/algo to minimize the distance?
* random TV
* difference in input/features might not mean valuable difference for us (background)
* novel states might be "close" from our distance's pov (e.g. cross6 env: (1,1) is farthest from (1,0) but the natural distance doesn't show this)

Solutions:
* lower-dimensional representation, e.g. through RF. Could also train VAE or IDF but becomes parametric.
* See a true online K-means
* see first how K-means works, because we want to maximize coverage: if a center is always used, it might still be fine
* who solves that anyway? And that's to add to ExploreOption
* not too problematic, also IDF solves this (focuses on actions)

Ideas:
* divide reward by the number of visits of used center to "punish" recurring states.
* substract reward by a moving average of the reward to scale
