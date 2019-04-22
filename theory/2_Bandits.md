# Multi-armed bandits

## Definition
A <img src="/theory/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/>-armed bandit problem is one where the agent gets to choose one of <img src="/theory/tex/63bb9849783d01d91403bc9a5fea12a2.svg?invert_in_darkmode&sanitize=true" align=middle width=9.075367949999992pt height=22.831056599999986pt/> actions, all resulting in some reward; and the goal is to maximize the *expected total reward*.

Each action has a *mean* or *expected* reward, denoted <img src="/theory/tex/11cf084e7fd83c09088280b2d91d5497.svg?invert_in_darkmode&sanitize=true" align=middle width=14.07350669999999pt height=14.15524440000002pt/>, i.e. <img src="/theory/tex/77cc33cd82f876e437a4ed1231728cad.svg?invert_in_darkmode&sanitize=true" align=middle width=162.87133004999998pt height=24.65753399999998pt/>.

Knowing these values <img src="/theory/tex/77ad9f02dbf52502a817cb81fb049df7.svg?invert_in_darkmode&sanitize=true" align=middle width=17.82160214999999pt height=22.831056599999986pt/> would solve the problem, as we could just maximize over them. Therefore, we can try to approximate <img src="/theory/tex/11cf084e7fd83c09088280b2d91d5497.svg?invert_in_darkmode&sanitize=true" align=middle width=14.07350669999999pt height=14.15524440000002pt/> by an estimator <img src="/theory/tex/025b11cd28d6c936d3062a554bbaf0b5.svg?invert_in_darkmode&sanitize=true" align=middle width=17.96121689999999pt height=22.465723500000017pt/>.

## Exploration Exploitation (XX) dilemma
With some estimates <img src="/theory/tex/9ee5fc9fca722cbd0f762e76db8201c7.svg?invert_in_darkmode&sanitize=true" align=middle width=69.03791069999998pt height=24.65753399999998pt/>, we have the possibility to *exploit* this current knowledge by acting *greedily* w.r.t <img src="/theory/tex/025b11cd28d6c936d3062a554bbaf0b5.svg?invert_in_darkmode&sanitize=true" align=middle width=17.96121689999999pt height=22.465723500000017pt/> by simply taking the most interesting (maximum valued) action. This is, at a given time-step, the optimal way of accumulating the most reward for the next step.

We can, instead, *explore* different actions to gather knowledge about their values. This would enable us to perform a better action later on, so this is a more far-sighted strategy that only makes sense if we have many steps available.

As we cannot perform both at once, we need a XX strategy.

## Action-value
An *action-value function*, in this context, estimates the value of each action, like <img src="/theory/tex/025b11cd28d6c936d3062a554bbaf0b5.svg?invert_in_darkmode&sanitize=true" align=middle width=17.96121689999999pt height=22.465723500000017pt/> recently introduced.

The most natural way of estimating is by simply averaging over the rewards obtained for each action. This *sample-average* method converges to <img src="/theory/tex/11cf084e7fd83c09088280b2d91d5497.svg?invert_in_darkmode&sanitize=true" align=middle width=14.07350669999999pt height=14.15524440000002pt/> according to the law of large numbers - that is, provided infinite visits of each action.

This can be ensured by an <img src="/theory/tex/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode&sanitize=true" align=middle width=6.672392099999992pt height=14.15524440000002pt/>-greedy strategy, acting greedily most of the time, but randomly under a small probability <img src="/theory/tex/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode&sanitize=true" align=middle width=6.672392099999992pt height=14.15524440000002pt/>. Obviously, <img src="/theory/tex/7ccca27b5ccc533a2dd72dc6fa28ed84.svg?invert_in_darkmode&sanitize=true" align=middle width=6.672392099999992pt height=14.15524440000002pt/>-greedy and sample-average comes with no warranty of sample-efficiency.
