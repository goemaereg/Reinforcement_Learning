# Memory Based Function Approximation
## Core idea
So the basic concept is to store experiences - this can be value functions observed for states, or observed dynamics; but not *learn* anything in the sense of parameter adjusting.
When a query is asked (e.g.  a state value function), we can use the gathered knowledge at once to answer it: the simplest idea is (k-)nearest neighbor, with some definition of distance over states. From there we can easily extend all the way to something similar to RBFs, with states as centers, using a Gaussian kernel.
The basic idea is that we don't necessarily need a function approximator to instigate generalization. We can get by with a definition of distance or something else, and work with past experience.

## Applied along/for Unsupervised Behavior Learning?
This is an interesting observation because it can fall within the scope of Unsupervised Learning we talk about in `core.md`: classifying states just from their observation.
The natural problem that stems from this is that for now we've mainly taken as an example the state value estimation. However one of the core ideas of Unsupervised Behavior Learning was:
> to rid ourselves of environment goal; first developing (from curiosity or whatever) a useful set of policies, and *then* applying them to whatever task we're asked - the idea that if we just change the reward function (but not the whole MDP), much of our knowledge still applies.

So how can we apply memory-based function approximation there? We can learn to classify states and policies, or learn a measure distance, and use it to understand that this whole class of states or policies leads to this kind of results; hence we don't need to learn all over again - just generalize our behavior over this class.
