[[0.99930021 0.99940015 0.9995001  0.99960006 0.99970003 0.99980001 0.9999     1.         0.        ]
 [0.99920028 0.99930021 0.99940015 0.9995001  0.99960006 0.99970003 0.99980001 0.9999     1.        ]
 [0.99910036 0.99920028 0.99930021 0.99940015 0.9995001  0.99960006 0.99970003 0.99980001 0.9999    ]
 [0.99900045 0.         0.         0.         0.         0.         0.         0.         0.99980001]
 [0.99890594 0.99900319 0.99910085 0.99920028 0.99930021 0.99940015 0.9995001  0.99960006 0.99970003]
 [0.9989781  0.99906539 0.99915747 0.9992565  0.99920028 0.99930021 0.99940015 0.9995001  0.99960006]]

"""
Value at start state (5,3) should not be higher than the ones on the right!
Values from the goal down are supposed to be :"""
[   1.0,
    0.9999,
    0.9998000100000001,
    0.999700029999,
    0.9996000599960001,
    0.9995000999900006,
    0.9994001499800016,
    0.9993002099650036,
    0.9992002799440071,
    0.9991003599160126]
"""
Which is verified until start state where 0.9992565 != 0.99910035 == gamma**9
It's even higher than gamma**8 so it seems to have a 2-step ahead advantage or something
It could be a noisy approximation of gamma**7 == 0.9993002099650036
"""
"""
So the probable reason is that the option implementation was wrong, not discounting the bootstrapping
along with the rewards, i.e. I was doing r1 + g*r2 + g^2*r3 + g*maxQ instead of ..+ g^3*maxQ.
So the option worked as a teleporter in values, and it got good backups when it lead close to the goal
But not to the goal itself (where the backup is true) which is why it worked well for whole episode explore.
Also explains why it didn't work even without the option in test: the 2nd best thing to do is to go down, sacrifice
*gamma reward, but end back up on the same state where you can retry to explore.
After fix:
"""


[[0.93192289 0.94147569 0.95099002 0.96059601 0.970299   0.9801     0.99       1.         0.        ]
 [0.92274469 0.93206535 0.94148015 0.95099005 0.96059601 0.970299   0.9801     0.99       1.        ]
 [0.91351725 0.92274465 0.93206534 0.94148015 0.95099005 0.96059601 0.970299   0.9801     0.99      ]
 [0.90438208 0.         0.         0.         0.         0.         0.         0.         0.9801    ]
 [0.89533825 0.90398849 0.91351638 0.92274469 0.93206535 0.94148015 0.95099005 0.96059601 0.970299  ]
 [0.88638487 0.89428802 0.90437881 0.91351725 0.92274469 0.93206535 0.94148015 0.95099005 0.96059601]]

"Compared to the true gamma=0.99 exponent values:"
[   1.0,
    0.99,
    0.9801,
    0.970299,
    0.96059601,
    0.9509900498999999,
    0.941480149401,
    0.9320653479069899,
    0.9227446944279201,
    0.9135172474836408]
"So the problem is fixed! Redoing all runs."
