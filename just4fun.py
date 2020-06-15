# just for fun
# given a random generator - can you compute pi?

import random
import numpy as np
from numpy import random
from tqdm import trange

# Compute pi using only RNG .
# This solution is in N^2 time
# (0.5-x)^2 + (0.5-y)^2

N = int(5e3)
count = 0

for _ in trange(N):
    for _ in range(N):
        x , y = np.random.uniform(0,1,2)
        dist = np.sqrt((x-.5)**2 + (y-.5)**2)
        if (dist <= .5):
            count += 1

# 2*pi*r^2 = A
A = count/(N**2) # Area
pi_est = 4*A

print("Estimate of pi is %f" % pi_est)