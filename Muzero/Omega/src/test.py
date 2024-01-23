import numpy as np

result=[0, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1]

print(result)
zero_score = np.sum(x==0 for x in result)
one_score = np.sum(x==1 for x in result)

print(zero_score, one_score)