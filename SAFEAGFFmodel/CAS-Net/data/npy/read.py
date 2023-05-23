import numpy as np

file = np.load('./img/1.npy')
print(file)
np.savetxt('./img/1.txt',file)