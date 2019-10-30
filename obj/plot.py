
import numpy as np
import matplotlib.pyplot as plt


x = np.loadtxt("run.out")

plt.plot(x[:,0],x[:,1])
plt.plot(x[:,0],x[:,2])

#plt.savefig('A.png')
plt.show()
plt.close()

