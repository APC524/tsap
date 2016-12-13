import numpy as np
import matplotlib.pyplot as plt
import reduction

# read SP500 data
SP500 = np.genfromtxt('../data/SP500array.csv', delimiter=',')
SP500 = SP500.T
nStock = len(SP500[:,0])
nTime = len(SP500[0,:])

# preprocessing, subtract each stock price mean and normalize by std
x = np.copy(SP500)
for i in range(nStock):
    x[i,:] = (x[i,:] - np.mean(x[i,:]))/np.std(x[i,:])

# call and test PCA
n_components = 5
SPreduction = reduction.Reduction(x)
ux, at, energy_content = SPreduction.PCA(n_components)
# PCA compressed data
xPCA = ux.dot(at)

# plot principal components
plt.figure()
for i in range(n_components):
    plt.plot(range(1,nStock+1), ux[:,i],'-',linewidth=2)
plt.xlim([1,500])
plt.xlabel('SP500 Stock', fontsize=20)
plt.title('Principal components', fontsize=20)
plt.show()

# plot principal coefficient
plt.figure()
for i in range(n_components):
    plt.plot(range(1,nTime+1), at[i,:],'-',linewidth=2)
plt.xlim([1,500])
plt.xlabel('Time/days, starting at 1/2/15', fontsize=20)
plt.ylabel('PCA coefficient a(t)', fontsize=20)
plt.title('Principal components coefficient', fontsize=20)
plt.show()

# plot SP500 and PCA predictions for the first Stock
stockIndex = 1
plt.figure()
plt.plot(range(1,nTime+1), x[stockIndex,:],'-',linewidth=2,label='raw data')
plt.plot(range(1,nTime+1), xPCA[stockIndex,:],'-',linewidth=2,label='PCA compressed data')
plt.xlabel('Time/days, starting at 1/2/15', fontsize=20)
plt.ylabel('Stock price/USD', fontsize=20)
plt.title('raw data vs PCA compressed data, stock #1', fontsize=20)
plt.legend(loc='upper center', shadow=True, fontsize=20)
plt.show()