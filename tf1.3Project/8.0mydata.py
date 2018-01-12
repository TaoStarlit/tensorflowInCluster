import numpy as np
import matplotlib.pyplot as plt




n=100
b=6
'''
mu1=[0.1,0.3,0.5]
mu2=[0.8,0.4,0.6]
sigma1=[0.01,0.02,0.03]
sigma2=[0.05,0.01,0.02]




plt.figure("np.random.normal")
for i in range(len(mu1)):
    x = np.random.normal(mu1[i], sigma1[i], n)
    y = np.random.normal(mu2[i], sigma2[i], n)
    plt.plot(x, y, 'x')


plt.show()
'''


mean=[
    [0.3,0.5],
    [0.4,0.2],
    [0.8,0.7]]

sigmax=[0.01,0.04,0.03]
sigmay=[0.03,0.02,0.02]
cov=[
        [[sigmax[0]**2,0.03**2],[0.03**2,sigmay[0]**2]],# variance = standard deviation ** 2    sigma*2
        [[sigmax[1]**2,0.00],[0.00,sigmay[1]**2]],
        [[sigmax[2]**2,0.00],[0.00,sigmay[2]**2]]]

wantedx=[]
wantedy=[]
plt.figure("np.random.multivariate_normal")
for i in range(len(mean)):
    x,y= np.random.multivariate_normal(mean[i], cov[i], n).T
    plt.plot(x, y, 'x')
    wantedx.append(x)
    wantedy.append(y)


x0 = np.random.uniform(0,1,3*n)
y0 = np.random.uniform(0,1,3*n)
delete_index=[]

for j in range(3*n):
    for i in range(len(mean)):
        if((x0[j]>mean[i][0]-b*sigmax[i] and x0[j]<mean[i][0]+b*sigmax[i])
            and (y0[j]>mean[i][1]-b*sigmay[i] and y0[j]<mean[i][1]+b*sigmay[i])):

            print("delete", j, x0[j], y0[j])
            delete_index.append(j)

            break


x=np.delete(x0,delete_index)
y=np.delete(y0,delete_index)
print("x0, y0",len(y0))
print("x, y",len(y))
plt.plot(x[0:n], y[0:n], 'x')
wantedx.append(x[0:n])
wantedy.append(y[0:n])

plt.axis([0, 1, 0, 1])
plt.show()


plt.figure("final show")

for i in range(len(mean)+1):
    plt.plot(wantedx[i], wantedy[i], 'x')

plt.show()