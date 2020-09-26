import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from math import *

def dxdt(x,y,z,h=2e-2):

    K1, L1, M1 = f1(x,y,z), f2(x,y,z), f3(x,y,z)
    dx, dy, dz = h*K1/2, h*L1/2, h*M1/2
    K2, L2, M2 = f1(x+dx,y+dy, z+dz), f2(x+dx,y+dy, z+dz), f3(x+dx,y+dy, z+dz)
    dx, dy, dz = h*K2/2, h*L2/2, h*M2/2
    K3, L3, M3 = f1(x+dx,y+dy, z+dz), f2(x+dx,y+dy, z+dz), f3(x+dx,y+dy, z+dz)
    dx, dy, dz = h*K3, h*L3, h*M3
    K4, L4, M4 = f1(x+dx,y+dy, z+dz), f2(x+dx,y+dy, z+dz), f3(x+dx,y+dy, z+dz)

    dx = (K1 + 2*K2 + 2*K3 + K4)*h/6
    dy = (L1 + 2*L2 + 2*L3 + L4)*h/6
    dz = (M1 + 2*M2 + 2*M3 + M4)*h/6
    return dx, dy, dz

def trajectory(initial_point = [0.1, 0.1, 0.1], num_points=1e4, h=2e-2):
    x0, y0, z0 = initial_point[0], initial_point[1], initial_point[2]
    n = int(num_points)
    x = np.zeros([n,3])
    x[0,:] = [x0,y0,z0]

    for k in range(1,n):
        dx,dy,dz = dxdt(x[k-1,0],x[k-1,1],x[k-1,2],h)
        x[k,0] = x[k-1,0] + dx
        x[k,1] = x[k-1,1] + dy
        x[k,2] = x[k-1,2] + dz

    return x.T


def f1(x, y, z):
    A = 10
    return A * (y - x)


def f2(x, y, z):
    B = 28;
    return B * x - y - x * z;


def f3(x, y, z):
    C = 8 / 3
    return x * y - C * z


if __name__ == '__main__':
    N = 5000
    scale = 50
    x = trajectory([0.1, 0.100001, 0.1], N) / scale
    x_ = trajectory([0.1, 0.1, 0.1], num_points=N) / scale
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.plot(*x, 'g')
    plt.show()
    plt.subplot(3, 1, 1)
    plt.plot(x[0], 'r'), plt.plot(x_[0], 'g')
    plt.subplot(3, 1, 2)
    plt.plot(x[1], 'r'), plt.plot(x_[1], 'g')
    plt.subplot(3, 1, 3)
    plt.plot(x[2], 'r'), plt.plot(x_[2], 'g')
    plt.show()

    np.savetxt('Lorenz_data.txt', x)

