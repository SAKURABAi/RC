from scipy.integrate import odeint
import numpy as np
import matplotlib.pyplot as plt

def lorenz(w, t, p, r, b):
    # 位置矢量w， 三个参数p, r, b
    x, y ,z = w.tolist()
    # 分别计算dx/dt, dy/dt, dz/dt
    return p * (y-x), x*(r-z)-y, x*y-b*z

def Lorenz_odeint(t, initial_val, args):
    track = odeint(lorenz, initial_val, t, args)
    return track.T

if __name__ == '__main__':
    t = np.arange(0, 100, 0.02)
    initial_val = (0.0, 1.00, 0.0)
    args = (10.0, 28.0, 8.0 / 3.0)
    X, Y, Z = Lorenz_odeint(t, initial_val, args)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot(X, Y, Z, label='lorenz')
    ax.legend()
    plt.show()
