#! /usr/bin/env python3
# -*- coding: utf-8
'''
TODO:
    1. Wout of equation 6 in paper (falied)
    2. Lyapunov exponents
    3. Lorenz map to plot z_n+1/z_n
'''
import numpy as np
import networkx as nx
from decimal import Decimal
import matplotlib.pyplot as plt
from scipy.io import loadmat
import Lorenz_Runge_Kutta
import Lorenz_scipy_odeint

class Reservoir:
    def __init__(self, dataset, start_node=300, step=1, end_node=300, D=6, sigma=0.1, rho_desired=1.2, beta=0,
                 init_len=0, train_T=100, delta_t=0.02, test_T=25):
        '''
        M:          number of nodes in input layer
        N:          number of nodes in reservoir
        input_len:  length of input
        start_node: start number when searching for optimal number of nodes in reservoir
        end_node:   end number when searching for optimal number of nodes in reservoir
        D:          average degree of reservoir
        beta:       parameter of ridge regression
        sigma:      range of randomized weight initialization, [-σ, σ]
        P:          number of nodes in output layer
        map_len:    length to plot z_n+1/z_n of Lorenz map
        '''

        self.dataset = dataset

        # Input layer
        self.M = dataset.shape[0] # number of nodes in input layer
        self.input_len = dataset.shape[1] #

        # Reservoir layer
        self.start_node = start_node
        self.step = step
        self.end_node = end_node
        self.D = D
        self.sigma = sigma
        self.beta = beta
        self.rho_desired = rho_desired

        # Output layer
        self.P = dataset.shape[0]

        # Training relevant
        self.init_len = init_len
        self.train_len = np.int(train_T / delta_t)
        self.test_len = np.int(test_T / delta_t)
        self.error_len = self.test_len - 1

    def train(self):
        '''
        r:   state of reservoir at time t
        R:   record input and state while training
        S:   target of prediction
        A:   adjacent matrix of reservoir
        rho: spectral radius of original A
        '''

        # collection of reservoir state vectors
        self.R = np.zeros((self.N, self.train_len - self.init_len))
        R_z = np.zeros((self.N, self.train_len - self.init_len))
        # collection of input signals
        self.S = np.vstack([x[self.init_len + 1:self.train_len + 1] for x in self.dataset])
        self.r = np.zeros((self.N, 1))
        np.random.seed(43) #42
        self.Win = np.random.uniform(-self.sigma, self.sigma, (self.N, self.M))
        self.Wout = np.zeros((self.P, self.N))
        g = nx.erdos_renyi_graph(self.N, self.D / self.N, seed=42, directed=True)
        # nx.draw(g, node_size=self.N)
        plt.show()
        self.A = nx.adjacency_matrix(g).todense().astype(np.float)
        self.A[self.A != 0] = np.random.uniform(-1, 1, np.sum(self.A != 0))
        self.rho = max(abs(np.linalg.eig(self.A)[0]))
        self.A *= self.rho_desired / self.rho

        # run the reservoir with the data and collect r
        for t in range(self.train_len):
            u = np.vstack([x[t] for x in self.dataset])
            # r(t + \Delta t) = tanh(A * r(t) + W_in * u(t))
            self.r = np.tanh(np.dot(self.A,self.r) + np.dot(self.Win, u))
            if t >= self.init_len:
                self.R[:, [t - self.init_len]] = self.r[:, 0]
                R_z[0:np.int(self.N/2), [t - self.init_len]] = self.r[0:np.int(self.N/2), 0]
                R_z[np.int(self.N/2):self.N, [t - self.init_len]] = np.power(self.r[np.int(self.N/2):self.N, 0], 2)

        # train the output
        R_T = self.R.T  # Transpose
        R_z_T = R_z.T
        # W_out*r = s
        # W_out = (s * r^T) * ((r * r^T) + beta * I)^-1
        self.Wout = np.dot(np.dot(self.S, R_T), np.linalg.inv(np.dot(self.R, R_T)))
        '''
        # 假设Wout只依赖于reservoir的状态，根据论文方程6修改最小二乘法结果，有bug
        self.Wout[0, :] = np.dot(np.dot(self.S[0, :], R_T), np.linalg.inv(np.dot(self.R, R_T)))
        self.Wout[1, :] = np.dot(np.dot(self.S[1, :], R_T), np.linalg.inv(np.dot(self.R, R_T)))
        self.Wout[2, :] = np.dot(np.dot(self.S[2, :], R_z_T), np.linalg.inv(np.dot(R_z, R_z_T)))
        #'''
    def _run(self, len=None, calcRMS=True):
        # run the trained ESN in alpha generative mode. no need to initialize here,
        # because r is initialized with training data and we continue from there.
        # 运行阶段，S表示预测值，预测时用输出值作为新的输入
        if len == None:
            len = self.test_len
        self.S = np.zeros((self.P, len))

        u = np.vstack([x[self.train_len] for x in self.dataset])
        for t in range(len):
            # r(t + \Delta t) = tanh(A * r(t) + Win * Wout * r(t))
            self.r = np.tanh(np.dot(self.A, self.r) + np.dot(self.Win, u))
            s = np.dot(self.Wout, self.r)
            self.S[:, t] = np.squeeze(np.asarray(s))
            # use output as input
            u = s

        if calcRMS:
            # compute Root Mean Square (RMS) error for the first self.error_len time steps
            self.RMS = []
            for i in range(self.P):
                self.RMS.append(sum(np.square(self.dataset[i, self.train_len + 1:self.train_len + self.error_len + 1] - self.S[i, 0:self.error_len])) / self.error_len)

    def draw(self):
        f, plots = plt.subplots(self.M, 1, figsize=(self.M * 5, 5))
        plt.suptitle('N = ' + str(self.N) + ', Degree = %.5f' % (self.D))
        for i in range(self.M):
            p = plots[i]
            p.text(0.5, -0.1, 'RMS = %.15e' % self.RMS[i], size=10, ha="center", transform=p.transAxes)
            p.plot(self.S[i], label = 'prediction')
            p.plot(self.dataset[i][self.train_len + 1:self.train_len + self.test_len + 1], label = 'input signal')
            p.legend(loc = 'upper right')
        #'''
        # plot in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(*self.dataset, 'r')
        ax.plot3D(*self.S[:, 500:1250], 'g')
        #'''
        plt.show()

    def run(self):
        for i in range(self.start_node, self.end_node + 1, self.step):
            self.N = i
            # self.D = self.degree_func(self.N)
            self.train()
            self._run()
            self.summary()
            self.draw()

    def summary(self):
        res = 'N= ' + str(self.N) + ', D= ' + '%.15f' % self.D + ', '
        for j in range(self.P):
            res += 'RMS_{}= {:.5e}, '.format(j, Decimal(self.RMS[j]))
        res += '\n'
        print(res)

def plot_z_max(z_data, z_RC, map_len):
    zz_RC, zz_data = [], []
    for i in range(1, map_len - 1):
        if z_data[i] > z_data[i - 1] and z_data[i] > z_data[i + 1]:
            zz_data.append(z_data[i])
        # if z_RC[i] > z_RC[i - 1] and z_RC[i] > z_RC[i + 1] and z_RC[i] > 30 and z_RC[i] < 50:
        if z_RC[i] > z_RC[i - 1] and z_RC[i] > z_RC[i + 1]:
            zz_RC.append(z_RC[i])

    plt.figure()
    plt.subplot(121)
    plt.scatter(zz_RC[1000:-1], zz_RC[1001:], label='RC', linewidths=0.5)
    plt.xlabel(r'$z_n$', fontsize=14)
    plt.ylabel(r'$z_{n+1}$', fontsize=14)
    plt.legend(fontsize=14)
    plt.subplot(122)
    plt.scatter(zz_data[:-1], zz_data[1:], label='data', linewidths=0.5)
    plt.xlabel(r'$z_n$', fontsize=14)
    plt.ylabel(r'$z_{n+1}$', fontsize=14)
    plt.legend(fontsize=14)
    plt.show()



if __name__ == '__main__':

    LORENZ_MAP_REFERENCE_SOLUTION = np.array([-14.57, 0, 0.90])
    LORENZ_MAP_INITIAL_CONDITION = np.array([-5.76, 2.27, 32.82])
    N = np.int((100 + 25 + 1000) / 0.02)
    map_len = np.int(1000 / 0.02)

    ''' matlab ode45 
    data = loadmat('Lorenz_data.mat')
    x = np.zeros((3, data['x'].shape[0]))
    x[0] = data['x'].ravel()
    x[1] = data['y'].ravel()
    x[2] = data['z'].ravel()
    '''

    ''' Python Runge Kutta'''
    x = Lorenz_Runge_Kutta.trajectory(LORENZ_MAP_INITIAL_CONDITION, N)

    '''
    # Python odeint
    t = np.arange(0, 125, 0.02)
    args = (10.0, 28.0, 8.0 / 3.0)
    x = Lorenz_scipy_odeint.Lorenz_odeint(t, initial_val, args)
    '''
    r = Reservoir(x, rho_desired=1.2)
    r.run()
    # plot z_n+1 over z_n
    r._run(len=map_len, calcRMS=False)
    plot_z_max(r.dataset[2, r.train_len + r.test_len:], r.S[2, :], map_len)
