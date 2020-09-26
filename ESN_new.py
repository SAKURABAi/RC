#! /usr/bin/env python3
# -*- coding: utf-8

import numpy as np
import networkx as nx
from decimal import Decimal
import matplotlib.pyplot as plt
from scipy.io import loadmat
import Lorenz_Runge_Kutta
import Lorenz_scipy_odeint

class Reservoir:
    def __init__(self, data):
        '''
        M:          number of nodes in input layer
        N:          number of nodes in reservoir
        input_len:  length of input
        start_node: start number when searching for optimal number of nodes in reservoir
        end_node:   end number when searching for optimal number of nodes in reservoir
        D:          average degree of reservoir
        alpha:      leakage rate (learning rate)
        beta:       parameter of ridge regression
        sigma:      range of randomized weight initialization, [-σ, σ]
        P:          number of nodes in output layer
        '''

        self.dataset = data

        # Input layer
        self.M = data.shape[0] # number of nodes in input layer
        self.input_len = data.shape[1] #

        # Reservoir layer
        self.start_node = 300
        self.N = self.start_node
        self.step = 1
        self.end_node = 305
        self.degree_func = lambda x: np.sqrt(x)
        self.D = self.degree_func(self.start_node)
        self.sigma = 0.5
        self.bias = 1
        self.alpha = 0.3
        self.beta = 1e-8
        self.rho_desired = 1.25

        # Output layer
        self.P = data.shape[0]

        # Training relevant
        self.init_len = 1000 # 前1000组数据用来初始化
        self.train_len = 3000 # 前3000组数据用来训练
        self.test_len = 2000 # 后2000组数据用来测试
        self.error_len = 1999

    def train(self):
        '''
        r:   state of reservoir at time t
        R:   record input and state while training
        S:   target of prediction
        A:   adjacent matrix of reservoir
        rho: spectral radius of original A
        '''

        # collection of reservoir state vectors
        self.R = np.zeros((1 + self.N + self.M, self.train_len - self.init_len)) # bias + input + reservoir
        # collection of input signals
        self.S = np.vstack([x[self.init_len + 1: self.train_len + 1] for x in self.dataset])
        self.r = np.zeros((self.N, 1))
        np.random.seed(43) #42
        self.Win = np.random.uniform(-self.sigma, self.sigma, (self.N, self.M + 1)) # +1 for bias
        # TODO: the values of non-zero elements are randomly drawn from uniform dist [-1, 1]
        g = nx.erdos_renyi_graph(self.N, self.D / self.N, seed=42, directed=True)
        # nx.draw(g, node_size=self.N)
        self.A = nx.adjacency_matrix(g).todense()
        self.rho = max(abs(np.linalg.eig(self.A)[0]))
        self.A *= self.rho_desired / self.rho

        # run the reservoir with the data and collect r
        for t in range(self.train_len):
            u = np.vstack([x[t] for x in self.dataset])
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + W_in * u(t) + bias)
            self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,self.r) + np.dot(self.Win, np.vstack((self.bias, u))))
            if t >= self.init_len:
                self.R[:, [t - self.init_len]] = np.vstack((self.bias, u, self.r))[:, 0]
        # train the output
        R_T = self.R.T  # Transpose
        # W_out*r = s
        # W_out = (s * r^T) * ((r * r^T) + beta * I)
        # 假设Wout同时依赖于r，u和bias
        self.Wout = np.dot(np.dot(self.S, R_T), np.linalg.inv(np.dot(self.R, R_T) + self.beta * np.eye(self.M + self.N + 1)))

    def _run(self):
        # run the trained ESN in alpha generative mode. no need to initialize here,
        # because r is initialized with training data and we continue from there.
        # 运行阶段，S表示预测值
        self.S = np.zeros((self.P, self.test_len))
        u = np.vstack([x[self.train_len] for x in self.dataset])
        for t in range(self.test_len):
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + Win * u(t) + bias)
            self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A, self.r) + np.dot(self.Win, np.vstack((self.bias, u))))
            s = np.dot(self.Wout, np.vstack((self.bias, u, self.r)))
            self.S[:, t] = np.squeeze(np.asarray(s))
            # use output as input
            u = s
        # compute Root Mean Square (RMS) error for the first self.error_len time steps
        self.RMS = []
        for i in range(self.P):
            self.RMS.append(sum(np.square(self.dataset[i, self.train_len+1: self.train_len+self.error_len+1] - self.S[i, 0: self.error_len])) / self.error_len)

    def draw(self):
        f, plots = plt.subplots(1, self.M, figsize=(self.M * 5, 5))
        plt.suptitle('N = ' + str(self.N) + ', Degree = %.5f' % (self.D))
        for i in range(self.M):
            p = plots[i]
            p.text(0.5, -0.1, 'RMS = %.15e' % self.RMS[i], size=10, ha="center", transform=p.transAxes)
            p.plot(self.S[i], label = 'prediction')
            p.plot(self.dataset[i][self.train_len + 1 : self.train_len + self.test_len + 1], label = 'input signal')
            p.legend(loc = 'upper right')
        #'''
        # plot in 3D
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot3D(*self.dataset, 'r')
        ax.plot3D(*self.S, 'g')
        #'''
        plt.show()

    def run(self):
        with open('reservoir.output', 'a') as output:
            for i in range(self.start_node, self.end_node + 1, self.step):
                self.N = i
                self.D = self.degree_func(self.N)
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

if __name__ == '__main__':

    scale = 50

    ''' matlab ode45 
    data = loadmat('Lorenz_data.mat')
    x = np.zeros((3, data['x'].shape[0]))
    x[0] = data['x'].ravel()
    x[1] = data['y'].ravel()
    x[2] = data['z'].ravel()
    x /= scale
    '''

    ''' Python Runge Kutta
    N = 5000
    x = Lorenz_Runge_Kutta.trajectory([0.1, 0.100001, 0.1], N) / scale
    '''

    # Python odeint
    t = np.arange(0, 100, 0.02)
    initial_val = (0.1, 0.100001, 0.1)
    args = (10.0, 28.0, 8.0 / 3.0)
    x = Lorenz_scipy_odeint.Lorenz_odeint(t, initial_val, args) / scale

    r = Reservoir(x)
    r.run()
