#! /usr/bin/env python3
# -*- coding: utf-8
'''
TODO:

'''
import numpy as np
import networkx as nx
from decimal import Decimal
import matplotlib.pyplot as plt
from scipy.io import loadmat

class Reservoir:
    def __init__(self, dataset, start_node=300, step=1, end_node=300, sigma=0.1, rho_desired=1.2, beta=1e-8,
                 alpha=0.3, bias=1.0, init_len=0, train_T=100, delta_t=0.02, test_T=25):
        '''
        dataset:    format: M × len
        M:          number of nodes in input layer
        N:          number of nodes in reservoir
        input_len:  length of input
        start_node: start number when searching for optimal number of nodes in reservoir
        end_node:   end number when searching for optimal number of nodes in reservoir
        D:          average degree of reservoir
        alpha:      leaking rate(learning rate)
        beta:       parameter of ridge regression
        sigma:      range of randomized weight initialization, [-σ, σ]
        P:          number of nodes in output layer
        '''

        self.dataset = dataset

        # Input layer
        self.M = dataset.shape[0] # number of nodes in input layer
        self.input_len = dataset.shape[1] #

        # Reservoir layer
        self.start_node = start_node
        self.step = step
        self.end_node = end_node
        self.degree_func = lambda x: np.sqrt(x)
        self.sigma = sigma
        self.bias = bias
        self.alpha = alpha
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
            # r(t + \Delta t) = (1 - alpha)r(t) + alpha * tanh(A * r(t) + W_in * u(t) + bias)
            self.r = (1- self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A,self.r) + np.dot(self.Win, u) + self.bias)
            if t >= self.init_len:
                self.R[:, [t - self.init_len]] = self.r[:, 0]

        # train the output
        R_T = self.R.T  # Transpose

        # W_out*r = s
        # W_out = (s * r^T) * ((r * r^T) + beta * I)^-1
        self.Wout = np.dot(np.dot(self.S, R_T), np.linalg.inv(np.dot(self.R, R_T)))

    def predict(self, len=None, calcRMS=True):
        # run the trained ESN in alpha generative mode. no need to initialize here,
        # because r is initialized with training data and we continue from there.
        # 运行阶段，S表示预测值，预测时用输出值作为新的输入
        if len == None:
            len = self.test_len
        self.S = np.zeros((self.P, len))

        u = np.vstack([x[self.train_len] for x in self.dataset])
        for t in range(len):
            # r(t + \Delta t) = tanh(A * r(t) + Win * Wout * r(t))
            self.r = (1 - self.alpha) * self.r + self.alpha * np.tanh(np.dot(self.A, self.r) + np.dot(self.Win, u) + self.bias)
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
        plt.savefig('test.jpg')
        plt.show()

    def run(self):
        for i in range(self.start_node, self.end_node + 1, self.step):
            self.N = i
            self.D = self.degree_func(self.N)
            self.train()
            self.predict()
            self.summary()
            self.draw()

    def summary(self):
        res = 'N= ' + str(self.N) + ', D= ' + '%.15f' % self.D + ', '
        for j in range(self.P):
            res += 'RMS_{}= {:.5e}, '.format(j, Decimal(self.RMS[j]))
        res += '\n'
        print(res)


if __name__ == '__main__':

    data = loadmat('F:\BehaviorTest\\20191027\wenIs1035\B1\curvatures.mat')
    x = data['curvatures'].T
    r = Reservoir(x, init_len=100, test_T=800, train_T=1000, delta_t=1, rho_desired=1.2)
    r.run()
