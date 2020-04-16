import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
from wbident import PSO
from scipy.io import loadmat

sys.path.append('./')


class EqSystem:
    def __init__(self, params=None):
        super(EqSystem, self).__init__()
        if params is None:
            parameters = {}
        self._params = params
        self._unknown_const = None
        self._path = None
        self.x0 = None
        self.data = None
        self.parameters_initializer()

    def parameters_initializer(self):
        self._params = self._params['dyn_system']
        self._path = self._params['model_path']
        self.x0 = torch.tensor([self._params['x0']])
        sim = loadmat(self._path)
        data_reduction = 20
        self.data = np.hstack((sim['t'].reshape(-1, 1), sim['desl'], sim['press']))[::data_reduction, :]
        return None

    @property
    def k(self):
        return self._unknown_const

    @k.setter
    def k(self, value):
        self._unknown_const = value

    def evaluate(self, k):
        def model(t, x):
            x = x.T

            mL = 1.03 / 2
            mP = 1.03 / 2
            A1 = 25e-3
            A2 = 10e-3
            Pa = 101400
            g = 9.8

            u = 0

            dx = torch.zeros(2, 1)
            dx[0] = x[1]
            dx[1] = (A1*u-(A1-A2)*Pa - k[0] * x[1])/(mL+mP)-k[1]
            return dx.T
        self.k = k
        t = torch.tensor([self.data[:, 0]]).T
        y_true = self.ode45(model, self.data[:, 0], self.x0)
        y_pred = torch.tensor(self.data[:, 1:3])
        mse = torch.sub(y_true, y_pred).pow(2).mean()
        # mse = k.pow(2).sum()
        return mse

    def ode45(self, f, t, x0, *args):
        """
        4th Order Runge-Kutta method
        """

        def ode45_step(f, x, t, dt, *args):
            k = dt
            k1 = k * f(t, x, *args)
            k2 = k * f(t + 0.5 * k, x + 0.5 * k1, *args)
            k3 = k * f(t + 0.5 * k, x + 0.5 * k2, *args)
            k4 = k * f(t + dt, x + k3, *args)
            return x + 1 / 6. * (k1 + 2 * k2 + 2 * k3 + k4)

        n = len(t)
        x = torch.zeros((n, x0.shape[1]))
        x[0] = x0[0][:]
        for i in range(n - 1):
            dt = t[i + 1] - t[i]
            x[i + 1] = ode45_step(f, x[i].reshape(1, -1),
                                  t[i], dt, *args)
        return x


def main():
    params = {'optmizer': {'lowBound': [0, 0],
                           'upBound': [200, 200],
                           'maxVelocity': 5,  # TODO create a vector
                           'minVelocity': -5,
                           'nPop': 100,
                           'nVar': 2,
                           'social_weight': 2,
                           'cognitive_weight': 2,
                           'w': 0.9,
                           'w_damping': 0.99},
              'dyn_system': {'model_path': 'data.mat',
                             'x0': [0., 0.]
                             }
              }

    f_fit = EqSystem(params)
    pso = PSO(f_fit, params)
    cost = []
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    particle = fig.add_subplot(121, frameon=False)
    graph_cost = fig.add_subplot(122, frameon=False)
    plt.show(block=False)


    for i in range(100):
        print(f'interation {i} {pso.pbg_cost},{pso.pbg_position}')
        pso.run()
        particle.cla()
        for j in range(params['optmizer']['nPop']):
            particle.plot(pso.p_position_[j, 0], pso.p_position_[j, 1], '.')
        particle.set_xlim(params['optmizer']['lowBound'][0],
                          params['optmizer']['upBound'][0],)
        particle.set_ylim(params['optmizer']['lowBound'][1],
                          params['optmizer']['upBound'][1],)
        particle.grid()

        cost.append(pso.pbg_cost.numpy()[0])
        graph_cost.cla()
        graph_cost.plot(cost)
        graph_cost.set_xlim(0, 100)

        plt.draw()
        plt.pause(0.01)


if __name__ == "__main__":
    main()
