import matplotlib.pyplot as plt
import sys
import torch
import numpy as np
from wbident import PSO, Model

class EqSystem(Model):
    def __init__(self, params=None):
        super(EqSystem, self).__init__(params)
        self._params = params

    def model(self, t, x):
        # def delta = 

        k = self.unknown_const
        ks   = k[0]
        c    = k[1]
        wn   = np.sqrt(k[0]/2)
        zeta = k[1]/(2*2*wn)

        dx = torch.zeros(len(self.x0),)
        dx[0] = x[1]
        dx[1] = -2 * zeta * wn * x[1] - wn ** 2 * x[0]
        return dx


def main():
    params = {'optmizer': {'lowBound': [0.5, 0.5],
                            'upBound': [255, 10],
                            'maxVelocity': 5, 
                            'minVelocity': -5,
                            'nPop': 10,
                            'nVar': 2,
                            'social_weight': 2,
                            'cognitive_weight': 2,
                            'w': 0.9,
                            'w_damping': 0.99},
                'dyn_system': {'model_path': 'tests/data.mat',
                                'x0': [0., 1.],
                                't': [0,6,100]
                                }
                }

    f_fit = EqSystem(params)
    k = torch.tensor([120,5])
    f_fit.y_true = f_fit.simulation(k)

    pso = PSO(f_fit, params)
    cost = []
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    particle = fig.add_subplot(221, frameon=False)
    graph_cost = fig.add_subplot(222, frameon=False)
    func1  = fig.add_subplot(223, frameon=False)
    func2  = fig.add_subplot(224, frameon=False)

    plt.show(block=False)

    for i in range(100):
        print(f'interation {i} {pso.pbg_cost},{pso.pbg_position}')
        pso.run()
        particle.cla()
        for j in range(params['optmizer']['nPop']):
            particle.plot(pso.p_position_[j, 0], pso.p_position_[j, 1], '.')
        particle.set_xlim(params['optmizer']['lowBound'][0],
                          params['optmizer']['upBound'][0], )
        particle.set_ylim(params['optmizer']['lowBound'][1],
                          params['optmizer']['upBound'][1], )
        particle.grid()

        cost.append(pso.pbg_cost.numpy()[0])
        graph_cost.cla()
        graph_cost.plot(cost)
        graph_cost.set_xlim(0, 100)
        graph_cost.set_title('Error')

        func1.cla()
        func1.plot(pso.y_true[:,0])
        func1.plot(pso.y_pred[:,0])
        func1.set_ylabel('x_0')

        func2.cla()
        func2.plot(pso.y_true[:,1])
        func2.plot(pso.y_pred[:,1])
        func2.set_ylabel('x_1')

        plt.draw()
        plt.pause(0.01)


if __name__ == "__main__":
    main()