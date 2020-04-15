from wbident import PSO
import torch
from torch import nn


class EqSystem:
    def __init__(self, parameters={}):
        super(EqSystem, self).__init__()
        self._params = parameters

    def parameters_initializer(self):
        return None

    def forward(x):
        y=0
        for i in range(len(x)):
            y+=x[i]**2
        return y
        
    def ode45(f, t, x0, *args):
        """
        4th Order Runge-Kutta method
        """
        def ode45_step(f, x, t, dt, *args):
            k = dt
            k1 = k * f(t, x, *args)
            k2 = k * f(t + 0.5*k, x + 0.5*k1, *args)
            k3 = k * f(t + 0.5*k, x + 0.5*k2, *args)
            k4 = k * f(t + dt, x + k3, *args)
            return x + 1/6. * (k1 + 2*k2 + 2*k3 + k4)

        n = len(t)
        x = torch.zeros((n, x0.shape[1] ))
        x[0] = x0[0][:]
        for i in range(n-1):
            dt = t[i+1] - t[i] 
            x[i+1] = ode45_step(f, x[i].reshape(1,-1),
                                t[i], dt, *args)
        return x


def main():
    params={'lowBound':[-10, -10],
            'upBound': [10, 10],
            'maxVelocity': 2, #TODO create a vector
            'minVelocity': -2,
            'nPop':200,
            'nVar': 2,
            'social_weight': 2,
            'cognitive_weight': 2,
            'w': 0.9,
            'w_damping': 0.99}

    pso = PSO(EqSystem,params)

if __name__ == "__main__":
    main()

