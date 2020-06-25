import matplotlib.pyplot as plt
import torch
import numpy as np
from wbident import PSO, Model

class EqSystem(Model):
    def __init__(self, params=None):
        super(EqSystem, self).__init__(params)
        self._params = params

    def model(self, t, x):
        def delta(vel):
            if abs(vel) > 0.1:
                d = 5.0
            else:
                d = 0.5
            return d
        k = self.unknown_const
        ks   = k[0]
        c    = k[1]
        w    = k[2]
        m    = 1
        wn   = np.sqrt(k[0]/m)
        zeta = k[1]/(2*m*wn)
        dx = torch.zeros(len(self.x0),)
        dx[0] = x[1]
        dx[1] = -2 * zeta * wn * delta(x[1])*x[1] - wn ** 2 * x[0] + 4*np.sin(2*np.pi*k[2]*t)
        return dx

def suface_plot(params):
    f_fit = EqSystem(params)
    k = torch.tensor([5.0,5.0,0.1],dtype=torch.float32)
    f_fit.y = f_fit.simulation(k)
    x = torch.linspace(1, 10,10, dtype=torch.float32)
    y = torch.linspace(1, 10,10, dtype=torch.float32)
    xlen = len(x)
    ylen = len(y)
    xg, yg = np.meshgrid(x, y)
    cost = torch.zeros(xlen, ylen)
    for i in range(xlen):
        for j in range (ylen):
            k = torch.tensor([x[i], y[j]])
            cost[i,j],_,_= f_fit.evaluate(k)
            print(f"Position {x[i]}, {y[j]}, cost {cost[i,j]}")
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(xg, yg, cost.numpy(), rstride=1, cstride=1,cmap=plt.cm.jet, linewidth=0, antialiased=False)
    plt.show()

def main():
    params = {'optmizer': {'lowBound': [0.5, 0.5,0.1],
                            'upBound': [10, 10, 2],
                            'maxVelocity': 5, 
                            'minVelocity': -5,
                            'nPop': 15,
                            'nVar': 3,
                            'social_weight': 2,
                            'cognitive_weight': 2,
                            'w': 0.9,
                            'beta': 0.1,
                            'w_damping': 0.99},
                'dyn_system': {'model_path': '',
                                'x0': [0., 0.],
                                't': [0,6,1000]
                                }
                }
    # suface_plot(params)
    f_fit = EqSystem(params)
    k = torch.tensor([4.5,5.3, 0.5],dtype=torch.float32)
    f_fit.y = f_fit.simulation(k)
    pso = PSO(f_fit, params)
    cost = []
    position = []
    fig = plt.figure(figsize=(12, 4), facecolor='white')
    particle = fig.add_subplot(221, frameon=False)
    graph_cost = fig.add_subplot(222, frameon=False)
    func1  = fig.add_subplot(223, frameon=False)
    func2  = fig.add_subplot(224, frameon=False)

    plt.show(block=False)

    for i in range(100):
        print(f'interation {i} {pso.pbg_cost},{pso.pbg_position}')
        particle.cla()
        for j in range(params['optmizer']['nPop']):
            particle.plot(pso.p_position_[j, 0], pso.p_position_[j, 1], '.')
        particle.set_xlim(params['optmizer']['lowBound'][0],
                          params['optmizer']['upBound'][0], )
        particle.set_ylim(params['optmizer']['lowBound'][1],
                          params['optmizer']['upBound'][1], )
        particle.grid()

        if pso.pbg_cost != float('inf'):
            cost.append(pso.pbg_cost.numpy()[0])
            position.append(pso.pbg_position.numpy()[0])
        graph_cost.cla()
        graph_cost.plot(cost)
        graph_cost.set_xlim(0, 100)
        graph_cost.set_title('Error')

        func1.cla()
        func1.plot(pso.y[:,0])
        func1.plot(pso.pbg_y_hat[:,0],'--')
        func1.plot(pso.y[:,1])
        func1.plot(pso.pbg_y_hat[:,1],'--')
        func1.legend(['y0','y0_hat','y1','y1_hat'])

        func2.cla()
        func2.plot(pso.y[:,0],pso.y[:,1])
        func2.plot(pso.pbg_y_hat[:,0], pso.pbg_y_hat[:,1],'--')
        func2.legend(['y','y_hat'])
        func2.set_xlabel('y')
        func2.set_ylabel('dot_y')

        plt.draw()
        plt.pause(0.01)
        pso.run()

if __name__ == "__main__":
    main()