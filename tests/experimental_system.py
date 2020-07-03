import matplotlib.pyplot as plt
import torch
import numpy as np
from wbident import PSO, Model
from scipy.io import loadmat

class EqSystem(Model):
    def __init__(self, params=None):
        super(EqSystem, self).__init__(params)
        self._params = params

    def model(self, t, y, *args):
        Patm = 1e5                          # Pressão atmosférica
        Ps   = 3e5 + Patm                   # Pressão suprimento
        M = 1.323+0.146                     # Massa Total
        A1 = 0.025                         # Área do êmbolo
        A2 = 0.01                          
        A3 = A1 - A2
        Ao = 0.001                           # Área Orifício
        Vb0 = 0.056                         # Volume morto da Câmara A
        Va0 = 0.144                         # Volume morto da Câmara B
        R = 287                             # Constante universal dos gases
        T = 293                             # Temperatura do ar de suprimento
        L = 0.2                             # Curso útil do cilindro
        kv = 90                            # Coeficiente de atrito viscoso
        gamma = 1.4                         # Relação entre os calores específicos do ar
        g = 9.8                             # Força Gravitacional

        u = args[0][0]

        k = self.unknown_const

        def Psi(sigma):
            if sigma >= 0.528 and sigma <= 1:
                psi = 2 * 0.259 * np.sqrt(sigma*(1-sigma))
            elif sigma < 0.528 and sigma > 0:
                psi = 0.259
            else:
                print('Error')
                psi = 0
            return psi

        # Dynamic Model Valvule 1
        def dm1(Ao, Pc):
            if Pc <= Ps:
                # charging
                sigma = Pc/Ps
                dm =   Ao*Ps*np.sqrt( 2*gamma / (R*T*(gamma-1))) * Psi(sigma)
            else:
                # discharging
                sigma = Ps/Pc
                dm = - Ao*Pc*np.sqrt( 2*gamma / (R*T*(gamma-1))) * Psi(sigma)
            return dm

        # Dynamic Model Valvule 2
        def dm2(Ao, Pc):
            if Pc <= Patm:
                # charging
                sigma = Pc/Patm
                dm =   Ao*Patm*np.sqrt( 2*gamma / (R*T*(gamma-1))) * Psi(sigma)
            else:
                # discharging
                sigma = Patm/Pc
                dm = - Ao*Pc*np.sqrt( 2*gamma / (R*T*(gamma-1))) * Psi(sigma)
            return dm
        
        dy = torch.zeros(len(self.x0),)
        dy[0] = y[1]
        dy[1] = ( y[2]*A1 - y[3]*A2 - Patm*A3 - kv*y[1] )/M -g
        dy[2] = 1/(Va0 + A1*(0.5*L + y[0]))*(R*T*dm1(Ao*(u), y[2])  + R*T*dm2(Ao*(1-u), y[2]) - y[2]*y[1]*A1)
        dy[3] = 1/(Vb0 + A2*(0.5*L - y[0]))*(R*T*dm1(Ao*(0), y[3])  + R*T*dm2(Ao*(1), y[3])   + y[3]*y[1]*A2)
        return dy


def main():
    # load data
    data = loadmat('./data/teste0902_02.mat')
    y0 = data['desl']/1000
    u = data['atuador']
    pu = data['press'] * 1e5 + 1e5
    t = data['t'].reshape(-1,1)
    ts = (t[1]-t[0])[0]
    fs = 1/(ts)
    fs = int(fs)
    del data

    params = {'optmizer': {'lowBound': [0.5,
                                        0.5,
                                        0.1],
                            'upBound': [10,
                                        10,
                                        2],
                            'maxVelocity': 5, 
                            'minVelocity': -5,
                            'nPop': 1,
                            'nVar': 3,
                            'social_weight': 2,
                            'cognitive_weight': 2,
                            'w': 0.9,
                            'beta': 0.1,
                            'w_damping': 0.99},
                'dyn_system': {'model_path': '',
                                'external': u,
                                'state_mask' : [1., 0., 1., 0],
                                'x0':  [0, 
                                       0,
                                       1e5,
                                       1e5],
                                't': [0,10-ts,len(t)],
                                }
                }

    f_fit = EqSystem(params)
    k = torch.tensor([5.0,5.0,0.1],dtype=torch.float32)
    f_fit.y = f_fit.simulation(k)
    plt.subplot(4,1,1)
    plt.plot(f_fit.y[:-1,0])
    
    plt.subplot(4,1,2)
    plt.plot(f_fit.y[:-1,1])
    
    plt.subplot(4,1,3)
    plt.plot(f_fit.y[:-1,2])
    plt.subplot(4,1,4)
    plt.plot(f_fit.y[:-1,3])
    plt.show()
    return 0

    # experimental values
    zero_vec =  np.zeros((len(t),1))
    exp_values = np.hstack((y0,zero_vec ,pu, zero_vec))
    f_fit.y = torch.from_numpy(exp_values)

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
        # func1.plot(pso.pbg_y_hat[:,0],'--')
        func1.plot(pso.y[:,1])
        # func1.plot(pso.pbg_y_hat[:,1],'--')
        func1.legend(['y0','y0_hat','y1','y1_hat'])

        func2.cla()
        func2.plot(pso.y[:,0],pso.y[:,1])
        # func2.plot(pso.pbg_y_hat[:,0], pso.pbg_y_hat[:,1],'--')
        func2.legend(['y','y_hat'])
        func2.set_xlabel('y')
        func2.set_ylabel('dot_y')

        plt.draw()
        plt.pause(0.01)
        # pso.run()

if __name__ == "__main__":
    main()