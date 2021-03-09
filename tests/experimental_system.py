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
        Ps   = 5e5 + Patm                   # Pressão suprimento
        M = 1.323+0.146                     # Massa Total
        d1 = 0.025
        d2 = 0.005
        A1 = np.pi * (d1/2)**2              # Área do êmbolo
        A2 = np.pi * (d2/2)**2                         
        A2 = A2 - A1 
        Ao = A1*2                           # Área Orifício
        Vb0 = 0.056                         # Volume morto da Câmara A
        Va0 = 0.144                         # Volume morto da Câmara B
        R = 287                             # Constante universal dos gases
        T = 293                             # Temperatura do ar de suprimento
        L = 0.125                           # Curso útil do cilindro
        kv = 410                            # Coeficiente de atrito viscoso
        gamma = 1.4                         # Relação entre os calores específicos do ar
        g = 9.8                             # Força Gravitacional

        u = args[0][0]

        k = self.unknown_const
        # # rewrite unknown variables
        kv =    k[0]
        Ps =    k[1]
        Ao =    k[2]


        def Psi(sigma):
            if sigma > 0.528:
                # subsonic flow
#                 psi = np.sqrt( gamma/(gamma-1) * ((sigma)**(2/gamma) - (sigma)**((gamma+1)/gamma)) )
                psi = (2/(gamma+1) )**(1/(gamma-1)) * np.sqrt(gamma/(gamma+1))
            else:
                # chocked flow
                        psi = (2/(gamma+1) )**(1/(gamma-1)) * np.sqrt(gamma/(gamma+1))
            return psi

        def dm1(Ao, Pc):
            if Pc < Ps:
                # charging
                sigma = Pc/Ps
                dm =   Ao*Ps*np.sqrt( 2*gamma / (R*T*(gamma-1))) * Psi(sigma)
            elif Pc > Ps:
                # discharging
                sigma = Ps/Pc
                dm = - Ao*Pc*np.sqrt( 2*gamma / (R*T*(gamma-1))) * Psi(sigma)
            else:
                dm = 0
            return dm

        # Dynamic Model relief valve
        def dm2(Ao, Pc):
            if Pc < Patm:
                # charging
                sigma = Pc/Patm
                dm =   Ao*Patm*np.sqrt( 2*gamma / (R*T*(gamma-1))) * Psi(sigma)
            elif Pc > Patm:
                # discharging
                sigma = Patm/Pc
                dm = - Ao*Pc*np.sqrt(   2*gamma / (R*T*(gamma-1))) * Psi(sigma)
            else:
                dm = 0
            return dm
        
        dy = torch.zeros(len(self.x0),)

        dy[0] = y[1]
        dy[1] = ( (y[2]*A1+y[3]*A2)  - kv*y[1] )/M -g
        # dy[1] = ( Ps*u*A1  - kv*y[1] + Fc(y[0], y[1]))/M -g        
        x_a = 0.5*L + y[0]
        VA = Va0 + A1*(x_a)
        if VA < Va0:
            VA = Va0
        # connect to    [Supply            ]   [Atm                  ]
        dy[2] = 1/(VA)*(R*T*dm1(Ao*(u), y[2]) + R*T*dm2(Ao*(1-u), y[2]) - y[2]*y[1]*A1)
        x_b = 0.5*L - y[0]
        VB = Vb0 + A2*(x_b)
        if VB < Vb0:
            VB = Vb0   
        # connect to    [Supply            ]   [Atm                ]
        dy[3] = 1/(VB)*(R*T*dm1(Ao*(0), y[3]) + R*T*dm2(Ao*(0   ), y[3]) + y[3]*y[1]*A2)
        return dy

def main():
    def v2Pascal(value):
        Patm = 1e5
        gainV2P = 1000e3/10
        value = value * gainV2P + Patm
        return value 
    # load data
    data = loadmat('./data/teste0902_02.mat')
    sample_select = 0.3
    t = data['t'].reshape(-1,1)
    t_len = len(t)
    N = int( t_len * sample_select)
    y0 = - 0.125/2 + data['desl']/1000
    y0 = y0[:N,:]
    u = data['atuador']
    u = u[:N,:]
    pa = v2Pascal(data['press']) 
    pa = pa[:N,:] 
    t = t[:N,:]
    ts = (t[1]-t[0])[0]
    fs = 1/(ts)
    fs = int(fs)
    del data

        # kv =    k[0]
        # Ao =    k[1]
        # Ps =    k[2]

    params = {'optmizer': {'lowBound': [100.0,
                                        1e5,
                                        0.001],
                            'upBound': [1000,
                                        7e5,
                                        0.1],
                            'maxVelocity':  100, 
                            'minVelocity': -100,
                            'nPop': 5,
                            'nVar': 3,
                            'social_weight': 2,
                            'cognitive_weight': 2,
                            'w': 0.9,
                            'beta': 1,
                            'w_damping': 0.99},
                'dyn_system': {'model_path': '',
                                'external': u,
                                'state_mask' : [True, False, False, False],
                                'x0':  [y0[0,0], 
                                       (y0[1,0]-y0[0,0])/ts,
                                       pa[0,0],
                                       pa[0,0]],
                                't': [0,10-ts,len(t)],
                                }
                }

    f_fit = EqSystem(params)
    # experimental values
    zero_vec =  np.zeros((len(t),1))
    exp_values = np.hstack((y0,zero_vec ,pa, zero_vec))
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
        func1.plot(pso.pbg_y_hat[:,0],'--')
        func1.legend(['y0','y0_hat'])

        func2.cla()
        func2.plot(pso.y[:,2])
        func2.plot(pso.pbg_y_hat[:,2],'--')
        func2.legend(['y2','y2_hat'])

        plt.draw()
        plt.pause(0.01)

        pso.run()

if __name__ == "__main__":
    main()