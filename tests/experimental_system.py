import matplotlib.pyplot as plt
import torch
import numpy as np
from wbident import PSO, Model
from scipy.io import loadmat

class EqSystem(Model):
    def __init__(self, params=None):
        super(EqSystem, self).__init__(params)
        self._params = params

    def model(self, t, x, *args):
        Patm = 0                          # Pressão atmosférica
        Ps   = 6e5 + Patm                   # Pressão suprimento
        M = 1.323+0.146                     # Massa Total
        A = 4.9e-4                          # Área do êmbolo
        A1 = 0                              # Área haste ?
        Ao = 0.001                          # Área Orifício
        Vu0 = 2.5e-4                        # Volume morto da Câmara A
        Vb0 = 2.5e-4                        # Volume morto da Câmara B
        R = 287                             # Constante universal dos gases
        T = 293                             # Temperatura do ar de suprimento
        L = 0.2                               # Curso útil do cilindro
        kv = 90                             # Coeficiente de atrito viscoso
        gamma = 1.4                         # Relação entre os calores específicos do ar
        g    = 9.8                          # Força Gravitacional

        '''
        - 25mm - embolo diamentro
        - 10mm - haste diametro

        estados
            x0 = posição
            x1 = velocidade
            x2 = Pressão em U - Pa ou Pda (montante)
            x3 = Pressão em B - Pb ou Pda (montante)
        
            Ps   - Source Pressure
            Patm - Atmosphere Pressure
            Pu   - Upper Chamber Pressure
            Pb   - Bottom Chamber Pressure
            Pc   - Chamber Pressure

        '''
        u = args[0][0]
        k = self.unknown_const
        # Mass Flow equation
        def dm(Ao,Ps,Pc,gamma,R,T):

            # debug
            print(f't:{t.item()} \t x:{x.numpy()} \t\t\t u: {u}')

            if Ps >= Pc:
                P1 = Ps
                P2 = Pc
            else:
                P1 = Pc
                P2 = Ps

            if P1 == 0:
                psi = 0
            elif P2/P1 > 0.528:     # subsonic flow
                psi = np.sqrt(gamma/(gamma-1)*(
                        (P2/P1)**(2/gamma) - (P2/P1)**((gamma+1)/gamma) 
                        )
                )
            else:                   # cloked flow
                psi = (2./(gamma+1)**(1/(gamma-1))) * np.sqrt(
                    gamma/(gamma+1)
                )

            dot_m = Ao*psi*P1*np.sqrt(2./(R*T))
            return dot_m

        dx = torch.zeros(len(self.x0),)
        # state space
        # mudar dy
        dx[0] = x[1]
        dx[1] = ((A*x[2] - kv*x[1])/M)-g
        dx[2] = 1/(Vb0+A*x[0])*(-A*gamma*x[1]*x[2]     + R*gamma*T*dm(Ao, (Ps*u+Patm),  x[2].item(), gamma, R, T))
        dx[3] = 1/(Vu0+A*(L-x[0]))*( A*gamma*x[1]*x[3] - R*gamma*T*dm(Ao, Patm,   x[3].item(), gamma, R, T))

        return dx
# rever L relacao deslocamento em dx3

def main():
    # load data
    data = loadmat('./data/teste0902_02.mat')
    y0 = data['desl']/1000
    u = data['atuador']
    Patm = 0
    pu = data['press'] * 1e5 + Patm
    t = data['t'].reshape(-1,1)
    ts = (t[1]-t[0])[0]
    fs = 1/(ts)
    fs = int(fs)
    del data

    params = {'optmizer': {'lowBound': [0.5, 0.5,0.1],
                            'upBound': [10, 10, 2],
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
                                'x0': [0, 
                                       0,
                                       pu[0,0],
                                       Patm],
                                't': [0,10-ts,len(t)],
                                }
                }

    f_fit = EqSystem(params)
    k = torch.tensor([5.0,5.0,0.1],dtype=torch.float32)
    test  =  f_fit.simulation(k)

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