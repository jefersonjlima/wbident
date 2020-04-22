import torch

class Model:
    def __init__(self, params=None):
        if params is None:
            parameters = {}
        self._params = params
        self._unknown_const = None
        self._path = None
        self.x0 = None
        self.data = None
        self.y_pred = None
        self.y_true = None
        self.parameters_initializer()

    def parameters_initializer(self):
        self._params = self._params['dyn_system']
        self._path = self._params['model_path']
        self.x0 = torch.tensor(self._params['x0'])
        t = self._params['t']
        t = torch.linspace(t[0], t[1], t[2])
        self.t = t.reshape(-1,1)

    @property
    def unknown_const(self):
        return self._unknown_const

    @unknown_const.setter
    def unknown_const(self, value):
        self._unknown_const = value

    def simulation(self, k):
        self.unknown_const = k
        return self.ode45(self.model, self.t, self.x0)
        
    def evaluate(self, k):
        self.unknown_const = k
        y_hat = self.ode45(self.model, self.t, self.x0)
        mse = (self.y-y_hat).pow(2).mean()
        return mse, self.y, y_hat

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
        x = torch.zeros((n,len(x0)))
        x[0,:] = x0
        for i in range(n - 1):
            dt = t[i + 1] - t[i]
            x[i + 1,:] = ode45_step(f, x[i,:],  t[i], dt, *args)
        return x
