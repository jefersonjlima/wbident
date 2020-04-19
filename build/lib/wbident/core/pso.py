import torch


class Particle:
    def __init__(self, params):
        self._params = params['optmizer']
        self.nVar = self._params['nVar']
        self.nPop = self._params['nPop']
        self.cognitive_weight = self._params['cognitive_weight']
        self.social_weight = self._params['social_weight']
        self.w = self._params['w']
        self.w_damping = self._params['w_damping']
        self.minVelocity = self._params['minVelocity']
        self.maxVelocity = self._params['maxVelocity']
        self.lowBound = torch.empty(self.nPop, self.nVar)
        self.upBound = torch.empty(self.nPop, self.nVar)
        self.p_position_ = torch.empty(self.nPop, self.nVar)
        self.p_velocity_ = torch.empty(self.nPop, self.nVar)
        self.pb_position_ = torch.empty(self.nPop, self.nVar)

        self.pbg_position = torch.empty(1, self.nVar)

        self.particles_initializer()

    def particles_initializer(self) -> None:
        self.lowBound = torch.ones(self.nPop, self.nVar) * torch.tensor(self._params['lowBound'])
        self.upBound = torch.ones(self.nPop, self.nVar) * torch.tensor(self._params['upBound'])
        self.p_position_ = (self.upBound - self.lowBound) * torch.rand(self.nPop, self.nVar) + self.lowBound
        self.p_velocity_ = torch.zeros(self.nPop, self.nVar)

    def limits(self, values, state=None):
        if state == 'velocity':
            values[values < self.minVelocity] = self.minVelocity
            values[values > self.maxVelocity] = self.maxVelocity
        elif state == 'position':
            values[values < self.lowBound] = self.lowBound[values < self.lowBound]
            values[values > self.upBound] = self.upBound[values > self.upBound]
        return values

    def update_particle(self):
        # update velocity
        up_vel = self.w * self.p_velocity_ \
                 + self.cognitive_weight * torch.rand(self.nPop, self.nVar) * (self.pb_position_ - self.p_position_) \
                 + self.social_weight * torch.rand(self.nPop,  self.nVar) * (self.pbg_position - self.p_position_)
        self.p_velocity_ = self.limits(up_vel, state='velocity')
        # update position
        up_pos = self.p_position_ + self.p_velocity_
        self.p_position_ = self.limits(up_pos, state='position')


class PSO(Particle):
    def __init__(self, eq_system, params=None):
        super(PSO, self).__init__(params)
        if params is None:
            params = {}
        self._params = params['optmizer']
        self._fitness = eq_system
        self.p_cost_ = torch.empty(self.nPop, 1)
        self.pb_cost_ = torch.empty(self.nPop, 1)
        self.pbg_cost = torch.empty(1)
        self.pso_initializer()

    def pso_initializer(self):
        self.pbg_cost = float('inf')
        for i in range(self.nPop):
            self.p_cost_[i], self.y_true, self.y_pred = self._fitness.evaluate(self.p_position_[i, :])
            self.pb_position_[i, :] = self.p_position_[i, :]
            self.pb_cost_[i] = self.p_cost_[i]
        # self.pbg_cost = self.pb_cost_.min()
        self.pbg_position = self.pb_position_[self.pb_cost_.argmin(), :]

    def update_cost(self):
        for i in range(self.nPop):
            self.p_cost_[i], self.y_true, self.y_pred = self._fitness.evaluate(self.p_position_[i, :])
            # print(f'{self.p_cost_},{self.pb_cost_},{self.pbg_cost}')
            if self.p_cost_[i] < self.pb_cost_[i]:
                # update best particle values
                self.pb_position_[i, :] = self.p_position_[i, :]
                self.pb_cost_[i] = self.p_cost_[i]
                # update best global particle values
            if self.pb_cost_[i] < self.pbg_cost:
                self.pbg_cost = self.pb_cost_[i]
                self.pbg_position = self.p_position_[i, :]
        self.w *= self.w_damping

    def run(self):
        self.update_particle()
        self.update_cost()
