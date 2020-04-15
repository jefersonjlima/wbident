class Particle:
    def __init__(self, params):
        self._params = params
        self.particles_initializer()
        
    def particles_initializer(self)->None:
        self.nVar = self._params['nVar']
        self.nPop = self._params['nPop']
        self.cognitive_weight = self._params['cognitive_weight']
        self.social_weight = self._params['social_weight']
        self.w = self._params['w']
        self.w_damping = self._params['w_damping']
        self.minVelocity = self._params['minVelocity']
        self.maxVelocity = self._params['maxVelocity']
        # assert((self._params['upBound'] - self._params['lowBound']) > 0), \
        #                                             'Bound limits error!'
        self.lowBound = torch.ones(self.nPop,self.nVar) * torch.tensor(self._params['lowBound'])
        self.upBound = torch.ones(self.nPop,self.nVar) * torch.tensor(self._params['upBound'])

        # import pdb; pdb.set_trace() 
        # TODO move to func
        self.position_ = (self.upBound - self.lowBound)* \
                                torch.rand(self.nPop,self.nVar) + self.lowBound

        self.velocity_ = torch.zeros(self.nPop, self.nVar)
        # TODO not necessary
        self.best_cognitive_pos_ = (self.upBound - self.lowBound)* \
                                torch.rand(self.nPop,self.nVar) + self.lowBound

        self.best_social_pos_ = (self.upBound - self.lowBound)* \
                                torch.rand(self.nPop,self.nVar) + self.lowBound

    def limits(self, values, state = None):
        # TODO 
        if state == 'velocity':
            values[values < self.minVelocity] = self.minVelocity
            values[values > self.maxVelocity] = self.maxVelocity
        elif state == 'position':
            values[values < self.lowBound] = self.lowBound[values < self.lowBound]
            values[values > self.upBound] = self.upBound[values > self.upBound]
        return values
    
    def update(self):
        # update velocity
        up_vel = self.w * self.velocity_  \
            + self.cognitive_weight*(self.best_cognitive_pos_ - self.position_)*torch.rand((self.nPop, self.nVar)) \
            + self.social_weight* (self.best_social_pos_ - self.position_)*torch.rand((self.nPop, self.nVar))
        self.velocity_ = self.limits(up_vel, state='velocity')
        # update position
        up_pos = self.position_ + self.velocity_
        self.position_ = self.limits(up_pos, state='position')


class PSO(Particle):
    def __init__(self, EqSystem, params={}):
        super(PSO, self).__init__(params)
        self.params = params
        self._fitness = EqSystem
        self.social_cost_ = torch.empty(self.nPop, 1)
        self.cost()
        self.best_social_cost_ = self.social_cost_                               
        self.global_cost = float('inf')

    def cost(self):
        # multitheading
        for i in range(self.nPop):
            self.social_cost_[i] = self._fitness.evaluate(self.position_[i,:])

    def cost_update(self):
        # update social 
        # import pdb; pdb.set_trace() 
        for i in range(self.nPop):
            if (self.social_cost_[i] < self.best_social_cost_[i]):
                self.best_social_cost_[i] = self.social_cost_[i]
                self.best_social_pos_[i,:] = self.position_[i,:]
        # update global
            if (self.best_social_cost_[i] < self.global_cost):
                self.global_cost = self.best_social_cost_[i]
                self.best_cognitive_pos_[i,:] = self.best_social_pos_[i,:] 

    def run(self):
        self.cost()
        self.cost_update()
