class BaseModel:
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        self.name = 'Base_Model'
        self.N = N
        self.B = B
        self.C = C
        self.P = P
        self.R = R
        self.L = L
        self.Z = Z
        self.D = D
        self.config = config

    def route(self, x, *args, **kwargs):
        raise NotImplementedError

    def fitness(self, x):
        raise NotImplementedError

    def get_details(self, x):
        raise NotImplementedError
