class BaseModel:
    """!
    @brief The abstract base class for all EV station routing models.
    
    @details This class defines the standard interface and shared state initialization 
    for all routing algorithms in the project. It stores the core network matrices 
    and configuration settings so that inheriting classes do not need to duplicate 
    this boilerplate code. It enforces a strict contract by requiring child classes 
    to implement the `route`, `fitness`, and `get_details` methods.
    
    @par Class Connections:
    - Acts as the parent class for specific models like `AlternatingRouting`, 
      `CustomerRouting`, and `StationRouting`.
    - Instantiated dynamically by `Experiment.__init__` based on the provided configuration.
    """
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        """!
        @brief Initializes the base model with network parameters and configuration.
        
        @param N Number of nodes in the network.
        @param B Battery capacity per station.
        @param C Base cost vector for building stations.
        @param P Profit margin per unit of demand served.
        @param R Recurring cost vector per active station.
        @param L Distance matrix between all nodes.
        @param Z Distance tolerance vector for customers.
        @param D Initial demand vector for all nodes.
        @param config The experiment configuration dictionary.
        """
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
