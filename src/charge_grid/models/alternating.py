import numpy as np
import heapq

from charge_grid.utils import E, O
from .base import BaseModel

class AlternatingRouting(BaseModel):
    """!
    @brief A routing model that simulates demand allocation using an alternating proposal algorithm.
    
    @details This model assigns customer demand to EV stations using a priority-based, 
    multi-round matching process (similar to Gale-Shapley/stable matching). Customers 
    always attempt to visit their nearest available station. If multiple customers converge 
    on the same station, the station prioritizes fulfilling the closest customers first 
    using a priority queue (min-heap) until its battery capacity is depleted.
    
    @par Class Connections:
    - Inherits core properties and matrix definitions from `BaseModel`.
    - Instantiated dynamically in `Experiment.__init__` when 'AlternatingRouting' is 
      specified in the configuration dictionary.
    """
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        """!
        @brief Initializes the model and pre-computes static customer preferences.
        
        @details Calls the parent `BaseModel` constructor to load the network data. 
        To optimize the routing simulation, it pre-computes a strict preference list 
        for every customer node. It sorts all available stations by distance and filters 
        out any stations that exceed the customer's maximum walking tolerance (`Z`).
        
        @par Function Connections:
        - The `self.customer_pref` list generated here is consumed heavily by the 
          `route` method to avoid resorting distances during every GA evaluation.
          
        @param N, B, C, P, R, L, Z, D Core network parameters passed down to the BaseModel.
        @param config The experiment configuration dictionary.
        """
        super().__init__(N, B, C, P, R, L, Z, D, config)
        self.name = 'Alternating_Routing'

        self.customer_pref = []
        for i in range(self.N):
            sorted_stations = np.argsort(self.L[i, :])
            valid_stations = [j for j in sorted_stations if self.L[i, j] <= self.Z[i]]
            self.customer_pref.append(valid_stations)

    def route(self, x, eval_order=None):
        """!
        @brief Determines the flow of demand from customers to active stations.
        
        @details Executes the multi-round stable matching algorithm:
        1. **Proposal:** Active customers propose to the nearest active station with 
           remaining battery capacity, iterating through their pre-computed preference list.
        2. **Queuing:** Stations collect all incoming proposals for the round into a 
           min-heap, sorted by distance.
        3. **Fulfillment:** Stations pop customers from the heap (closest first) and 
           allocate battery capacity until depleted or the demand is met.
        4. **Iteration:** Customers with remaining unfulfilled demand are pushed to 
           the next round to propose to their next preferred station.
           
        @par Function Connections:
        - Relies on the `self.customer_pref` lists built in `__init__`.
        - Called internally by `self.fitness()` and `self.get_details()`.
        
        @param x A sequence of integers (1s and 0s) representing active station locations.
        @param eval_order Optional parameter (unused in this specific model, maintained 
                          for structural compatibility with `BaseModel` polymorphism).
                          
        @return F A 2D NumPy array representing the finalized flow of demand from 
                  customers (rows) to stations (columns).
        """
        station_battery = {j: self.B for j in range(self.N) if x[j] == 1}
        F = np.zeros((self.N, self.N))
        local_D = self.D.copy()

        cust_ptr = [0] * self.N
        active_custs = [i for i in range(self.N) if local_D[i] > 0]

        while active_custs:
            station_heaps = {j: [] for j in range(self.N) if x[j] == 1 and station_battery[j] > 0}
            next_active_custs = []

            for i in active_custs:
                found_station = False
                while cust_ptr[i] < len(self.customer_pref[i]):
                    j = self.customer_pref[i][cust_ptr[i]]
                    if x[j] == 1 and station_battery[j] > 0:
                        heapq.heappush(station_heaps[j], (self.L[i, j], i))
                        found_station = True
                        break
                    else:
                        cust_ptr[i] += 1
                
                if not found_station:
                    pass

            for j, heap in station_heaps.items():
                while heap:
                    dist, i = heapq.heappop(heap)
                    
                    if station_battery[j] > 0:
                        served = min(local_D[i], station_battery[j])
                        F[i, j] += served
                        local_D[i] -= served
                        station_battery[j] -= served
                        
                        if local_D[i] > 0:
                            cust_ptr[i] += 1
                            next_active_custs.append(i)
                    else:
                        cust_ptr[i] += 1
                        next_active_custs.append(i)

            active_custs = next_active_custs

        return F

    def fitness(self, x):
        """!
        @brief Calculates the overall optimization score for the current layout.
        
        @details Executes the routing simulation to determine the demand flow, then 
        evaluates the layout based on a weighted difference between economic profit 
        and operational dissatisfaction. The formula applied is: 
        (lambda * total_E) - (mu * total_O).
        
        @par Function Connections:
        - Calls `self.route(x)` to generate the flow matrix `F`.
        - Calls the global `E` and `O` functions from `utils.py` to compute raw metrics.
        - Called continuously by the PyGAD evaluation loop via `Experiment._fitness_handler_helper`.
        
        @param x A sequence of integers (1s and 0s) representing active station locations.
        
        @return A single float representing the finalized fitness score.
        """
        F = self.route(x)
        total_E, _, _ = E(x, F, self.C, self.P, self.R)
        total_O, _, _ = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
        return self.config['lambda'] * total_E - self.config.get('mu', 1.0) * total_O

    def get_details(self, x):
        """!
        @brief Extracts detailed performance metrics for the current layout.
        
        @details Runs the routing algorithm and breaks down the resulting economic 
        and operational scores into their granular components (revenue, costs, unmet 
        demand penalties, and distance penalties).
        
        @par Function Connections:
        - Calls `self.route(x)` to generate the flow matrix `F`.
        - Calls the global `E` and `O` functions from `utils.py`.
        - Invoked by `Experiment.log_handler` to record generation history and by 
          the interactive map script to display UI statistics.
        
        @param x A sequence of integers (1s and 0s) representing active station locations.
        
        @return A dictionary containing keys for 'avg_E', 'avg_rev', 'avg_cost', 
                'avg_O', 'avg_unmet', and 'avg_dist'.
        """
        F = self.route(x)
        total_E, rev, cost = E(x, F, self.C, self.P, self.R)
        total_O, unmet, dist = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
        
        return {
            'avg_E': total_E, 'avg_rev': rev, 'avg_cost': cost,
            'avg_O': total_O, 'avg_unmet': unmet, 'avg_dist': dist
        }