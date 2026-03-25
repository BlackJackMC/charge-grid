import numpy as np
import heapq
from utils import E, O
from .base import BaseModel

class AlternatingRouting(BaseModel):
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        super().__init__(N, B, C, P, R, L, Z, D, config)
        self.name = 'Alternating_Routing'

        self.customer_pref = []
        for i in range(self.N):
            sorted_stations = np.argsort(self.L[i, :])
            valid_stations = [j for j in sorted_stations if self.L[i, j] <= self.Z[i]]
            self.customer_pref.append(valid_stations)

    def route(self, x, eval_order=None):
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
        F = self.route(x)
        total_E, _, _ = E(x, F, self.C, self.P, self.R)
        total_O, _, _ = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
        return self.config['lambda'] * total_E - self.config.get('mu', 1.0) * total_O

    def get_details(self, x):
        F = self.route(x)
        total_E, rev, cost = E(x, F, self.C, self.P, self.R)
        total_O, unmet, dist = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
        
        return {
            'avg_E': total_E, 'avg_rev': rev, 'avg_cost': cost,
            'avg_O': total_O, 'avg_unmet': unmet, 'avg_dist': dist
        }