import random
import numpy as np
from sklearn.cluster import KMeans
from utils import E, O
from .base import BaseModel

class ClusterRouting(BaseModel):
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        super().__init__(N, B, C, P, R, L, Z, D, config)
        self.name = 'Cluster_Centric_Greedy'

        self.precomputed = []
        for j in range(self.N):
            sorted_demands = np.argsort(self.L[:, j])
            valid_demands = [i for i in sorted_demands if self.L[i, j] <= self.Z[i]]
            self.precomputed.append(valid_demands)

        self.rng = random.Random(self.config['random_seed'])

        num_clusters = self.config.get('num_clusters', max(1, self.N // 5))
        num_clusters = min(num_clusters, self.N)

        kmeans = KMeans(n_clusters=num_clusters, random_state=self.config['random_seed'], n_init='auto')
        labels = kmeans.fit_predict(self.L)

        self.clusters = [[] for _ in range(num_clusters)]
        for i, label in enumerate(labels):
            self.clusters[label].append(i)
            
        self.clusters = [c for c in self.clusters if c]

        self.evaluation_orders = []
        for _ in range(self.config['num_shuffles']):
            shuffled_clusters = list(self.clusters)
            self.rng.shuffle(shuffled_clusters)
            
            order = []
            for c in shuffled_clusters:
                c_copy = list(c)
                self.rng.shuffle(c_copy)
                order.extend(c_copy)
            self.evaluation_orders.append(order)

    def route(self, x, eval_order=None):
        station_battery = {j: self.B for j in range(self.N) if x[j] == 1}
        F = np.zeros((self.N, self.N))
        local_D = self.D.copy()

        if eval_order is None:
            eval_order = list(range(self.N))
            self.rng.shuffle(eval_order)

        for j in eval_order:
            if x[j] == 1:
                for i in self.precomputed[j]:
                    if local_D[i] > 0 and station_battery[j] > 0:
                        served = min(local_D[i], station_battery[j])
                        F[i, j] += served
                        local_D[i] -= served
                        station_battery[j] -= served
                    if station_battery[j] == 0:
                        break
        return F

    def fitness(self, x):
        fitness_vals = np.zeros(len(self.evaluation_orders))
        for idx, order in enumerate(self.evaluation_orders):
            F = self.route(x, order)
            total_E, _, _ = E(x, F, self.C, self.P, self.R)
            total_O, _, _ = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
            fitness_vals[idx] = self.config['lambda'] * total_E - self.config.get('mu', 1.0) * total_O
        return np.mean(fitness_vals)

    def get_details(self, x):
        e_vals, rev_vals, cost_vals, o_vals, unmet_vals, dist_vals = [], [], [], [], [], []
        for order in self.evaluation_orders:
            F = self.route(x, order)
            total_E, rev, cost = E(x, F, self.C, self.P, self.R)
            total_O, unmet, dist = O(F, self.D, self.L, self.config['alpha'], self.config['beta'])
            
            e_vals.append(total_E); rev_vals.append(rev); cost_vals.append(cost)
            o_vals.append(total_O); unmet_vals.append(unmet); dist_vals.append(dist)
            
        return {
            'avg_E': np.mean(e_vals), 'avg_rev': np.mean(rev_vals), 'avg_cost': np.mean(cost_vals),
            'avg_O': np.mean(o_vals), 'avg_unmet': np.mean(unmet_vals), 'avg_dist': np.mean(dist_vals)
        }