import random
import numpy as np
from utils import E, O
from .base import BaseModel

class BehavioralRouting(BaseModel):
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        super().__init__(N, B, C, P, R, L, Z, D, config)
        self.name = 'BehavioralRouting_Model'

    def route(self, x):
        K = self.config['K']
        w1 = self.config['w1']
        w2 = self.config['w2']
        epsilon = self.config['epsilon']
        
        x_arr = np.array(x)
        
        open_j = np.nonzero(x_arr)[0] 
        M = len(open_j)
        
        F = np.zeros((self.N, self.N), dtype=int)
        
        if M == 0 or np.sum(self.D) == 0:
            return F
            
        batt = np.full(M, float(self.B))
        local_D = np.array(self.D, dtype=int)
        
        chunk_size = np.maximum(1, np.ceil(local_D / K).astype(int))
        
        L_sub = np.array(self.L)[:, open_j]
        Z_col = np.array(self.Z)[:, None]
        
        valid_mask = L_sub <= Z_col
        V = np.sum(local_D[:, None] * valid_mask, axis=0)
        
        norm_dist_all = np.divide(L_sub, Z_col, out=np.zeros_like(L_sub, dtype=float), where=Z_col!=0)
        congestion_all = V / self.B if self.B > 0 else np.zeros(M)
        
        base_term1 = w1 * norm_dist_all
        base_coeff = norm_dist_all * congestion_all

        while True:
            active_i = np.nonzero(local_D > 0)[0] 
            if len(active_i) == 0:
                break
                
            delta_req_active = np.minimum(local_D[active_i], chunk_size[active_i])
            depletion = self.B / (batt + epsilon)
            
            Loss = base_term1[active_i, :] + w2 * np.log1p(base_coeff[active_i, :] * depletion)
            
            L_active = L_sub[active_i, :]
            Z_active = Z_col[active_i, :]
            invalid_mask = (L_active > Z_active) | (batt == 0)
            Loss[invalid_mask] = np.inf
            
            min_loss = np.min(Loss, axis=1)
            best_rel_j = np.argmin(Loss, axis=1)
            
            valid_assignments = min_loss != np.inf
            if not np.any(valid_assignments):
                break
                
            assigned_i = active_i[valid_assignments]
            assigned_rel_j = best_rel_j[valid_assignments]
            reqs = delta_req_active[valid_assignments]
            
            requested_amount = np.zeros(M, dtype=int)
            np.add.at(requested_amount, assigned_rel_j, reqs)
            stations_with_reqs = np.nonzero(requested_amount > 0)[0]
            
            for rel_j in stations_with_reqs:
                abs_j = open_j[rel_j]
                req_j = requested_amount[rel_j]
                avail_batt = batt[rel_j]
                
                mask_j = (assigned_rel_j == rel_j)
                cust_i_for_j = assigned_i[mask_j]
                req_for_j = reqs[mask_j]
                
                if req_j <= avail_batt:
                    F[cust_i_for_j, abs_j] += req_for_j
                    local_D[cust_i_for_j] -= req_for_j
                    batt[rel_j] -= req_j
                else:
                    ratio = avail_batt / req_j
                    allocated = (req_for_j * ratio).astype(int)
                    
                    F[cust_i_for_j, abs_j] += allocated
                    local_D[cust_i_for_j] -= allocated
                    
                    leftover = int(avail_batt - np.sum(allocated))
                    batt[rel_j] = 0
                    
                    if leftover > 0:
                        dist_to_j = np.array(self.L)[cust_i_for_j, abs_j]
                        sort_idx = np.argsort(dist_to_j)
                        
                        sorted_cust_i = cust_i_for_j[sort_idx]
                        sorted_req = req_for_j[sort_idx]
                        sorted_alloc = allocated[sort_idx]
                        
                        for idx_c in range(len(sorted_cust_i)):
                            if leftover == 0:
                                break
                            c_id = sorted_cust_i[idx_c]
                            if sorted_req[idx_c] > sorted_alloc[idx_c] and local_D[c_id] > 0:
                                F[c_id, abs_j] += 1
                                local_D[c_id] -= 1
                                leftover -= 1
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