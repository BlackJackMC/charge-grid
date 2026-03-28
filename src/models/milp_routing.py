import numpy as np
from ortools.linear_solver import pywraplp

from utils import E, O
from .base import BaseModel


class MILPRoutingORTools(BaseModel):
    def __init__(self, N, B, C, P, R, L, Z, D, config):
        super().__init__(N, B, C, P, R, L, Z, D, config)
        self.name = "MILP_Routing_ORTools"

        # Precompute valid edges (i -> j reachable)
        self.valid_pairs = [
            (i, j)
            for i in range(self.N)
            for j in range(self.N)
            if self.L[i][j] <= self.Z[i]
        ]

    # --------------------
    # SOLVER CORE
    # --------------------
    def solve_flow(self, x):
        solver = pywraplp.Solver.CreateSolver("CBC")
        if not solver:
            raise RuntimeError("CBC solver not available.")

        INF = solver.infinity()

        # --------------------
        # VARIABLES
        # --------------------
        # Flow variables
        F = {
            (i, j): solver.NumVar(0, INF, f"F_{i}_{j}")
            for (i, j) in self.valid_pairs
        }

        # Unmet demand (slack)
        U = {
            i: solver.NumVar(0, INF, f"U_{i}")
            for i in range(self.N)
        }

        # --------------------
        # OBJECTIVE
        # --------------------
        objective = solver.Objective()

        for (i, j) in self.valid_pairs:
            coef = (
                self.P
                - self.config["beta"] * self.L[i][j]
            )
            objective.SetCoefficient(F[(i, j)], coef)

        # Penalize unmet demand
        alpha = self.config["alpha"]
        for i in range(self.N):
            objective.SetCoefficient(U[i], -alpha)

        objective.SetMaximization()

        # --------------------
        # CONSTRAINTS
        # --------------------

        # 1. Demand satisfaction (flow + unmet = demand)
        for i in range(self.N):
            ct = solver.Constraint(float(self.D[i]), float(self.D[i]))
            for (ii, j) in self.valid_pairs:
                if ii == i:
                    ct.SetCoefficient(F[(ii, j)], 1)
            ct.SetCoefficient(U[i], 1)

        # 2. Capacity constraints at stations
        for j in range(self.N):
            capacity = float(self.B if x[j] == 1 else 0)
            ct = solver.Constraint(0.0, capacity)

            for (i, jj) in self.valid_pairs:
                if jj == j:
                    ct.SetCoefficient(F[(i, jj)], 1)

        # --------------------
        # SOLVE
        # --------------------
        solver.SetTimeLimit(self.config.get("milp_time_limit", 3000))
        status = solver.Solve()

        # --------------------
        # EXTRACT SOLUTION
        # --------------------
        F_mat = np.zeros((self.N, self.N))
        U_vec = np.zeros(self.N)

        if status in (
            pywraplp.Solver.OPTIMAL,
            pywraplp.Solver.FEASIBLE,
        ):
            for (i, j) in self.valid_pairs:
                F_mat[i, j] = F[(i, j)].solution_value()

            for i in range(self.N):
                U_vec[i] = U[i].solution_value()

        return F_mat, U_vec

    # --------------------
    # FITNESS
    # --------------------
    def fitness(self, x):
        if sum(x) == 0:
            return -1e12

        F, U = self.solve_flow(x)

        total_E, _, _ = E(x, F, self.C, self.P, self.R)
        total_O, _, _ = O(
            F,
            self.D,
            self.L,
            self.config["alpha"],
            self.config["beta"],
        )

        return (
            self.config["lambda"] * total_E
            - self.config.get("mu", 1.0) * total_O
        )

    # --------------------
    # DEBUG / ANALYSIS
    # --------------------
    def get_details(self, x):
        F, U = self.solve_flow(x)

        total_E, rev, cost = E(x, F, self.C, self.P, self.R)
        total_O, unmet, dist = O(
            F,
            self.D,
            self.L,
            self.config["alpha"],
            self.config["beta"],
        )

        return {
            "avg_E": total_E,
            "avg_rev": rev,
            "avg_cost": cost,
            "avg_O": total_O,
            "avg_unmet": unmet,
            "avg_dist": dist,
            "unmet_vector": U,
        }