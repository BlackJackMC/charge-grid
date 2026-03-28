import numpy as np

# Read data
with open("input_q1.txt") as f:
    N, B, C, P = map(float, f.readline().split())
    N = int(N); B = int(B)

    L = np.array([list(map(float, f.readline().split())) for _ in range(N)])
    R = np.array(list(map(float, f.readline().split())))
    Z = np.array(list(map(float, f.readline().split())))
    D = np.array(list(map(float, f.readline().split())))

# Track
opened = np.zeros(N)
assigned = -np.ones(N)  # which station serves j

remaining_demand = D.copy()

while True:
    best_i = -1
    best_profit = 0

    for i in range(N): 
        if opened[i] == 1:
            continue

        feasible = [j for j in range(N)
                    if remaining_demand[j] > 0 and L[i][j] <= Z[j]]

        if not feasible:
            continue

        total_demand = sum(remaining_demand[j] for j in feasible)
        profit = total_demand * P - (C + R[i])

        if profit > best_profit:
            best_profit = profit
            best_i = i

    if best_i == -1:
        break

    # Open station
    opened[best_i] = 1

    capacity = B

    # Assign customers
    for j in sorted(range(N), key=lambda x: L[best_i][x]):
        if remaining_demand[j] > 0 and L[best_i][j] <= Z[j]:
            take = min(capacity, remaining_demand[j])
            assigned[j] = best_i
            capacity -= take
            remaining_demand[j] -= take

        if capacity <= 0:
            break

# Output x_i
for i in range(N):
    print(int(opened[i]), end=" ")