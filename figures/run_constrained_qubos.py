import csv
import numpy as np
import json
import os
from tqdm import tqdm
from utils import QUBO, generate_problem, create_contrained_qubo, generate_constraint

def original_is_max_anywhere(vectors):
    """
    vectors: array-like of shape (n, m), where vectors[0] is the unconstrained vector.
    Returns True if ∃ an index s with vectors[0, s] >= vectors[i, s] ∀ i.
    """
    arr = np.asarray(vectors)
    col_max = arr.max(axis=0) # takes the maximum over all objectives
    return np.any(arr[0] == col_max) # if any maximum was from the first vector this means that the original problem was the minimum at that s -> it was allowed

def append_performance_csv(
    csv_path: str,
    n_samples: int,
    size_list: list[int],
    p_max: int,
    gamma: int,
    tau_scale: float,
    num_constraints: int,
    seed_offset: int = 10001
):

    seen = set()
    file_exists = os.path.exists(csv_path)
    if file_exists:
        with open(csv_path, newline='') as f:
            reader = csv.DictReader(f)
            for row in reader:
                key = (
                    int(row['seed']),
                    int(row['size']),
                )
                seen.add(key)

    mode = 'a' if file_exists else 'w'
    with open(csv_path, mode, newline='') as f:
        writer = csv.DictWriter(
            f,
            fieldnames=['seed', 'size', 'wrong_idx', 'energy_diff', 'allowed', 'possible', 'ratio_values', 'ratio_values_p']
        )
        if not file_exists:
            writer.writeheader()

        for size in size_list:
            desc = f"size={size}"
            for n in tqdm(range(n_samples), desc=desc):
                seed = seed_offset + n
                key = (seed, size)
                if key in seen:
                    continue

                A, a, alpha = generate_problem(size=size, seed=seed + n + size*n_samples)
                constraint_list = []
                for num_constr in range(num_constraints):
                    constraint = generate_constraint(size=size, gamma=gamma, tau_scale=tau_scale, seed=seed + num_constr + 10*size*n_samples)
                    constraint_list.append(constraint)

                m_qubos = create_contrained_qubo(QUBO(A, a, alpha), constraint_list)
                m_qubos.calculate_h_p(p_max=p_max)

                wrong_idx = [
                    0 if (m_qubos.h_max[0][t[2]] - m_qubos.h_max[1]) == 0 else 1
                    for t in m_qubos.h_p_list
                ]
                energy_diff = [
                    (m_qubos.h_max[0][t[2]] - m_qubos.h_max[1]) / m_qubos.h_max[1]
                    for t in m_qubos.h_p_list
                ]
                allowed = [
                    0 if any(m_qubos.qubos_list[0].H[t[2]] - q.H[t[2]] > 0 for q in m_qubos.qubos_list[1:]) else 1
                    for t in m_qubos.h_p_list
                ]

                possible = original_is_max_anywhere(
                    [q.H for q in m_qubos.qubos_list]
                )
                ratio_values = m_qubos.h_max[4]
                ratio_values_p = [t[4] for t in m_qubos.h_p_list]

                writer.writerow({
                    'seed': seed,
                    'size': size,
                    'wrong_idx': json.dumps(wrong_idx),
                    'energy_diff': json.dumps(energy_diff),
                    'allowed': json.dumps(allowed),
                    'possible': possible,
                    'ratio_values': json.dumps(ratio_values),
                    'ratio_values_p': json.dumps(ratio_values_p),
                })
                seen.add(key)  # avoid duplicates within one run
                if n % 100 == 0:
                    f.flush()
                    os.fsync(f.fileno())


if __name__ == "__main__":
    N_SAMPLES = 10000
    SIZE_LIST = [4, 8, 12, 16, 20]
    P_MAX = 21 # should be max(SIZE_LIST)
    seed_offset = 10001
    tau = 0
    gamma = 10
    num_constraints = 4
    csv_path = f"data/performance_nconstr{num_constraints}_tau{tau}_gamma{gamma}.csv"
    append_performance_csv(
        csv_path=csv_path,
        n_samples=N_SAMPLES,
        size_list=SIZE_LIST,
        p_max=P_MAX,
        gamma=gamma,
        tau_scale=tau,
        num_constraints=num_constraints,
        seed_offset=seed_offset
    )
    print(f"Appended new runs to {csv_path}")