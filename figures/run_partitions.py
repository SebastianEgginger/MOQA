import csv
import numpy as np
import json
import os
from tqdm import tqdm
from utils import QUBO, generate_random_graph, generate_distance_matrix, create_t_pm, M_QUBOs

def append_performance_csv(
    csv_path: str,
    n_samples: int,
    size_list: list[int],
    p_max: int,
    sample_mode = 'gaussian',
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
            fieldnames=['seed', 'size', 'wrong_idx', 'energy_diff', 'ratio_values', 'ratio_values_p']
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

                points, vertices = generate_random_graph(n=size, mode=sample_mode, seed=seed)
                W, v = generate_distance_matrix(points, vertices=vertices)
                A, a, alpha = create_t_pm(W, v)

                q_list = [QUBO(A, a, alpha), QUBO(A, -a, alpha)]
                m_qubos = M_QUBOs(q_list)
                m_qubos.calculate_h_p(p_max=p_max)

                wrong_idx = [
                    0 if (m_qubos.h_max[0][t[2]] - m_qubos.h_max[1]) == 0 else 1
                    for t in m_qubos.h_p_list
                ]
                energy_diff = [
                    (m_qubos.h_max[0][t[2]] - m_qubos.h_max[1]) / m_qubos.h_max[1]
                    for t in m_qubos.h_p_list
                ]
                ratio_values = m_qubos.h_max[4]
                ratio_values_p = [t[4] for t in m_qubos.h_p_list]

                writer.writerow({
                    'seed': seed,
                    'size': size,
                    'wrong_idx': json.dumps(wrong_idx),
                    'energy_diff': json.dumps(energy_diff),
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
    mode = 'gaussian'  # 'gaussian' or 'uniform'
    csv_path = f"data/performance_partitions_{mode}.csv"
    append_performance_csv(
        csv_path=csv_path,
        n_samples=N_SAMPLES,
        size_list=SIZE_LIST,
        p_max=P_MAX,
        sample_mode=mode,
        seed_offset=seed_offset
    )
    print(f"Appended new runs to {csv_path}")