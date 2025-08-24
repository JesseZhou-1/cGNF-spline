#!/usr/bin/env python3
"""
End-to-end integration test for cGNF using the SplineNormalizer.
This script:
  1. Simulates a small causal dataset.
  2. Writes CSV and DAG files.
  3. Calls `process`, `train`, and `sim` with norm_type='spline'.
  4. Verifies that the model trains without error and produces output files.
"""

import os
import shutil
import numpy as np
import pandas as pd
import networkx as nx
from causalgraphicalmodels import CausalGraphicalModel

# Make sure the cGNF package is in your PYTHONPATH or installed in your venv
from cGNF import process, train, sim

def simulate_and_save(path, dataset_name, N=20000, seed=0):
    np.random.seed(seed)
    # Simulate discrete/categorical graph:
    #   C in {1,2,3}; A,M,Y binary; L in {1,2,3}
    # --- New discrete DGP (C: 3 levels, A/M/Y: binary, L: 3 levels) ---
    obs = N
    # C takes values {1,2,3} with specified probabilities
    prob_C = [0.3, 0.5, 0.2]
    C = np.random.choice([1, 2, 3], size=obs, p=prob_C)

    # Treatment A, binary, probability influenced by C
    prob_A = 0.3 + 0.1 * C  # yields {0.4, 0.5, 0.6}
    A = np.random.binomial(1, prob_A)

    # L is a 3-level categorical variable with probabilities depending on (C, A)
    def assign_L(C_val, A_val):
        if A_val == 1:
            if C_val == 1:
                return np.random.choice([1, 2, 3], p=[0.5, 0.3, 0.2])
            elif C_val == 2:
                return np.random.choice([1, 2, 3], p=[0.3, 0.5, 0.2])
            else:  # C_val == 3
                return np.random.choice([1, 2, 3], p=[0.2, 0.3, 0.5])
        else:  # A_val == 0
            return np.random.choice([1, 2, 3], p=[0.6, 0.2, 0.2])

    L = np.array([assign_L(c, a) for c, a in zip(C, A)])

    # Mediator M, binary via logistic link
    logit_M = -0.5 + 0.4 * A + 0.2 * C + 0.3 * L
    prob_M = 1 / (1 + np.exp(-logit_M))
    M = np.random.binomial(1, prob_M)

    # Outcome Y, binary via logistic link with interaction A*M
    logit_Y = -0.5 + 0.3 * A + 0.1 * C + 0.3 * M + 0.3 * A * M + 0.3 * L
    prob_Y = 1 / (1 + np.exp(-logit_Y))
    Y = np.random.binomial(1, prob_Y)

    df = pd.DataFrame({'C': C, 'A': A, 'L': L, 'M': M, 'Y': Y})
    csv_file = os.path.join(path, f"{dataset_name}.csv")
    df.to_csv(csv_file, index=False)
    print(f"Saved synthetic data to {csv_file}")

    # Build DAG and save adjacency matrix
    simDAG = CausalGraphicalModel(
        nodes=["C","A","L","M","Y"],
        edges=[
            ("C","A"),("C","L"),("C","M"),("C","Y"),
            ("A","L"),("A","M"),("A","Y"),
            ("L","M"),("L","Y"),
            ("M","Y")
        ]
    )
    adj = nx.to_pandas_adjacency(simDAG.dag, dtype=int)
    dag_file = os.path.join(path, f"{dataset_name}_DAG.csv")
    adj.to_csv(dag_file)
    print(f"Saved DAG adjacency to {dag_file}")

def main():
    base = "discrete_spline_run/"
    if os.path.isdir(base):
        shutil.rmtree(base)
    os.makedirs(base, exist_ok=True)

    dataset_name = "discrete"
    simulate_and_save(base, dataset_name, N=500, seed=123)

    # Preprocess
    process(
        path=base,
        dataset_name=dataset_name,
        dag_name=f"{dataset_name}_DAG",
        test_size=0.2,
        cat_var=["C", "A", "L", "M", "Y"],
        seed=123
    )
    print("Preprocessing complete.")

    # Train with SplineNormalizer for a few epochs
    model, data = train(
        path=base,
        dataset_name=dataset_name,
        model_name="discrete_model_spline",
        trn_batch_size=128,
        val_batch_size=2048,
        learning_rate=1e-4,
        seed=123,
        nb_epoch=50000,
        emb_net=[90, 80, 70, 60, 50],
        int_net=[50, 40, 30, 20, 10],
        norm_type="spline",
        num_bins=16,
        bound=5,
        spline_hidden_dims=(128, 64, 32, 16, 8),
        nb_estop=100,
        val_freq=1
    )
    print("Training with SplineNormalizer complete.")

    # Simulation / potential outcomes
    sim(
        path=base,
        dataset_name=dataset_name,
        model_name="discrete_model_spline",
        n_mce_samples=50000,
        inv_datafile_name="sim_discrete_spline",
        treatment="A",
        cat_list=[0,1],
        moderator=None,
        mediator=["L","M"],
        outcome="Y"
    )
    print(f"Simulation complete. Outputs in folder: {base}")

if __name__ == "__main__":
    main()