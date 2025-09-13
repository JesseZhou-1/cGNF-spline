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
    
    # Confounder with Laplace distribution error
    loc_c, scale_c = 0, 1
    C = np.random.laplace(loc_c, scale_c, N)

    # Treatment, binary, derived from logistic transformation of a normal variable
    loc_a, scale_a = 0, 1
    A = (np.random.logistic(loc_a, scale_a, N) + 0.1 * C) > 0  # binary treatment
    A = A.astype(int)

    def simulate_generalized_lambda_distribution_4p(mu=0, sigma=1, lam=0, gamma=0, size=1):
        u = np.random.uniform(0, 1, size)
        return mu + (u ** lam - (1 - u) ** gamma) / sigma

    # Example usage
    loc_l = 0
    scale_l = 1
    lam_l = 0.3
    gamma_l = 0.7

    # Exposure-induced confounder with Tukey Lambda distribution error
    L = simulate_generalized_lambda_distribution_4p(mu=loc_l, sigma=scale_l, lam=lam_l, gamma=gamma_l,
                                                    size=N) + 0.2 * A + 0.2 * C + 0.1 * A * C

    # Mediator with Cauchy distribution error, interaction, polynomial term, and correlated errors
    M = np.random.standard_t(10, N) + 0.1 * A + 0.2 * np.square(C) + 0.25 * L + 0.15 * A * L

    # Outcome with heteroscedastic error and polynomial terms
    loc_y = 0
    Y = np.random.normal(loc_y, np.abs(C), N) + 0.1 * A + 0.1 * np.square(
        C) + 0.2 * M + 0.2 * A * M + 0.25 * np.square(L)
    
    df = pd.DataFrame({'C':C, 'A':A, 'L':L, 'M':M, 'Y':Y})
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
    base = "mixed_spline_run/"
    if os.path.isdir(base):
        shutil.rmtree(base)
    os.makedirs(base, exist_ok=True)

    dataset_name = "mixed"
    simulate_and_save(base, dataset_name, N=500, seed=123)

    # Preprocess
    process(
        path=base,
        dataset_name=dataset_name,
        dag_name=f"{dataset_name}_DAG",
        test_size=0.2,
        cat_var=['A'],  # C is continuous under the new DGP
        seed=123
    )
    print("Preprocessing complete.")

    # Train with SplineNormalizer for a few epochs
    train(
        path=base,
        dataset_name=dataset_name,
        model_name="mixed_model_spline",
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
        model_name="mixed_model_spline",
        n_mce_samples=50000,
        inv_datafile_name="sim_mixed_spline",
        treatment="A",
        cat_list=[0,1],
        moderator=None,
        mediator=["L","M"],
        outcome="Y"
    )
    print(f"Simulation complete. Outputs in folder: {base}")

if __name__ == "__main__":
    main()