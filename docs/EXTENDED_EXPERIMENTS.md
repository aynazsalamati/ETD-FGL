# Extended ETD-FGL Experiments

This document describes the five additional studies used to strengthen the experimental evaluation.

## 1. Multi-seed evaluation

Seeds: 42, 43, 44.

The main comparison is repeated for all implemented methods. For sampled Reddit, the 8,000-node sample is held fixed with `reddit_sample_seed=42`; the training and client-partition seed changes across runs. This avoids conflating model stochasticity with a different Reddit sample.

Final metrics are aggregated as mean ± sample standard deviation (`ddof=1`).

## 2. Non-IID Dirichlet evaluation

Dirichlet alpha values: 0.1, 0.5, 1.0.

Partitioning is label-skewed and split-aware: train, validation, test, and unused nodes remain identifiable, while class membership is distributed across clients according to independent Dirichlet draws. Every node is assigned exactly once. A minimal repair procedure guarantees at least one training node per client and at least one non-target training node for the designated malicious client when mathematically possible.

The default comparison includes Attacked FedAvg, PD-FL, FLPurifier-GNN, and ETD-FGL.

## 3. Real component ablation

The following variants are trained from scratch under the same attacked setting:

- Full ETD-FGL: topology 0.45, explanation 0.35, update anomaly 0.20.
- ETD-FGL w/o Topology: topology evidence removed and remaining weights renormalized.
- ETD-FGL w/o Explanation: explanation evidence removed and remaining weights renormalized.
- ETD-FGL w/o Update Anomaly: round-wise update evidence removed and remaining weights renormalized.
- ETD-FGL w/o Trust Weighting: evidence can be computed, but aggregation falls back to sample-size weighting, isolating the effect of trust-weighted aggregation.

The ablation does not use synthetic or hand-entered values.

## 4. Sensitivity analysis

Two independent one-factor-at-a-time studies are included:

- Poisoning rate: 0.05, 0.10, 0.20, 0.30 with trigger size fixed at 3.
- Trigger size: 2, 3, 4, 5 with poisoning rate fixed at 0.20.

Attacked FedAvg and ETD-FGL are retrained for every condition. Clean test accuracy, ASR, trigger effect, and runtime are recorded.

## 5. Number-of-clients scalability

Client counts: 5, 10, 20, 30, 50.

The IID partitioner uses rotating stratified assignment so that public Planetoid training nodes are spread across high client counts without producing empty training clients. The study records clean accuracy, ASR, trigger effect, and runtime.

The default comparison includes Attacked FedAvg, PD-FL, FLPurifier-GNN, and ETD-FGL. This can be changed through `--methods`.

## Reproducibility and fairness

- Each compared method resets the RNG to the condition seed before training.
- All methods within a condition share the same dataset, partition, attack configuration, number of rounds, local epochs, model architecture, and optimization settings unless the method itself requires a specific training procedure.
- Existing completed conditions are reused unless `--force` is supplied.
- Figure generation reads only completed `final_summary.json` files and never fills missing runs with estimated values.
