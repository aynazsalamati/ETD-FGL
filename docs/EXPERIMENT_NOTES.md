# Experiment and reporting notes

## Baseline status

The repository contains adaptations of PD-FL, GSP, FLPurifier, and DMGNN concepts to a federated GCN experiment. These implementations are not claimed to be official code releases from the cited authors. Report them explicitly as adaptations.

## Two experiment families

The repository intentionally retains two workflows:

1. The original Cora workflow under `outputs/federated_cora`, with the longer Cora-specific training and trust-analysis setup.
2. The unified benchmark under `outputs/final_multidataset`, which applies one coordinated configuration to Cora, PubMed, and Reddit.

Do not mix their numerical results without stating the different configurations.

## Reddit sampling

The unified Reddit loader uses a deterministic stratified sample configured by `reddit_sample_size` in `multidata_suite/config.py`. The default is 8,000 nodes. Results must be described as sampled-Reddit results unless the configuration is changed to use the complete dataset.

## Seeds and uncertainty

The default unified configuration uses seed 42. For publication-grade reporting, rerun each method with multiple seeds and report mean, standard deviation, and preferably confidence intervals or error bars.

## Generated figures

Plotting scripts read experiment histories and summaries. They do not invent missing curves. A missing input should be treated as an incomplete experiment, not replaced with interpolated or synthetic values.

## Retained development variant

`multidata_suite/defenses_before_consensus_gate.py` is retained as a historical/development variant for provenance and comparison. The active unified pipeline imports `multidata_suite/defenses.py`; do not treat the retained variant as the default implementation.
