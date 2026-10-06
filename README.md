# ETD-FGL

**Federated Graph Learning Against Backdoor Attacks Using Topology-aware Defense**

**Author:** Aynaz Salamati

ETD-FGL is the research code accompanying a manuscript on structural backdoor attacks and topology-aware defense in federated graph learning. The repository contains the original Cora workflow, a unified Cora/PubMed/Reddit benchmark, baseline adaptations, trust-weighted aggregation, explanation/topology analysis, and journal-style figure generation.

> **Reproducibility note:** the repository separates generated datasets/results from source code. Dataset files are downloaded by PyTorch Geometric on first use and experiment outputs are written under `outputs/`, both of which are excluded from Git.

## Repository scope

### Datasets

- Cora
- PubMed
- Reddit

The unified Reddit experiment uses a deterministic, stratified 8,000-node subgraph to keep the benchmark practical on limited-memory GPUs. The sampling metadata is written to the generated experiment summaries.

### Compared methods

- Clean FedAvg
- Attacked FedAvg
- PD-FL adaptation
- GSP-FL adaptation
- FLPurifier-GNN adaptation
- DMGNN-FL adaptation
- ETD-FGL

The baseline implementations are research adaptations to the federated GCN setting. They must not be described as official author implementations unless replaced with the corresponding official repositories and configurations.

## Project structure

```text
ETD-FGL/
├── baseline_suite/              # Cora baseline components
├── journal_figures/             # Journal-style plotting helpers
├── multidata_suite/             # Unified Cora/PubMed/Reddit pipeline
├── docs/                        # Persian quick start and experiment notes
├── results/                     # Optional lightweight published artifacts
├── run_real_dataset_pipeline.py
├── check_environment.py         # Dependency/CUDA sanity check
├── build_real_comparisons_and_figures.py
├── run_all_real_datasets.bat
├── run_all_baselines.bat
├── train_*.py
├── test_*.py
├── build_*.py
├── requirements.txt
└── .gitignore
```

The generated `data/`, `outputs/`, checkpoints, caches, and virtual environment are intentionally excluded from Git.

## Environment setup

The project is intended for Python 3.12 on Windows. Install a PyTorch build compatible with the local CUDA driver first, then install the remaining dependencies:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
# Install the appropriate PyTorch build for the local CPU/CUDA environment.
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Alternatively, after creating `.venv` and installing PyTorch, run:

```text
install_requirements.bat
```

After installation, verify the environment without starting an experiment:

```powershell
.\.venv\Scripts\python.exe check_environment.py
```

On Linux/macOS, the equivalent setup is:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
# Install a CPU/CUDA-compatible PyTorch build first.
python -m pip install -r requirements.txt
python check_environment.py
```

## Reproducing the main experiments

For the coordinated Cora/PubMed/Reddit benchmark, the platform-independent Python entry points are:

```bash
python run_real_dataset_pipeline.py --dataset cora
python build_real_comparisons_and_figures.py --dataset cora

python run_real_dataset_pipeline.py --dataset pubmed
python build_real_comparisons_and_figures.py --dataset pubmed

python run_real_dataset_pipeline.py --dataset reddit
python build_real_comparisons_and_figures.py --dataset reddit

python build_real_comparisons_and_figures.py --dataset all
python build_section4_extra_figures.py
```

Windows users can run the equivalent wrapper `run_all_real_datasets.bat`. Existing summaries are reused unless `--force` is supplied to `run_real_dataset_pipeline.py`. For publication tables, report the exact configuration and seed(s) used, and distinguish sampled-Reddit results from full-graph Reddit results.

## Unified three-dataset benchmark

Run all real experiment pipelines and build the per-dataset and cross-dataset figures:

```text
run_all_real_datasets.bat
```

Run datasets separately:

```text
run_pubmed_real.bat
run_reddit_real.bat
```

The unified configuration is defined in `multidata_suite/config.py`. Its default settings are 5 clients, 50 communication rounds, 3 local epochs, seed 42, malicious client 0, poison rate 0.20, target class 0, and a 3-node clique trigger.

Generated outputs are written to:

```text
outputs/final_multidataset/federated_cora/
outputs/final_multidataset/federated_pubmed/
outputs/final_multidataset/federated_reddit/
outputs/final_multidataset/cross_dataset_figures/
```

For each method and dataset, the pipeline writes a training history, best checkpoint, and final summary. `build_real_comparisons_and_figures.py` produces comparison tables and multi-panel PNG/PDF/SVG figures from those real outputs.

## Original Cora workflow

The original Cora pipeline preserves the longer experimental setup and the existing topology/explanation/trust analysis scripts:

```text
run_all_baselines.bat
```

Its main output root is:

```text
outputs/federated_cora/
```

This runner reuses completed stages. The defended Cora stage expects the topology, explanation, and pilot trust outputs to exist when rebuilding from scratch. The supporting analysis scripts are retained at the repository root.

## Figure generation

Baseline comparison figures for the original Cora workflow:

```powershell
.\.venv\Scripts\python.exe build_all_baseline_journal_figures.py
```

ETD-FGL journal figures from available `outputs/federated_*` runs:

```text
run_all_journal_figures.bat
```

Unified multi-dataset figures are generated automatically by `build_real_comparisons_and_figures.py` after each dataset run.

Section 4 cross-dataset summary figures are generated by:

```text
run_section4_extra_figures.bat
```

The corresponding Python script is `build_section4_extra_figures.py`. It reads only the real `comparison.csv` files produced by the three-dataset pipeline and writes PNG/PDF/SVG versions of final clean accuracy, ASR, trigger effect, and runtime figures to:

```text
outputs/final_multidataset/section4_extra_figures/
```

## Reproducibility and reporting

- No synthetic result curves are generated by the plotting scripts.
- Figures are read from real `training_history.csv` and `final_summary.json` files.
- The original Cora workflow and the unified three-dataset workflow use different default round/epoch settings and should be reported as separate experiment groups.
- A single unified run uses one seed (default 42). The extended multi-seed study runs seeds 42, 43, and 44 and can be used for mean ± standard deviation reporting.
- The Reddit result represents the configured deterministic sampled subgraph, not the complete Reddit graph.

See `docs/EXPERIMENT_NOTES.md` for the main scientific caveats and `docs/QUICKSTART_FA.md` for a Persian execution guide.

## Citation

If you use ETD-FGL in academic work, please cite the software using the repository's [`CITATION.cff`](CITATION.cff) file. GitHub can also render this metadata through the **Cite this repository** action.

Current citation metadata identifies **Aynaz Salamati** as the author. Journal, conference, DOI, volume, issue, and page metadata can be added after the paper's final publication details are available.

## License

This project is released under the [MIT License](LICENSE).

## Extended journal evaluation

The repository includes a real, non-synthetic extended evaluation suite for the following studies:

1. **Multi-seed statistical evaluation** using seeds 42, 43, and 44, reported as mean ± standard deviation.
2. **Non-IID robustness** using split-aware Dirichlet label skew with α ∈ {0.1, 0.5, 1.0}.
3. **ETD-FGL component ablation** removing topology evidence, explanation evidence, update-anomaly evidence, or trust-weighted aggregation one component at a time.
4. **Sensitivity analysis** over poisoning rate {0.05, 0.10, 0.20, 0.30} and trigger size {2, 3, 4, 5}.
5. **Federation-size scalability** with {5, 10, 20, 30, 50} clients.

The extended experiments never fabricate missing results. Tables and figures are generated only when the corresponding training summaries exist.

### Recommended execution order

Run one study at a time on a resource-constrained workstation:

```text
run_extended_multiseed.bat
run_extended_noniid.bat
run_extended_ablation.bat
run_extended_sensitivity.bat
run_extended_clients.bat
```

Or execute the complete sequence with:

```text
run_all_extended_experiments.bat
```

Completed conditions are skipped automatically unless `--force` is passed to `run_extended_experiments.py`.

### Extended outputs

Raw outputs are written under:

```text
outputs/extended_experiments/
```

Final journal tables and multi-panel figures are built under:

```text
outputs/extended_experiments/journal_summary/
├── tables/
└── figures/
```

Expected final figures include:

- `Fig_Ext_01_MultiSeed_MeanStd`
- `Fig_Ext_02_NonIID_Dirichlet`
- `Fig_Ext_03_ETDFGL_Ablation`
- `Fig_Ext_04_Sensitivity_PoisonRate`
- `Fig_Ext_05_Sensitivity_TriggerSize`
- `Fig_Ext_06_Client_Scalability`

All figures are exported as PNG, PDF, and SVG.
