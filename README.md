# ScaleOp

**Learning Parameter-Space Scaling Operators for Transformer Model Families**

Can we learn a general operator that transforms a pretrained transformer of one size into another within the same model family — instead of training or distilling every size independently?

## Project structure

```
ScaleOp/
├── README.md
├── pyproject.toml / uv.lock     # uv env (torch from the cu130 index)
├── spark.sh                     # Mac <-> DGX Spark sync-and-run helper
├── lib/                         # library — importable API (models, alignment, ...)
├── tools/                       # utility scripts you run (smoke test, model download)
├── experiments/                 # runnable experiment scripts (M1+)
├── configs/                     # experiment YAMLs
├── tests/                       # sanity tests
├── data/                        # frozen corpora & activation caches (gitignored)
└── project_docs/
    ├── reports/                 # milestone summaries → become Paper 1
    └── results/                 # experiment outputs (gitignored)
```

`lib/` is the importable library (the alignment/projection/residual machinery, imported as `from lib import ...`); `tools/` and `experiments/` are scripts you run that use it — `tools/` for setup/utility, `experiments/` for the research runs. `project_docs/` holds the plans and the generated reports/results.

## Documents

Milestone reports live in `project_docs/reports/` (M0–M5 plus the paper
outline); review decisions in `project_docs/reviews/`; a human-friendly
tracker in `project_docs/status.html`.

## The question, in one sentence

After alignment, do the residuals between a projected larger Pythia model (1.4B) and an independently trained smaller one (410M) contain systematic, learnable structure — or are they noise? Either answer is a paper.

## Development workflow (uv + a remote GPU host)

Development happens on a workstation; runs execute on a GPU host (ours: a DGX Spark — GB10, 128GB, CUDA 13; host settings in a local `.spark.env`, see `.spark.env.example`). Code is mirrored to the host and run there with `uv`. The env is pinned in `pyproject.toml` + `uv.lock` (torch from the cu130 index).

```bash
./spark.sh setup            # create the uv env on the Spark (torch cu130 + deps)
./spark.sh run <cmd...>     # rsync code up, then `uv run <cmd>` on the Spark
./spark.sh pull             # bring back uv.lock + reports + results
```

Setup checks live in `tools/`; experiment scripts (M1+) live in
`experiments/`. Run the setup checks:

```bash
./spark.sh run "pytest -q tests/"
./spark.sh run "python tools/smoke_test.py"
./spark.sh run "python tools/download_models.py"
```

## Run on AWS / any CUDA machine (no Spark needed)

The code is device-generic; the Spark is just our compute box. On any CUDA
Linux instance (AWS g5/g6/p4, GCP, Lambda — 24 GB+ GPU, 150 GB disk):

```bash
git clone <this repo> && cd ScaleOp
bash tools/cloud_setup.sh --bootstrap   # uv env + checks + models/corpus/caches
```

Then run experiments directly (no `spark.sh` indirection), e.g.:

```bash
uv run python experiments/m5_train.py --config configs/m5.yaml --init hybrid
```

Notes: `uv.lock` is cross-platform (aarch64 + x86_64; torch from the cu130
index). Determinism is per-hardware — expect bit-identical reruns on the same
GPU, small numeric drift across GPU architectures.

## Hardware

- Phase 1: NVIDIA DGX Spark (128GB unified memory) — all Pythia-scale work fits locally; any 24 GB+ CUDA GPU works (see above)
- Phase 2/3: cloud GPU burst for 7B+ runs

## Rules of the road

1. Determinism everywhere — greedy decoding, fixed seeds; sampling randomness is noise, not signal.
2. Controls are non-negotiable — every residual-structure test runs against a shuffled control.
3. Read `project_docs/reports/M2.md` and `project_docs/reports/M4.md` personally before allowing M5 fine-tuning spend.
4. Open-source and arXiv as soon as Phase 1 results are solid — citations and adoption start the clock for everything downstream.
