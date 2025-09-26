# TReF-6: Inferring Task-Relevant Frames from a Single Demonstration

[![Paper](https://img.shields.io/badge/Paper-CoRL_2025-blue)](./TReF_6__CoRL_2025.pdf)

TReF-6 is a framework for **one-shot skill generalization** in robot manipulation.  
It infers a **task-relevant 6-DoF frame** from a single demonstration, enabling motion primitives (e.g., DMPs) to adapt robustly across novel object poses and scene configurations.

<p align="center">
  <img src="media/overview.png" alt="TReF-6 Overview" width="80%">
</p>

---

## ✨ Key Features

- **Single-Demonstration Generalization:** Extracts task-relevant frames without CAD models, dense labels, or multiple demos.
- **Geometry-Driven Optimization:** Uses a directional consistency score to infer the "influence point" governing motion dynamics.
- **Semantic Grounding:** Anchors frames to meaningful object parts via a vision-language model and Grounded-SAM segmentation.
- **Plug-and-Play with DMPs:** Reparameterizes trajectories in inferred frames, improving generalization of classic controllers.

---

## 🛠 Installation

```bash
git clone https://github.com/iqr-lab/tref-6.git
cd tref-6
python3 -m venv tref-env
source tref-env/bin/activate
pip install --upgrade pip
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
```

---

## 🚀 Quick Start

Run a demo in simulation:

```bash
python examples/run_demo.py --task door_opening --policy dmp
```

Train and evaluate with configuration files:

```bash
python examples/train.py --config-name door_opening_baseline.yaml
python examples/eval.py --checkpoint outputs/.../latest.ckpt --task door_opening
```

---

## 📂 Repository Structure

```
tref-6/
├── tref/                   # Core library
│   ├── tasks/              # Task definitions (datasets, environments)
│   ├── policies/           # Policy abstractions (e.g., DMPs)
│   ├── runners/            # Execution/training loops
│   └── utils/              # Shared utilities
├── configs/                # Hydra-based configs
├── examples/               # Training/evaluation scripts
├── tests/                  # Unit tests
└── docs/                   # Documentation
```

---

## 🧠 Method Overview

TReF-6 consists of three stages:

1. **Influence Point Inference**  
   Optimize a *directional consistency score* to find the spatial point best explaining trajectory dynamics.

2. **Semantic Grounding**  
   Align the inferred point with visual features identified by a VLM, then extract a full 6-DoF frame using surface normals and interaction directions.

3. **DMP Reparameterization**  
   Transform the trajectory into the new frame and fit DMPs over relative motions, allowing reuse in new scenes.

```text
Trajectory → Influence Point → Local Frame → DMP Fitting → Generalized Motion
```

---

## 📊 Results (CoRL 2025)

- **Peg-in-Hole Dropping:** 53.3% overall success (+33.3% over baseline)
- **Cabinet Door Opening:** 66.7% overall success (+58.3% over baseline)
- **Surface Wiping:** 66.7% overall success (+33.3% over baseline)

TReF-6 consistently outperformed privileged baselines by preserving **functional constraints** (hinge arcs, contact continuity) even under OOD variations.

---

## 📊 Logging & Visualization

TReF-6 integrates with [Weights & Biases](https://wandb.ai):

```bash
wandb login
```

Logs include metrics, rollout videos, and checkpointed models.

---

## 🧾 License

This project is released under the **MIT License**.  
See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

We thank members of Yale’s Inquisitive Robotics Lab and Qian Wang for valuable feedback and contributions.  
Core design draws inspiration from task-parameterized movement learning, affordance-based imitation, and modern VLM-based scene understanding.
