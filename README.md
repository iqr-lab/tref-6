# [<u>TReF-6</u>](https://arxiv.org/pdf/2509.00310): Inferring Task-Relevant Frames from a Single Demonstration for One-Shot Skill Generalization


[Yuxuan Ding](https://github.com/EasonDi), [Shuangge Wang](https://github.com/wshuangge), [Tesca Fitzgerald](https://www.tescafitzgerald.com/)

Yale University

<img src="media/pipeline.png" alt="drawing" width="70%"/>

TReF-6 is a framework for **one-shot skill generalization** in robot manipulation.  
It infers a **task-relevant 6-DoF frame** from a single demonstration, enabling motion primitives (e.g., DMPs) to adapt robustly across novel object poses and scene configurations.

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

Dependencies installation:

Install **KINOVA KORTEX API** following [this guide](https://github.com/Kinovarobotics/Kinova-kortex2_Gen3_G3L)

Install **Grounded SAM 2** following [this guide](https://github.com/IDEA-Research/Grounded-SAM-2)


---

## 🚀 Quick Start

**Collect data:**

```bash
python tref_6/gravity_compensation.py
```

**Evaluate in simulation:**


For single influence point detection in 2D scenarios:
```bash
python simulation/test_2d.py
```

For single influence point detection in 3D scenarios:
```bash
python simulation/test_3d.py
```

For sequential influence points detection in 3D scenarios:
```bash
python simulation/test_sequential.py
```

**Run a demo in real world:**
```bash
python tref_6/run.py
```

---

## 📂 Repository Structure

```
demonstrations/             # Task definitions (datasets)
├── door_open/
├── drop/                  
└── wiping/
docs/                       # Documentation
features/                   # Extracted local features & visualizations for each task
├── door_open/
├── drop/                  
└── wiping/
media/                      # Pipeline picture
simulation/                 # Evaluation scripts in simulated environment
tref_6/                     # Core library
├── tref/                   # Core library
│   ├── tasks/              # Task definitions (datasets, environments)
│   ├── policies/           # Policy abstractions (e.g., DMPs)
│   ├── runners/            # Execution/training loops
│   └── utils/              # Shared utilities
├── configs/                # Hydra-based configs
├── examples/               # Training/evaluation scripts
├── tests/                  # Unit tests
└── docs/                   # Documentation
visualization/              # Intermediate step visualization
```

---

## 📊 Visualization
To visualize the score landscape of Directional Consistency Score on an example trajectory:
```bash
python visualization/score_landscape.py
```

To visualize the detected influence point on the image:
```bash
python visualization/extract_feature.py
```

To visualize the generated trajectory:
```bash
python visualization/visualize_trajectory.py
```

---

## 🧾 License

This project is released under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

We thank members of Yale’s Inquisitive Robotics Lab and Qian Wang for valuable feedback and contributions.  
