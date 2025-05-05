# CIVIL

[[PDF]](https://civil2025.github.io/static/resources/CIVIL_anonymous.pdf) [[Project Page]](https://civil2025.github.io/) [[Video]](https://youtu.be/7KeM9StNTnA?si=0DMCP-GnoROBuv3P) ![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

This repository contains the implementation for CIVIL, a framework designed to enable real robots to learn from multimodal visual-language instruction data using imitation learning.

## Getting Started

### Set Up DEVA
We use DEVA (Tracking Anything with DEVA) as a dependency for visual tracking of offline data. Configure DEVA by following the instructions at the official repository:
[Tracking-Anything-with-DEVA](https://github.com/hkchengrex/Tracking-Anything-with-DEVA.git)

> ⚠️ Make sure DEVA is installed and working before proceeding to install the environment.

### Install the Environment

Use the provided `environment.yml` file to create the conda environment:

```bash
conda env create -f environment.yml
conda activate CIVIL
```

### Prepare Simulation Dataset

To work with the simulation dataset, refer to the instructions provided by [CALVIN](https://github.com/mees/calvin.git).

For generating CALVIN data:

- Use `data_generation/generate_calvin.py` to create marker data.
- Use `add_segmentation_calvin.py` to generate segmentation masks.

###  Real-World Dataset Sample

A sample of the real-world dataset we used with Panda robots is available under:

```
panda_data_example/
```

---

For further documentation on training scripts, experiment setup, and user study evaluations, refer to the relevant scripts in the repository.


Check out robot rollouts with CIVIL on [our website](https://civil2025.github.io/).