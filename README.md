# Genetic Neural Architecture Search (Genetic_NAS)

![GitHub Code License](https://img.shields.io/github/license/sypsyp97/Genetic_NAS?style=plastic&logo=github&logoColor=white&color=blue)
![GitHub last commit](https://img.shields.io/github/last-commit/sypsyp97/Genetic_NAS?style=plastic&logo=github&logoColor=white&color=yellow)
![GitHub pull request](https://img.shields.io/badge/PRs-not_welcome-red?style=plastic&logo=github&logoColor=white)
[![Documentation Status](https://img.shields.io/badge/Documentation-Online-green?style=plastic&logo=read-the-docs&logoColor=white)](https://sypsyp97.github.io/Genetic_NAS/)


Welcome to the Genetic_NAS repository, the home of Yipeng Sun's Master's Thesis project. This project explores the use of genetic algorithms to search for adaptable models specifically designed for Edge TPU. We aim to leverage the capabilities of the Edge TPU to enhance inference speed while maintaining a high level of accuracy in image classification tasks.

---

## Overview

The expanding use of edge devices and the constraints on cloud connectivity call for efficient, on-device neural networks for real-time applications. Manually designing these networks is complex due to the need for a balance between accuracy, speed, and efficiency, particularly for edge devices with computational and power limitations. Neural Architecture Search (NAS) has been introduced to automate the design process, often producing models that surpass human-designed ones. Genetic Algorithms, which traverse vast search spaces to find optimal solutions using mechanisms akin to natural selection, have shown potential in NAS. The goal of this project is to harness the power of Genetic Algorithm-based NAS to create edge-optimized models specifically designed for Edge TPUs. This aims to enable real-time, accurate image analysis across various applications while minimizing power consumption and computational resource usage.

If you find our work and our open-sourced efforts useful, ⭐️ to encourage our following development! :)

---
## Prerequisites

To get the most out of this project, you should have:

- Familiarity with Python 3.9 and above
- Basic understanding of neural networks
- Some knowledge about genetic algorithms

---
## Environment and Installation

This project is developed in Python 3.9 environment with TensorFlow 2.11 being the major library used. To set up the environment, follow these steps:

1. Clone the repository to your local machine：
 ```bash
git clone https://github.com/sypsyp97/Genetic_NAS.git
cd Genetic_NAS
```
2. Set up a virtual environment and install the required packages using `pip`:

```bash
python3 -m venv env
source env/bin/activate
pip install -r requirements.txt
```
---
## Repository Structure

The repository is structured as follows:

- `src/`: This directory contains the main source code for the project and utility scripts that aid in various tasks throughout the project.
- `get_datasets/`: This directory includes scripts for data acquisition.
- `tf_flower_example.py`: A Python script that is used for testing the application with the TensorFlow Flowers dataset.
- `requirements.txt`: Specifies the libraries and their respective versions required for this project.
- `rst_generator.py`: A script used to generate reStructuredText (.rst) files for use with the Sphinx documentation generator.

---

## Usage Example

Before running the NAS process, please ensure that you have an Edge TPU device available. You will also need to install the necessary libraries and dependencies for working with the Edge TPU. Instructions for setting up the Edge TPU can be found in the [Coral documentation](https://coral.ai/docs/accelerator/get-started/).

Here's an example of how you can use the `start_evolution` function to initiate the process of NAS:

```python
from src.Evolutionary_Algorithm import start_evolution

population_array, max_fitness_history, average_fitness_history, best_models_arrays = start_evolution(
        train_ds=train_dataset,
        val_ds=val_dataset,
        test_ds=test_dataset,
        generations=4,
        population=20,
        num_classes=5,
        epochs=30,
        time=formatted_date
    )
```

You can also easily start the NAS process by running the `tf_flower_example.py` script. This script has been designed to make running the NAS process simpler by predefining certain parameters and steps.
```bash
python tf_flower_example.py
```
---
## Documentation

The detailed documentation for the functions used in this project is available [online](https://sypsyp97.github.io/Genetic_NAS/).

Alternatively, you can clone the `gh-pages` branch of this repository to view the documentation offline:

1. Open a terminal.
2. Run the command `git clone -b gh-pages https://github.com/sypsyp97/Genetic_NAS.git`. This will clone only the `gh-pages` branch.
3. Navigate to the cloned directory using `cd Genetic_NAS`.
4. Open the `index.html` file in a web browser to view the documentation.

---
## License

This project is licensed under the terms of the [MIT License](LICENSE). 

---
## Citation

If this work is helpful, please cite as:

```bibtex
@Misc{Genetic_NAS,
  title = {Genetic Neural Architecture Search for Edge TPU},
  author = {Yipeng Sun},
  howpublished = {\url{https://github.com/sypsyp97/Genetic_NAS.git}},
  year = {2023}
}
```
