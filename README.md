# TJ Quantum Research Lab's Implementation of the ADAPT-VQE Algorithm

## Overview
The Adaptive Derivative-Assembled Pseudo-Trotter Variational Quantum Eigensolver (ADAPT-VQE) is a hybrid quantum-classical algorithm that aims to find the ground state of a molecule by finding the smallest eigenvalue of the molecular hamiltonian. This repository is a framework to run the ADAPT-VQE algorithm with custom optimizer, pools, and molecules, as well as generating plots comparing different experiments.
<br>
Currently implemented optimizers: `SGD`, `Adam`, `TrustRegion`, `Cobyla`, `BFGS`, and `LBFGS`.
<br>
Currently implemented pools: `FSD`, `GSD`, `QEB`, `IndividiualTUPS`, `AdjacentTUPS`,  `MultiTUPS`, `FullTUPS`, `UnrestrictedTUPS`, `UnresIndividualTUPS`

## Getting Started

To install required python dependencies, run
```sh
pip3 install -r requirements.txt
```
or
```sh
nix develop
```

<br>
<br>

To run several experiments at a time with multiprocessing, run
```sh
python3 -m tj_adapt_vqe.experiments
```

To run a single experiment, run
```sh
python3 -m tj_adapt_vqe.train
```

<br>
<br>

To view experiment results before generating plots, run
```sh
./scripts/dashboard.sh
```

After running experiments, to generate plots (in the `results` directory) run
```
python3 -m tj_adapt_vqe.post_process
```