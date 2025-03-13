# Supporting Data for "Assessing zero-shot generalisation behaviour in graph-neural-network interatomic potentials"

---

This repository contains the supporting data of:

<div align="center">

> **[Assessing zero-shot generalisation behaviour in graph-neural-network interatomic potentials](https://arxiv.org/abs/2502.21317)**\
> _Chiheb Ben Mahmoud, Zakariya El-Machachi, Krystian A. Gierczak, John L. A. Gardner, and Volker L. Deringer_

</div>

## Repo Overview

- **[data/](data/)** contains the data necessary to generate the figures and the tables and organized as follows:
  - GO-MACE-23_PCA: contains the PCA map coordinates (Figure 1) under `.npy` format and the corresponding datasets in `extxyz` format
  - rMD17: contains the subset of rMD17 dataset used in this work
  - vibrational_spectra: contains the vibrational spectra of Figure 4
  - fullerenes: contains the DFT and ML predictions of fullerens in `extxyz` format
  - c60-traj: contains the encapsulated molecule reactions (X@C<sub>60</sub>) in `extxyz` format
  - aspirin-traj: contains the main events of the aspirin high temperature reaction in vacuum in `extxyz` format
- **[notebooks/](notebooks/)** contains the jupyter notebooks to generate the figures and tables (in latex format)
- **[models/](models/)** contains the MLIPs used in this work
- **[code_snippets/](code_snippets/)** contains examples of the DFT input files and how to generate
- **[example_vibration_workflow/](example_vibration_workflow/)** contains an example of how to obtain the vibrational spectra of molecules

## Dependencies

To run the notebooks, you need the usual dependencies (`numpy`, `matplotlib`, `jupyter`, `ase`).

To generate the supporting data, you might also need `mace`(can be found here [https://github.com/ACEsuit/mace](https://github.com/ACEsuit/mace)) and `GraphPES` (an archived version can be found here [https://zenodo.org/records/14956211](https://zenodo.org/records/14956211)).
