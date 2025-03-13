from pymatgen.io.ase import AseAtomsAdaptor
from ase.optimize import LBFGS
from ase.filters import FrechetCellFilter
from mace.calculators import MACECalculator
import numpy as np
from pymatgen.core.structure import Structure
import phonopy
from phonopy import Phonopy
from phonopy.units import VaspToTHz
from pymatgen.io.phonopy import get_phonopy_structure, get_pmg_structure
from ase.io import read, write
import os


def get_phonons_from_dft(config: str, output_file: str):
    phonon = phonopy.load(config)
    phonon.run_mesh([10, 10, 10])
    phonon.run_total_dos(sigma=0.3, freq_min=-1, freq_pitch=0.1)
    phonon.write_total_dos(output_file)


if __name__ == "__main__":

    displacements_castep = read("napht-castep.xyz", ":")
    forces_castep_list = [x.arrays["castep_forces"] for x in displacements_castep]
    ph_castep = phonopy.load("phonopy_gomace.yaml", is_symmetry=True)
    ph_castep.forces = forces_castep_list
    ph_castep.produce_force_constants()
    ph_castep.save("phonopy_castep.yaml", settings={"force_constants": True})
    get_phonons_from_dft("phonopy_castep.yaml", "castep-total-dos.dat")

    displacements_psi4 = read("napht-psi4.xyz", ":")
    forces_psi4_list = [x.arrays["spice_forces"] for x in displacements_psi4]
    ph_castep = phonopy.load("phonopy_gomace.yaml", is_symmetry=True)
    ph_castep.forces = forces_psi4_list
    ph_castep.produce_force_constants()
    ph_castep.save("phonopy_psi4.yaml", settings={"force_constants": True})
    get_phonons_from_dft("phonopy_psi4.yaml", "psi4-total-dos.dat")
