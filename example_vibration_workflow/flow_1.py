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


class OptimizeJob:
    """
    Class to carry out a full optimization of an atoms object

    Parameters
    ----------
    name
        Name of the job.
    fmax
        float that determines the forces after optimization
    constant_volume
        bool that if true will enforce an optimization with constant volume
    """

    name: str = "MACE-Optimization"
    fmax: float = 0.00005
    constant_volume: bool = True

    def make(self, structure: Structure, mace_model: str):
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        atoms.calc = MACECalculator(
            model_paths=mace_model,
            device="cpu",
            default_dtype="float32",
        )
        opt = LBFGS(
            FrechetCellFilter(
                atoms, constant_volume=self.constant_volume, hydrostatic_strain=True
            ),
            logfile="./optimization.log",
            trajectory="./optimization.traj",
        )
        opt.run(fmax=self.fmax)

        out_structure = adaptor.get_structure(atoms)
        # set cell to be the same as the input structure
        out_structure.lattice = structure.lattice

        results = {}
        results["optimized_structure"] = out_structure
        results["input_structure"] = structure
        results["energy"] = atoms.get_potential_energy()
        return results


class StaticJob:
    """
    Class to compute energy and forces for a structure

    Parameters
    ----------
    name
        Name of the job.
    """

    name: str = "MACE-StaticRun"

    def make(self, structure: Structure, path: str):
        adaptor = AseAtomsAdaptor()
        atoms = adaptor.get_atoms(structure)
        atoms.calc = MACECalculator(
            model_paths=path, device="cpu", default_dtype="float32"
        )
        results = {}
        results["input_structure"] = structure
        results["energy"] = atoms.get_potential_energy()
        results["force"] = atoms.get_forces()
        return results


def create_displacements(structure: Structure, supercell_matrix):
    factor = VaspToTHz  # we need this conversion factor. We will rely on eV
    cell = get_phonopy_structure(
        structure
    )  # this transforms the structure into a format that phonopy can use
    phonon = Phonopy(
        cell,
        supercell_matrix,  # we need to compute forces for displaced supercells of our initial structure
        factor=factor,
    )
    phonon.generate_displacements(
        distance=0.001
    )  # phonopy will generate the displacements

    supercells = (
        phonon.supercells_with_displacements
    )  # these are all displaced structures

    displacements = []
    idx = 0
    for icell in supercells:
        structure = get_pmg_structure(
            icell
        )  # we will collect all structures in a pymatgen structure format
        displacements.append(structure)

        # convert from pymatgen to ase
        ats = AseAtomsAdaptor().get_atoms(structure)
        # mkdir if it does not exist to store the displaced structures
        os.makedirs("displacements-CASTEP", exist_ok=True)
        os.makedirs("displacements-PSI4", exist_ok=True)
        # write the displaced structure to a file
        write(f"displacements-CASTEP/in-{idx}.cell", ats, format="castep-cell")
        idx += 1
        # write displacements for PSI4
        write("displacements-PSI4/psi4.xyz", ats, append=True)
    return displacements


def run_displacements(
    displacements,
    path="/u/vld/sedm6197/MACE-test/MD/1500K/potential/MACE_model_swa.model",
):
    static_maker = StaticJob()
    runs = []
    forces = []
    for displacement in displacements:
        displacement_run = static_maker.make(displacement, path=path)
        forces.append(displacement_run["force"])
        runs.append(displacement_run)

    return forces


def summarize_phonopy_results(
    structure: Structure,
    forces,
    supercell_matrix,
    mesh=[10, 10, 10],
    output_file="total-dos.dat",
    phonopy_config_output="phonopy.yaml",
):
    forces_set = []
    for force in forces:
        forces_set.append(np.array(force))

    factor = VaspToTHz
    cell = get_phonopy_structure(structure)
    phonon = Phonopy(
        cell,
        supercell_matrix,
        factor=factor,
    )
    phonon.generate_displacements()
    phonon.produce_force_constants(forces=forces_set)
    phonon.save(phonopy_config_output)

    phonon.run_mesh(mesh)
    phonon.run_total_dos(sigma=0.3, freq_min=-1, freq_pitch=0.1)
    phonon.write_total_dos(output_file)


if __name__ == "__main__":

    structure = read("napht.xyz")

    # load the molecule and relax it with MACE
    structure = AseAtomsAdaptor.get_structure(structure)
    optim = OptimizeJob()
    optim.make(structure=structure, mace_model="../models/GO-MACE-23_cpu.model")

    # load the relaxed molecule
    structure = read("./optimization.traj", "-1")
    structure = AseAtomsAdaptor.get_structure(structure)
    supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]

    # create the displacements useful for CASTEP and PSI4
    displacements = create_displacements(
        structure=structure, supercell_matrix=supercell_matrix
    )
    forces_gomace = run_displacements(
        displacements=displacements,
        path="../models/GO-MACE-23_cpu.model",
    )

    forces_maceoff23 = run_displacements(
        displacements=displacements,
        path="../models/MACE-OFF23_large.model",
    )

    summarize_phonopy_results(
        structure,
        forces_gomace,
        supercell_matrix,
        output_file="gomace-total-dos.dat",
        phonopy_config_output="phonopy_gomace.yaml",
    )
    summarize_phonopy_results(
        structure,
        forces_maceoff23,
        supercell_matrix,
        output_file="maceoff23-total-dos.dat",
        phonopy_config_output="phonopy_maceoff23.yaml",
    )
