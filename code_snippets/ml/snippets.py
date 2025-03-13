import ase.io as ase_io
import numpy as np
from mace.calculators import MACECalculator
from graph_pes.models import load_model
from graph_pes.utils.calculator import GraphPESCalculator

atoms = ase_io.read("napht.xyz")


# examples with MACE
calc1 = MACECalculator(
    model_paths="../../models/GO-MACE-23_cpu.model",
    device="cpu",
    model_type="MACE",
    default_dtype="float32",
)
calc2 = MACECalculator(
    "../../models/MACE-OFF23_large.model", device="cpu", default_dtype="float32"
)
calc3 = MACECalculator(
    "../../models/MACE-OFF24_medium.model", device="cpu", default_dtype="float32"
)

atoms.calc = calc1
atoms.info["gomace_energy"] = atoms.get_potential_energy()
atoms.arrays["gomace_forces"] = atoms.get_forces()

atoms.calc = calc2
atoms.info["maceoff23_energy"] = atoms.get_potential_energy()
atoms.arrays["maceoff23_forces"] = atoms.get_forces()

atoms.calc = calc3
atoms.info["maceoff24_energy"] = atoms.get_potential_energy()
atoms.arrays["maceoff24_forces"] = atoms.get_forces()

atoms.calc = None

ase_io.write("napht-ml.xyz", atoms)

# example with GraphPES
calc = load_model("../../trained_models/schnet.pt")
calc = GraphPESCalculator(calc, device="cpu")
atoms.calc = calc
atoms.info["graphPES_energy"] = atoms.get_potential_energy()
atoms.arrays["graphPES_forces"] = atoms.get_forces()

# get atomic energies with MACE
calc1 = MACECalculator(
    model_paths="../../models/GO-MACE-23_cpu.model",
    device="cpu",
    model_type="MACE",
    default_dtype="float32",
)
atomic_energies = calc1.get_property("node_energy", atoms)

# get features from MACE
features = calc1.get_descriptors(atoms)
features = np.mean(features, axis=0)
