import ase.io as ase_io
from ase.calculators.psi4 import Psi4
from glob import glob
import numpy as np

for f in ["traj.xyz"]:
    print(f, flush=True)
    frames = ase_io.read(f, ":")
    for x in frames:
        calc = Psi4(atoms=x,
            method = 'wb97m-d3(bj)',
            memory = '256GB', 
            basis = 'def2-TZVPPD',
            num_threads = "max",)
        print("arrived", flush=True)
        x.calc = calc
        try:
            x.info["spice_energy"] = x.get_potential_energy()
	    x.arrays["spice_forces"] = x.get_forces()
            print(x.info["spice_energy"], flush=True)
        except Exception:
            print("could not run this frame")
            x.info["spice_energy"] = "0.0"
            ase_io.write(f+"-failed-frame.xyz", x, append=True)
        ase_io.write(f+"-psi4.xyz", x, append=True
