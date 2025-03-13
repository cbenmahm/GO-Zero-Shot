from ase import units
from ase.md.npt import NPT
from ase.io import read
from ase.md import MDLogger
import time
from ase.md.velocitydistribution import (
    ZeroRotation,
    Stationary,
    MaxwellBoltzmannDistribution,
)
from mace.calculators import MACECalculator
import os


def write_frame():
    dyn.atoms.write(f"{path}/mace_md.xyz", append=True)


# set cuda visible devices
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

calculator = MACECalculator(
    model_paths="/path/ to/ MACE/ model", device="cuda", default_dtype="float32"
)
path = "/path / to / starting/ configuration"
init_conf = read(f"{path}/c60-CH4.xyz")

init_conf.set_calculator(calculator)

T_init = 500  # K

dt = 0.5 * units.fs  # fs time step

start = time.time()
# Set the momenta corresponding to T_init (K)
MaxwellBoltzmannDistribution(init_conf, temperature_K=T_init)
Stationary(init_conf)
ZeroRotation(init_conf)


print("Initialising structure...")
## Code for heating from 300 to 1500 goes here..

dyn = NPT(init_conf, dt, ttime=10.0 * units.fs, externalstress=0.0, temperature_K=500)
dyn.attach(write_frame, interval=100)
dyn.attach(
    MDLogger(dyn, init_conf, f"{path}/md.log", stress=False, peratom=True, mode="a"),
    interval=100,
)
# Run the holding phase
dyn.run(20000000)


end = time.time()
print("MD finished!")
print("Time Taken to run is:")
print(end - start)
