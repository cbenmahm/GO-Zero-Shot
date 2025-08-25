import ase
from ase import atoms
import numpy as np
from matscipy import rings

import warnings
from multiprocessing import Pool
from functools import partial


def rings_distribution(atoms, cutoff=None, maxlength=12, ddof=1, dump_all=False):
    if not isinstance(atoms, list):
        atoms = [atoms]

    nframe = len(atoms)

    rings_dist = np.zeros(maxlength + 1)
    rings_dist_frac = np.zeros(maxlength + 1)
    rings_all = []
    rings_frac_all = []
    for i in range(nframe):

        rings_tmp = rings.ring_statistics(atoms[i], cutoff, maxlength=maxlength)

        dist_tmp = np.pad(rings_tmp, (0, maxlength + 5 - len(rings_tmp)))
        dist = dist_tmp[0 : maxlength + 1]
        dist_frac = dist / np.sum(dist) * 100

        rings_dist += dist
        rings_dist_frac += dist_frac

        rings_all.append(list(dist))
        rings_frac_all.append(list(dist_frac))

    rings_dist = rings_dist / nframe
    rings_dist_frac = rings_dist_frac / nframe

    ### calculate the standard deviation
    rings_all = np.array(rings_all)
    rings_frac_all = np.array(rings_frac_all)

    err = np.zeros(maxlength + 1)
    err_frac = np.zeros(maxlength + 1)

    for i in range(maxlength + 1):
        err[i] = np.std(rings_all[:, i], ddof=ddof)
        err_frac[i] = np.std(rings_frac_all[:, i], ddof=ddof)

    rings_length = np.arange(0, maxlength + 1)

    if not dump_all:
        return rings_length, rings_dist, rings_dist_frac, err, err_frac

    if dump_all:
        return (
            rings_length,
            rings_dist,
            rings_dist_frac,
            err,
            err_frac,
            rings_all,
            rings_frac_all,
        )
