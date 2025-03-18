"""
Prediction of pertubated structure energies from the taylor expansion.
"""

import os
import argparse
import struct
import numpy as np
from ase.calculators.singlepoint import SinglePointCalculator
from aenet.xsf import read_xsf, write_xsf


def taylor_first(atoms, drmax=0.02, nsamples=20, fcut=[0.3, 3.0], seed=None):
    """Generate pertubated atoms configurations with energies
    predicted by the first-order Taylor expansion.

    This can be useful for incorporating force informations to ANN
    training procedure. Please see doi:10.1038/s41524-020-0323-8 for
    more details.

    Arguments:
        atoms       :   Atoms object
        drmax       :   magnitube of displacement (about 1 - 1.5% average bond length)
        nsamples    :   number of generated structures (20 - 80)
        fcut        :   lower cutoff of force magnitudes
        seed        :   seed of random generator

    Return:
        A list of disturbed atoms objects with predicted energy and forces.
    """
    if isinstance(seed, np.random.RandomState):
        rng = seed
    else:
        rng = np.random.RandomState(seed=seed)
    # energy and forces
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    fmag = np.linalg.norm(forces, axis=1)
    mask = np.logical_not((fmag > fcut[0]) & (fmag < fcut[1]))
    if np.all(mask):
        return None
    samples = []
    for i in range(nsamples):
        rvec = rng.uniform(-1, 1, size=(len(atoms), 3))
        rvec = rvec/np.linalg.norm(rvec, axis=1)[:, np.newaxis]
        disp = rng.uniform(0, drmax, size=len(atoms))[:, np.newaxis]*rvec
        disp[mask, :] = 0
        drift = np.sum(disp[np.logical_not(mask), :], axis=0)/np.logical_not(mask).sum()
        disp[np.logical_not(mask), :] -= drift[np.newaxis, :]
        delta_e = -np.sum(forces*disp)
        a = atoms.copy()
        a.positions += disp
        a.calc = SinglePointCalculator(a, energy=energy + delta_e, forces=forces)
        samples.append(a)

    return samples


def main():
    import tqdm
    parser = argparse.ArgumentParser(
        description="Generate augmented datasets with Taylor expansion.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add = parser.add_argument
    add("list", help="txt file containing list of xsf files to be augmented.")
    add("-d", "--drmax", type=float, default=0.02, metavar="drmax", help="max random displacements")
    add("-n", "--nsamples", type=int, default=10, metavar="nsamples", help="number of pertubated samples")
    add("--nfirst", type=int, metavar="nfirst", help="use nfirst structures for taylor expansion")
    add("-f", "--fcut", type=float, nargs=2, default=(0.3, 3.0), metavar="fcut", help="lower and higher force cutoffs")
    add("-s", "--seed", type=int, metavar="seed", help="seed of random number generator")
    add("-t", "--test_percent", type=str, metavar="percent", default="10%", help="percent of test samples")
    add("-o", "--output", type=str, metavar="output", default="taylor", help="output folder of generated files")
    args = parser.parse_args()
    if not os.path.isdir(args.output):
        os.makedirs(args.output)
    rng = np.random.RandomState(args.seed)
    open(f"{args.output}/seed", "w").write(f"{args.seed}\n")
    xsf_list = [line.strip() for line in open(args.list).readlines()]
    if args.nfirst is None:
        nfirst = len(xsf_list)
    else:
        nfirst = min(args.nfirst, len(xsf_list))
        nfirst = max(nfirst, 0)
    percent = float(args.test_percent.strip("%"))/100
    istest = np.array([False for _ in xsf_list])
    istest[0:int(percent*len(xsf_list))] = True
    rng.shuffle(istest)
    aug_list = []
    tags_seen = set()
    for k, v in tqdm.tqdm(enumerate(xsf_list)):
        tag = os.path.basename(v)[0:-4]
        if tag in tags_seen:
            raise RuntimeError("Each xsf file in list must have unique name!")
        tags_seen.add(tag)
        atoms = read_xsf(v)
        name = f"{args.output}/{tag}.xsf"
        aug_list.append([name, bool(istest[k])])
        write_xsf(name, atoms)
        if not istest[k] and k < nfirst:
            samples = taylor_first(atoms, drmax=args.drmax, nsamples=args.nsamples, fcut=args.fcut, seed=rng)
            if samples is not None:
                for isample, sample in enumerate(samples):
                    name = f"{args.output}/{tag}_taylor_{isample:03d}.xsf"
                    aug_list.append([name, bool(istest[k])])
                    write_xsf(name, sample)
    ngenerated = len(aug_list)
    infoout = open(f"{args.output}/info", "w")
    listout = open(f"{args.output}/list", "w")
    trnsplit = open(f"{args.output}/train.trnsplit", "wb")
    infoout.write(f"{ngenerated}\n")
    listout.write(f"{ngenerated}\n")
    trnsplit.write(struct.pack("i", ngenerated))
    for i in aug_list:
        infoout.write(f"{i[0]} {i[1]}\n")
        listout.write(i[0] + '\n')
        trnsplit.write(struct.pack("i", i[1]))
    infoout.close()
    listout.close()
    trnsplit.close()
    print("Generation done!")
    print("Please use generated 'list' and 'train.trnsplit' file for generation and training.")
    print("DO NOT set maxenergy in train.x input and use train.trnsplit to fix testing set.")


def _emt_test():
    from ase.calculators.emt import EMT
    from ase.build import bulk

    atoms = bulk("Au", cubic=True)*(2, 2, 2)
    atoms[3].symbol = "Pt"
    atoms.calc = EMT()
    atoms.rattle(0.1)
    e0 = atoms.get_potential_energy()/len(atoms)
    print(np.linalg.norm(atoms.get_forces(), axis=1))
    c = []
    s = []
    for k, i in enumerate(taylor_first(atoms, drmax=0.02, nsamples=20, fcut=[0.5, 3.0], seed=42)):
        e1 = i.get_potential_energy()/len(atoms)
        i.calc = EMT()
        e2 = i.get_potential_energy()/len(atoms)
        print(e0, e1, e2, abs(e1 - e2))
        s.append(abs(e1 - e2))
        c.append(abs(e0 - e2))
    print(np.mean(c), np.mean(s), np.std(s))


if __name__ == "__main__":
    main()

