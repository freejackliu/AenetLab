import numpy as np


def _random_deform(atoms, deformation_limits, rng):
    lo, hi = deformation_limits
    loc = 0.5*(lo + hi)
    scale = (hi - lo)/4
    xx, yy, zz, yz, xz, xy = rng.normal(loc=loc, scale=scale, size=6)
    s = np.array([(1.0 + xx, 0.5*xy, 0.5*xz),
                 (0.5*xy, 1.0 + yy, 0.5*yz),
                 (0.5*xz, 0.5*yz, 1.0 + zz)])
    atoms.set_cell(atoms.cell @ s, scale_atoms=True)

    return atoms


def _normal_deform_range(atoms, bond_range, threshold, rng):
    from aenet.geometry import compute_gr_peaks
    peak_list, gr_list = compute_gr_peaks(atoms,threshold=threshold)
    i = 0
    while peak_list[i] < 1.2:
        i += 1
    pf = peak_list[i]
    lo, hi = bond_range
    loc = 0.5*(lo + hi)
    scale = (hi - lo)/4 
    xx, yy, zz =  rng.normal(loc=loc, scale=scale, size=3)
    s = np.array([(xx/pf, 0., 0.),
                 (0., yy/pf, 0.),
                 (0., 0., zz/pf)])
    atoms.set_cell(atoms.cell @ s, scale_atoms=True)

    return atoms


def _normal_deform_eos(atoms, interv):
    image = atoms.copy()
    lscale = (1+interv) ** (1/3)
    s = np.array([(lscale, 0., 0.),
                  (0.,lscale, 0.),
                  (0., 0., lscale)])
    image.set_cell(atoms.cell @ s, scale_atoms=True)

    return image


def _random_displace(atoms, rattle_size, rng):
    natoms = len(atoms)
    displacements = rng.normal(loc=0.0, scale=rattle_size, size=(natoms, 3))
    drift = displacements.sum(axis=0)/natoms
    displacements -= drift[np.newaxis, :]
    atoms.positions += displacements

    return atoms


def _random_delete(atoms, percent, rng):
    percent = float(percent.strip("%"))/100
    ndelete = int(round(len(atoms)*percent))
    ndelete = rng.choice(ndelete + 1, 1)[0]
    ndelete = min(ndelete, len(atoms) - 1)
    if ndelete == 0:
        return atoms
    else:
        inds = rng.choice(len(atoms), ndelete, replace=False)
        del atoms[inds]
        return atoms


def disturb(atoms, nsamples, deformation_limits=(-0.1, 0.1),
            rattle_size=0.1, no_delete=True, delete_percent="0%",
            replicate=(2, 2, 2), replicate_uplimit=4,  
            bond_range=None, peak_threshold=0,
            eos_interv=None, seed=None, include_initial=False ):
    rng = np.random.RandomState(seed)
    assert atoms.pbc.all()
    images = []
    if include_initial:
        images.append(atoms)
    eos_count1, eos_count2 = 0, 0 
    for _ in range(nsamples):
        image = atoms.copy()
        if (not bond_range) and (not eos_interv):
            # Get supercell first
            if len(image) <= replicate_uplimit:
                image *= replicate
            # Remove image
            if not no_delete:
                image = _random_delete(image, delete_percent, rng)
            # Apply random deformation
            deformation_limits = sorted(deformation_limits)
            image = _random_deform(image, deformation_limits, rng)
            # Apply random displacements
            image = _random_displace(image, rattle_size, rng)
            images.append(image)
        elif bond_range and (not eos_interv):
            image = _normal_deform_range(image, bond_range, peak_threshold, rng)
            images.append(image)
        elif (not bond_range) and eos_interv:
            eos_count1 += 1
            eos_count2 -= 1
            image1 = _normal_deform_eos(image, eos_count1*eos_interv)
            images.append(image1)
            image2 = _normal_deform_eos(image, eos_count2*eos_interv)
            images.append(image2)
        else:
            raise RuntimeError("'bond_range' and 'eos_interv' can not be specified together.")
    return images


def main():
    import os
    import argparse
    from ase.io import read

    parser = argparse.ArgumentParser(
        description="Generate random pertubated structures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add = parser.add_argument
    add("nsamples", type=int, help="number of generated random samples")
    add("sample", type=str, help="sample file to be randomized")
    add("-s", "--strain", nargs=2, type=float, default=(-0.1, 0.1), metavar=("smin", "smax"),
        help="strain deformation sizes")
    add("-r", "--rdisp", type=float, default=0.1, metavar="r", help="size of random atom displacements")
    add("-d", "--delete", type=str, default="0%", metavar="percent", help="percent of deleted atoms")
    add("--replicate", type=int, nargs=3, default=(2, 2, 2), metavar=("nx", "ny", "nz"),
        help="replicate size of small cells")
    add("--replicate_uplimit", type=int, default=4, metavar="limit",
        help="uplimit of whether applying cell replication")
    add("--include_initial", action="store_false", help="whether include the initial sample in outputs")
    add("--prefix", type=str, default="struct", metavar="prefix", help="prefix of the names of output samples")
    args = parser.parse_args()

    atoms = read(args.sample)
    images = disturb(atoms, args.nsamples, deformation_limits=args.strain,
                     rattle_size=args.rdisp, delete_percent=args.delete,
                     replicate=args.replicate, replicate_uplimit=args.replicate_uplimit,
                     include_initial=args.include_initial)

    base, _ = os.path.split(args.prefix)
    if base != "" and not os.path.isdir(base):
        os.makedirs(base)

    for i, image in enumerate(images):
        fname = f"{args.prefix}_{i:05d}.vasp"
        image.write(fname, vasp5=True, sort=True)

    return


if __name__ == "__main__":
    main()

