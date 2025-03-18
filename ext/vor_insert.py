import numpy as np
import pandas as pd
from math import floor, pi, ceil
from ase import Atom, Atoms
from scipy.spatial import Voronoi
from scipy.special import perm
from aenet.geometry import *


def uniform_distribute(a, ip_vts, nvts, nbins, density, rng):
    center_pos = a.cell.sum(0)/2
    max_d = min([np.linalg.norm((a.cell[i%3]+a.cell[(i+1)%3])/2-center_pos)\
            for i in range(3)])
    d = np.sqrt(np.sum((ip_vts - center_pos) ** 2, axis=1))
    l = (max_d - min(d)) / (nbins - 1)
    vts_bins = [[] for i in range(nbins)]
    count, target_vts = 0, []
    for ind, ip_vt in enumerate(ip_vts):
        if d[ind] <= max_d:
            k = int((d[ind] - min(d)) // l)
        else:
            k = nbins - 1
        vts_bins[k].append(ip_vt)
    V = 4/3 * pi * (max(d)**3 - min(d)**3)
    for ind, vts_bin in enumerate(vts_bins):
        vts_bin = np.asarray(vts_bin)
        if ind+1 < nbins:
            if density:
                Vi = 4/3 * pi * ((min(d) + (ind + 1)*l)**3 - (min(d) + ind*l)**3)
                nvt = max(floor(nvts * Vi / V), 1) 
            else:
                nvt = max(floor(nvts/nbins), 1)
            count += nvt
        else:
            nvt = nvts - count
        if nvt <= len(vts_bin):
            vts_c = rng.choice(len(vts_bin), nvt, replace=False)
        else:
            vts_c = np.arange(len(vts_bin))
        target_vt = vts_bin[vts_c]
        target_vts += target_vt.tolist()
    return np.asarray(target_vts)


def cluster_distribute(a, ip_vts, ind0, nvts, dmin, dmax, rng):
    new_atoms = Atoms(positions=ip_vts, cell=a.cell, pbc="111")
    dlmax = np.min([dmax,np.max(a.get_cell_lengths_and_angles()[:3])/2])
    ds = new_atoms.get_distances(ind0, np.arange(len(ip_vts)),mic=True)
    final_ds = pd.Series(ds)
    final_ds.sort_values(ascending=True)
    final_index = []
    for i in final_ds.index:
        if dmin < final_ds[i] < dlmax:
            final_index.append(i)
    final_index = np.asarray(final_index)
    size = len(final_index)
    t = rng.choice(size, min(size,nvts),replace=False)
    target_vts = ip_vts[final_index[t]]
    return target_vts


def block_distribute(a, nvts):
    nL = round(nvts ** (1/3))
    dL = a.get_cell_lengths_and_angles()[:3] / nL
    NL = nL ** 3
    size = np.array([nL]*3)
    m = np.zeros(size)
    pp = np.where(m==0)
    target_vts = dL * np.transpose(pp) + 1/4 * dL
    return target_vts


def select_voronoi_vertices(atoms, ip_vts, nvts, ds, rng, **dsargs):
    '''
    Select voronoi vertices by distribution strategies.
    '''
    nbins   = dsargs['nbins']
    density = dsargs['density']
    dmin    = dsargs['dmin']
    dmax    = dsargs['dmax']
    only_selected = dsargs['only_selected']
    if ds == 'random':
        ind = rng.choice(len(ip_vts), nvts, replace=False)
        target_vts = ip_vts[ind]
    elif ds == 'uniform':
        target_vts = uniform_distribute(atoms, ip_vts, nvts, 
                nbins, density, rng)
    elif ds == 'cluster':
        ind0 = rng.choice(len(ip_vts), 1)
        target_vts = cluster_distribute(atoms, ip_vts, ind0, nvts, dmin, dmax, rng)
    elif ds == 'block':
        target_vts = block_distribute(atoms, nvts)

    target_atoms = Atoms(positions=target_vts,cell=atoms.cell,pbc='111')
    if not only_selected:
        for atom in atoms:
            target_atoms.append(atom)
    target_vts = check_overlap(target_atoms, target_vts)
    target_atoms = Atoms(positions=target_vts,cell=atoms.cell,pbc='111')

    if check_mic_distance(target_atoms,dmin=dmin,dmax=dmax):
        dist_flag = False
    else:
        dist_flag = True 
        print('%i atoms are inserted'%len(target_vts))    
    return target_vts, dist_flag


def vor_insert(atoms, nsamples, insert_elem='X', only_selected=False,
        strategy='random', dmin=0.9, dmax=None, nbins=10, density=False,
        max_rins=None, cnst_rins=False, cnst_num=None, 
        seed=None, include_initial=False):
    assert atoms.pbc.all()
    rng = np.random.RandomState(seed)

    INVALID_VALUE = 'Invalid value for "max_rins", \
            specify a float in range of (0,1) or a str like "50%".'
    if max_rins is not None:
        if type(max_rins) == float:
            if not 0<max_rins<1.:
                raise ValueError(INVALID_VALUE)
            else:
                mr = max_rins
        else:
            try:
                mr = float(max_rins.split('%')[0])/100
            except:
                raise ValueError(INVALID_VALUE)
    else:
        mr = 0.1
    max_nvts = round(mr*len(atoms))
    if not cnst_rins: 
        nvts = rng.choice(max_nvts, 1)[0]+1
    else:
        nvts = max_nvts
    if type(cnst_num) == int: 
        nvts = cnst_num 

    rng = np.random.RandomState(seed)
    images = []
    if include_initial:
        images.append(atoms)
    cal, echo = 0, 0
    images_pos_sets = []

    print("Voronoi tessellation:",flush=True)
    vor = Voronoi(atoms.positions)
    all_vts = vor.vertices
    ip_vts = check_in_parapipe(atoms, all_vts)
    print("Voronoi candidates ready!",flush=True)

    while cal<nsamples:
        if echo > nsamples + 300:#perm(len(ip_vts),nvts):
            break
        target_vts, dist_flag = select_voronoi_vertices(atoms, 
                ip_vts, nvts, strategy, rng, 
                density=density, dmin=dmin, dmax=dmax, nbins=nbins, 
                only_selected=only_selected)
        if not dist_flag:
            echo += 1
            continue
        image = atoms.copy()
        for target_vt in target_vts:
            image.append(Atom(insert_elem,position=target_vt))
        image_set = set()
        for pos in image.positions:
            image_set.add(tuple(pos))
        echo += 1
        if image_set not in images_pos_sets:
            images.append(image)
            images_pos_sets.append(image_set)
            cal += 1
    return images


def main():
    import os
    import argparse
    from ase.io import read

    parser = argparse.ArgumentParser(
        description="Generate voronoi insertion(doping) configurations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add = parser.add_argument
    add("sample", type=str, help="name of the initial sample")
    add("nsamples", type=int, help="number of generated samples")
    add("-i","--insert_elem", type=str, help="the element of voronoi insertions")
    add("-mr","--max_rins", help="max rate of voronoi insertions(VI)")
    add("-cr","--cnst_rins", type=logical, help="whether set rate of VI in constant (equal to max_rins)")
    add("-cs","--cluster_size", type=float, help="diameter of cluster, usually for constraints or high density impurity generation")
    add("-sw","--skin_width", type=float, help="skin width of a layer-built sphere, \
            usually used for generation of surface configurations")
    add("-si","--skin_index", type=int, help="skin index of a layer-built sphere")
    add("--seed", help="seed for random generator")
    add("-inc","--include_initial", action="store_false", help="whether include the initial sample in outputs")
    add("--prefix", type=str, default="struct", metavar="prefix", help="prefix of the names of output samples")
    args = parser.parse_args()

    atoms = read(args.sample)
    images = vor_insert(atoms, args.nsamples, 
            insert_elem=args.insert_elem, max_rins=args.max_rins, 
            cnst_rins=args.cnst_rins,
            seed=args.seed, include_initial=args.include_initial)

    base, _ = os.path.split(args.prefix)
    if base != "" and not os.path.isdir(base):
        os.makedirs(base)

    for i, image in enumerate(images):
        fname = f"{args.prefix}_{i:05d}.vasp"
        image.write(fname, vasp5=True, sort=True)

    return


if __name__ == "__main__":
    main()

