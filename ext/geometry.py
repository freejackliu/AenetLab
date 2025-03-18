import numpy as np
#from math import *
from aenet.find_mic import get_all_distances
from ase import Atom, Atoms
from functools import reduce


def compute_squared_edm(x):
    x = np.abs(x)
    m, n = x.shape
    g = np.dot(x.T, x)
    h = np.tile(np.diag(g), (n, 1))
    return h + h.T - 2*g


def _lcm(a,b):
    bb = max(a,b)
    aa = min(a,b)
    if bb%aa == 0:
        return aa
    return _lcm(bb%aa,aa)


def _gcd(a,b):
    lcm = _lcm(a,b)
    return int(a*b/lcm)


def rep2cube(atoms, natoms=100):
    cl = atoms.get_cell_lengths_and_angles()[:3]
    cl_int = np.rint(cl)
    cl_gcd = reduce(_gcd, cl_int)
    tmp_ext = cl_gcd / cl_int
    replicate = np.rint( tmp_ext * (natoms / len(atoms) / np.prod(tmp_ext)) ** (1/3))
    replicate = np.maximum(replicate, 1) 
    replicate = np.array(replicate.tolist(), dtype=int)
    return replicate


def compute_gr_peaks(atoms, rdelta=0.01, threshold=0):
    rep = rep2cube(atoms)
    atoms = atoms.repeat(rep)
    cell_lengths = atoms.get_cell_lengths_and_angles()[:3]
    MAXBIN = int(np.min(cell_lengths) / 2.0 / rdelta)
    rhototal = len(atoms) / atoms.get_volume()

    grresults = np.zeros(MAXBIN)
    dm = atoms.get_all_distances(mic=True)
    for ind, dm_i in enumerate(dm):
        distance = np.delete(dm_i,ind)
        Countvalue, BinEdge = np.histogram(distance, bins=MAXBIN, range=(0, MAXBIN * rdelta))
        grresults += Countvalue
    binleft = BinEdge[:-1]
    binright = BinEdge[1:]
    Nideal = 4.0 / 3 * np.pi * (binright**3 - binleft**3)
    grresults = grresults / len(atoms) / (Nideal * rhototal)
    bincen = binleft + 0.5 * rdelta

    tmp_diff = np.diff(grresults) < 0
    peak_r_list, peak_v_list = [], []
    for ind in range(len(tmp_diff)-1):
        if (not tmp_diff[ind]) and tmp_diff[ind+1]:
            if grresults[ind+1] > threshold:
                peak_r_list.append(bincen[ind+1])
                peak_v_list.append(grresults[ind+1])
    return peak_r_list, peak_v_list


def check_in_parapipe(atoms, vts):
    ipvts = []
    shift = np.min(atoms.positions,0)
    atoms_c = atoms.copy()
    atoms_c.positions -= shift
    scaled_vts = atoms_c.cell.scaled_positions(vts)
    for i, scaled_vt in enumerate(scaled_vts):
        ck_in = np.logical_or(scaled_vt < -0.1, scaled_vt > 1.1)
        if not ck_in.sum():
            ipvts.append(vts[i])
    ipvts = np.array(ipvts)
    return ipvts


def check_overlap(atoms, vts, dcr=0.001):
    _, d = get_all_distances(atoms.positions, atoms.cell)
    size = len(d)
    overlap_index = set()
    ipvts = []
    for i in range(size):
        for j in range(i+1,size):
            if d[i][j] < dcr:
                overlap_index.add(j)   
    for i, vt in enumerate(vts):
        if i not in overlap_index:
            ipvts.append(vts[i])
    ipvts = np.array(ipvts)
    return ipvts


def check_in_sphere(vts, posori, rs):
    ipvts = []
    d_tmp = []
    d_list = [np.linalg.norm(vts[i]-posori) for i in range(len(vts))]
    for ind, d in enumerate(d_list):
        if d <= rs:
            ipvts.append(vts[ind]) 
            d_tmp.append(d)
    ipvts = np.array(ipvts)
    rss = max(d_tmp)
    return rss, ipvts


def check_mic_distance(atoms,dmin=1.2,dmax=None):
    _, d = get_all_distances(atoms.positions, atoms.cell)
    size = len(d)
    rmax = np.max(atoms.get_cell_lengths_and_angles()[:3])/2
    # Non-zero diagonal elements to check the real minima of distances
    darr = [d[i][j] for i in range(size) for j in range(size) if i!=j]
    if dmax is not None:
        drange = [float(dmin),np.min([dmax,rmax])]
    else:
        drange = [float(dmin),rmax]
    print("Natoms : %i"%len(d),"Mindist : %.3f"%np.min(darr),"Maxdist : %.3f"%(np.max(darr)/2),flush=True)
    if np.min(darr) < drange[0] or np.max(darr)/2 > drange[1]:
        min_d_flag = True
    else:
        min_d_flag = False
    return min_d_flag


def check_nomic_distance(atoms,riso=4):
    dmat = np.sqrt(compute_squared_edm(atoms.positions.T))
    dmat += np.diag([atoms.get_cell_lengths_and_angles()[:3].max()]*len(dmat))
    dmin = np.min(dmat,axis=1)
    if dmin.max() > riso:
        iso_flag = True
    else:
        iso_flag = False
    return iso_flag


def shift_positions(atoms, target_index=None, target_position=None):
    """Shift positions to a target atom or position.
       Only valid for orthogonal cell.

       Parameters:

        atoms (AseAtoms obj) : 
           Atoms obj with pbc=[1,1,1] and zero-non-diagonal cell matrix.

        target_index (int):
           Index id for the target atom. 

        target_position (numpy.ndarray):(prior)
           Scaled target position.
    """
    
    out_atoms = atoms.copy()

    cell_angles = atoms.get_cell_lengths_and_angles()
    cell = cell_angles[:3]
    angles = cell_angles[3:]
    assert np.sum(angles-90)<1.e-9 and atoms.pbc.all()

    no_para_flag = 1
    if target_index is not None:
        target_vec = atoms[target_index].position
        no_para_flag = 0
    if target_position is not None:
        target_vec = target_position * cell
        no_para_flag = 0
    if no_para_flag:
        raise ValueError("'target_index' or 'target_position' argument shall be specified!")
    
    shift = np.array([0.5,0.5,0.5]) * cell - target_vec
    positions = atoms.positions + shift
    positions %= cell
    out_atoms.positions = positions

    return out_atoms


def shift_mass_center(atoms,mass_mask=True):
    tmp_atoms = atoms.copy()
    if mass_mask:
        tmp_atoms.symbols = 'H'*len(atoms)
    scaled_center = tmp_atoms.get_center_of_mass(scaled=True)
    out_atoms = shift_positions(atoms, target_position=scaled_center)
    return out_atoms


def modify_vacuum_width(atoms, vacuum_width=20):
    ''' 
    Decrease vacuum layer width to avoid high SW planning cost;
    Increase vacuum layer width to meet appr-pdb condition for nano-particles(clusters)
    '''                                                                                 
    tmp_atoms = shift_mass_center(atoms)
    min_set = tmp_atoms.positions.min(0)
    max_set = tmp_atoms.positions.max(0)
    ort_set = max_set - min_set
    new_cell = np.diag(ort_set + vacuum_width)
    tmp_atoms.cell = new_cell
    out_atoms = shift_mass_center(tmp_atoms)
    return out_atoms


def radial_dist(atoms, elem=None, index=None):
    """Radial distribution of specified elem or particle index.

       Parameters:
           
        atoms (AseAtoms obj): 
           Atoms obj with pbc=[1,1,1].

        elem (str):
           Symbol of target particles.

        index (int/slice/numpy.ndarray):(prior)
           Index of target particles.

    """
    ip_vts = []
    if elem is not None:
        for atom in atoms:
            if atom.symbol == elem:
                ip_vts.append(atom.position)
    if index is not None:
        ip_vts = atoms.positions[index]
    ip_vts = np.asarray(ip_vts)
    center_pos = atoms.cell.sum(0)/2
    d = np.sqrt(np.sum((ip_vts - center_pos) ** 2, axis=1))
    return d
