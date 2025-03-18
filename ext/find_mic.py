import numpy as np
from ase.cell import Cell


def naive_find_mic(v, cell):
    """
    Finds the minimum-image representation of vector(s) v.
    Safe to use for (pbc.all() and 
    (norm(v_mic) < 0.5 * min(cell.lengths()))).
    Can otherwise fail for non-orthorhombic cells.

    Described in:
    W. Smith,"The Minimum Image Convention in Non-Cubic MD Cells", 1989, 
    http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.57.1696.
"""
    f = cell.scaled_positions(v)
    f -= np.floor(f + 0.5)
    vmin = f @cell
    vlen = np.linalg.norm(vmin, axis=1)
    return vmin, vlen


def find_mic(v, cell):
    """
    Only safe for the all-pbc orthogonal cell! For tiled one, Minkowski reduction is more safe but relatively slow.
    """
    v = np.asarray(v)
    single = v.ndim == 1
    v = np.atleast_2d(v)
    vmin, vlen = naive_find_mic(v, cell)
    if single:
        return vmin[0], vlen[0]
    else:
        return vmin, vlen


def get_distances(p0,p1,cell):
    D = p1 - p0
    D, D_len = find_mic(D, cell=cell)
    return D, D_len


def get_all_distances(p1,cell):
    p1 = np.atleast_2d(p1)
    np1 = len(p1)
    ind1, ind2 = np.triu_indices(np1, k=1)
    D = p1[ind2] - p1[ind1]
    D, D_len = find_mic(D, cell=cell)
    Dout = np.zeros((np1, np1, 3))
    Dout[(ind1, ind2)] = D
    Dout -= np.transpose(Dout, axes=(1, 0, 2))

    Dout_len = np.zeros((np1, np1))
    Dout_len[(ind1, ind2)] = D_len
    Dout_len += Dout_len.T
    return Dout, Dout_len
