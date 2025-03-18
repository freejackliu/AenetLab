import numpy as np
from ase import Atom
from scipy.spatial import ConvexHull
from geometry import modify_vacuum_width


def get_normal_vector(facet_vertices):
    '''
    Get a unit normal vector by right-hand rule. Note that the facet vertices should be specified in a counter clockwise order.
    '''
    a = facet_vertices[1] - facet_vertices[0]
    b = facet_vertices[2] - facet_vertices[1]
    m = np.array([a,b])
    nv = []
    rank_ab = np.linalg.matrix_rank(m)
    if rank_ab == 1:
        raise Exception("Collineation Error!")
    for i in range(3):
        nv_factor = (-1) ** i
        nv_submat = np.delete(m, i, axis=1)
        nv.append(nv_factor * np.linalg.det(nv_submat))
    nv = nv / np.linalg.norm(nv)
    return nv


def get_tri_facet_area(facet_vertices):
    '''
    Get an area of certain triangular facet from a convex hull.
    '''
    l = []
    for i in range(3):
        for j in range(i+1,3):
            l.append(np.linalg.norm(facet_vertices[i]-facet_vertices[j]))
    p = np.sum(l) / 2
    S = np.sqrt(p * (p-l[0]) * (p-l[1]) * (p-l[2]))
    return S


def get_tri_facet_center(facet_vertices):
    '''
    Get a center of certain triangular facet from a convex hull.
    '''
    return np.sum(facet_vertices,axis=0) / 3


def conv_insert(atoms, nsamples, dmin = 1.0, 
            insert_elem='X', insert_number=1, rattle_size=1.0, 
            area_range=(2.0,4.0), vacuum_width=10, 
            seed=None, include_initial=False):

    # Pick proper facets for insertion
    sp_center = atoms.get_cell_lengths_and_angles()[:3] / 2
    hull = ConvexHull(atoms.positions)
    pick = []
    for simplice in hull.simplices:
        if len(simplice) == 3:
            vertices = atoms.positions[simplice]
            s = get_tri_facet_area(vertices)
            center = get_tri_facet_center(vertices)
            ratvec = get_normal_vector(vertices)
            new_positions = np.array([center + rattle_size * ratvec,
                                 center - rattle_size * ratvec])
            d2s = np.sum((new_positions - sp_center) ** 2, axis =1)
            new_position = new_positions[d2s.argmax()]
            d2 = np.sum((vertices - center) ** 2, axis=1)
            if area_range[0] <= s <= area_range[1] and d2.min() > dmin ** 2:
                pick.append(new_position)
    pick = np.array(pick)
    Npick = len(pick)
    if nsamples > Npick:
        raise ValueError("Number of wanted samples is over the size of picked ones")

    # Generate structures by rattling insertion from facet centers
    rng = np.random.RandomState(seed)
    images = []
    if include_initial:
        images.append(atoms)
    for _ in range(nsamples):
        image = atoms.copy()
        ips = rng.choice(np.arange(Npick), insert_number, replace=False)
        for ip in ips:
            new_position = pick[ip]
            image.append(Atom(symbol=insert_elem, position=new_position))
            image = modify_vacuum_width(image, vacuum_width=vacuum_width)
        images.append(image)
    return images


def main():
    import os
    import argparse
    from ase.io import read

    parser = argparse.ArgumentParser(
        description="Generate structures with feasible adsorption sites.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add = parser.add_argument
    add("nsamples", type=int, help="number of generated random samples")
    add("sample", type=str, help="path to the initial file")
    add("element", type=str, help="element to be inserted")
    add("-n", "--ninsert", type=int, help="number of atoms to be inserted")
    add("-r", "--rdisp", type=float, default=1, metavar="r", help="size of displacements from convex hull facet centers")
    add("--dmin", type=float, default=1.0, help="min size of distances from vertices to the insertion position")
    add("--area_range", type=float, nargs='+', metavar="area_min area_max",
        help="range of facet area for random choices")
    add("--include_initial", action="store_true", help="whether include the initial sample in outputs")
    args = parser.parse_args()

    atoms = read(args.sample)
    images = conv_insert(atoms, args.nsamples, 
                     insert_elem=args.element, insert_number=args.ninsert,
                     rattle_size=args.rdisp, area_range=args.area_range,
                     include_initial=args.include_initial)

    if not os.path.isdir("cluster"):
        os.makedirs("cluster")

    for i, image in enumerate(images):
        fname = f"cluster/cluster_{i:05d}.cfg"
        image.write(fname)
    return


if __name__ == "__main__":
    main()

