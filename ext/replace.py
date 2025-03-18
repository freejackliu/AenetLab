import numpy as np
from math import *
from scipy.spatial import ConvexHull


def get_surface_points(atoms):
    hull = ConvexHull(atoms.positions)
    return hull.vertices


def command_interpreter(symbols, command_line):
    symbols = list(symbols)
    elems_set = set(symbols)
    Natoms = len(symbols)
    command_words = command_line.strip().split()
    if len(command_words) % 2:
        raise Exception("Commands for elements loading should be specified as the following format: 'elem1 num1/rate1/auto/base elem2 num2/rate2/auto/base ...'.")
    else:
        ns_old, ns = {}, {}
        nnow = 0
        elem_auto, elem_base = [], []
        elems = command_words[slice(0,None,2)]
        values = command_words[slice(1,None,2)]
        elems_values = dict(zip(elems,values))
        all_elems_values = elems_values.copy()
        sp_elems_values = elems_values.copy()
        for elem, value in elems_values.items():
            ns_old[elem] = symbols.count(elem)
            if value == 'auto':
                del sp_elems_values[elem]
                if ns_old[elem]:
                    elem_auto.append(elem)
                else:
                    del all_elems_values[elem]
            elif value == 'base':
                del sp_elems_values[elem]
                if ns_old[elem]:
                    elem_base.append(elem)
                else:
                    del all_elems_values[elem]
        elems = list(sp_elems_values.keys())
        values = list(sp_elems_values.values())
        elems_copy = list(all_elems_values.keys())
 
        #Specific arguments
        if not len(elems):
            raise ValueError("You need to specify at least one specific argument.")
        for elem, value in zip(elems,values):
            vsp = value.split('%')
            if len(vsp) == 1:
                nelem = int(value)
            elif len(vsp) == 2:
                nelem = floor(int(vsp[0])*Natoms/100)
            ns[elem] = nelem
            nnow += nelem
        #'base' arguments
        if len(elem_base):
            np.random.shuffle(elem_base)
            nrem = Natoms - nnow
            nbase = 0
            for i in range(len(elem_base)-1):
                ns[elem_base[i]] = ns_old[elem_base[i]]
                nbase += ns_old[elem_base[i]]
            if len(elem_base) == len(elems_copy)-1:
                ns[elem_base[-1]] = nrem - nbase
            elif len(elem_base) < len(elems_copy)-1:
                ns[elem_base[-1]] = ns_old[elem_base[-1]]
                nbase += ns_old[elem_base[-1]]
            nnow += nbase
        #'auto' arguments
        if len(elem_auto):
            np.random.shuffle(elem_auto)
            nrem = Natoms - nnow
            for i in range(len(elem_auto)-1):
                nelem = np.random.randint(nrem)
                ns[elem_auto[i]] = nelem
                nrem -= nelem
            ns[elem_auto[-1]] = nrem
        return ns_old, ns


def atoms_replace(atoms, rng, style, **kwargs):
    Natoms = len(atoms)
    atoms_index = set(np.arange(Natoms))
    atoms_symbols = list(atoms.symbols)
    if kwargs['command'] != '':
        ns_old, ns = command_interpreter(atoms_symbols, kwargs['command'])
    else:
        raise ValueError("'command' can not be empty")

    index_old = {}
    for elem in ns_old.keys():
        index_old[elem] = []            
    for li in atoms_index:
        elem = atoms_symbols[li]
        index_old[elem].append(li)

    cl = []
    if kwargs['inner'] != []:
        surface_index = set(get_surface_points(atoms))
        inner_index = atoms_index - surface_index
        for inner_elem in kwargs['inner']:
            real_inner = rng.choice(np.array(list(inner_index)), 
                                ns[inner_elem])
            cl += index_old[inner_elem]
            atoms.symbols[real_inner] = inner_elem
            inner_index -= set(real_inner)
            del ns[inner_elem]
        last_index = surface_index | inner_index
    else:
        last_index = set([i for i in range(Natoms)])
    if style == 'random':
        last_sequence = []
        for ns_k, ns_v in ns.items():
            last_sequence += [ns_k] * ns_v
        rng.shuffle(last_sequence)
        atoms.symbols[np.array(list(last_index))] = last_sequence
    elif style == 'normal':
        change = [ns[elem] - ns_old[elem] for elem in ns.keys()]
        for elem, cn in zip(ns.keys(),change):
            if cn < 0:
                tmp_index = rng.choice(np.arange(ns_old[elem]),abs(cn),replace=False)
                cl += [index_old[elem][i] for i in tmp_index]
        for elem, cn in zip(ns.keys(),change):
            if cn > 0:
                for i in range(cn):
                    index_pick = rng.choice(cl,1)[0]
                    cl.remove(index_pick)
                    atoms.symbols[index_pick] = elem
    return atoms


def replace(atoms, nsamples, style='normal', command='', inner=[],
            seed=None, include_initial=False):
    rng = np.random.RandomState(seed)
    assert atoms.pbc.all()
    images = []
    if include_initial:
        images.append(atoms)
    for _ in range(nsamples):
        image = atoms.copy()
        image = atoms_replace(image, rng, 
                style=style, command=command, inner=inner)
        images.append(image)
    return images


def main():
    import os
    import argparse
    from ase.io import read

    parser = argparse.ArgumentParser(
        description="Generate alloy structures with different ratios.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    add = parser.add_argument
    add("nsamples", type=int, help="number of generated random samples")
    add("sample", type=str, help="sample file to be randomized")
    add("-c", "--command", help="command line for ratio settings")
    add("-i", "--inner", nargs='+', help="list of inner elements")
    add("--include_initial", action="store_false", help="whether include the initial sample in outputs")
    add("--prefix", type=str, default="struct", metavar="prefix", help="prefix of the names of output samples")
    args = parser.parse_args()

    atoms = read(args.sample)
    images = replace(atoms, args.nsamples, 
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

