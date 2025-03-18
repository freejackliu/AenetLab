"""
Run AIMD by a VaspAIMDCalc wrapper which works as-is with the native vasp executable   
"""


import numpy as np
from aenet.AenetLab.Calc.VaspCalc import VaspAIMDCalc
from aenet.AenetLab.aenet_io import load_nnfiles
from aenet.calculator import ANNCalculator
from aenet.opt import opt


def abinitio_MD(atoms, nsamples, vasp_mode='Langevin-NVT', 
        vasp_custm_confs=None, fast_relax_steps=None, 
        nn_opt=True, nn_path=None):
    # If you wanna to run the optimization, make sure the atoms is attached to a proper calc.
    if nn_opt:
        if nn_path:
            potentials = load_nnfiles
            calc = ANNCalculator(potentials=potentials)
            atoms.set_calculator(calc)
        opt_atoms = opt(atoms,cell_filter=False)
    else:
        opt_atoms = atoms.copy()
    images = VaspAIMDCalc(opt_atoms, nsamples, vasp_mode=vasp_mode, 
            vasp_custm_confs=vasp_custm_confs, 
            fast_relax_steps=fast_relax_steps)
    return images
            
"""
def classical_MD(atoms, nsamples, style='abinitio', mdalgo='Langevin',ensemble='NVT', mdconfs={},
         seed=None, include_initial=False):

    assert atoms.pbc.all()
    rng = np.random.RandomState(seed)
    images = []
 
    if include_initial:
        images.append(atoms)

    if len(atoms)<=10:
        ideal_atoms = atoms * (2,2,2)
    else:
        ideal_atoms = atoms.copy()

    cal = 0
    images_pos_sets = []
    while cal<nsamples:
        if not targets:
            target_id = rng.choice(len(ideal_atoms),1)[0]
        else:
            target_id = rng.choice(target_ids,1)[0]
        box_size = (box_range[1]-box_range[0])*np.random.ranf()+box_range[0]
        image = select_by_box(ideal_atoms,ideal_atoms.positions[target_id],box_size)
        if not len(image) > atoms_uplimit:
            box_vecs = np.array([[box_size,0.,0.],[0.,box_size,0.],[0.,0.,box_size]])
            image.cell = box_vecs
            image_set = set()
            for pos in image.positions:
                image_set.add(tuple(pos))
            if (image_set not in images_pos_sets) and check_mic_distance(image):
                images.append(image)
                images_pos_sets.append(image_set)
                cal += 1
    return images
"""
