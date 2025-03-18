from ase.optimize import LBFGSLineSearch
from ase.io.trajectory import Trajectory
from ase.constraints import ExpCellFilter
from aenet.select import get_index
from aenet.AenetLab.aenet_io import load_nnfiles

# If you wanna to run the optimization, make sure the atoms is attached to a proper calc.
def opt(atoms, cell_filter=False, traj_name='opt.traj',
        interval=5, fmax=0.05, steps=50, index=-1):
    traj = Trajectory(traj_name,'w',atoms)
    if cell_filter:
        ecf = ExpCellFilter(atoms)
        dyn = LBFGSLineSearch(ecf)
    else:
        dyn = LBFGSLineSearch(atoms)
    dyn.attach(traj,interval=interval)
    dyn.run(fmax=fmax,steps=steps)
    traj.close()
    traj = Trajectory(traj_name)
    opt_atoms = traj[index]
    return opt_atoms


def dft_opt(atoms, nsamples, nn_opt=True, nn_path=None, 
        cell_filter=False, seed=None, include_initial=False):
    if nn_opt:
       if nn_path:
           potentials = load_nnfiles(nn_path)
           calc = ANNCalculator(potentials=potentials)
           atoms.set_calculator(calc)
           images_pool = opt(atoms,cell_filter=cell_filter,
                   index=slice(0,None,1))
    else:
       images_pool = [atoms.copy()]
    images = []
    selected = get_index([0,len(images_pool)],nsamples)
    if include_initial and nn_opt:
        images.append(atoms)
    for i in selected:
        images.append(images_pool[i])
    return images
