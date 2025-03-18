from __future__ import print_function, division

import sys
import warnings

import mpi4py
import numpy as np
import numba as nb

from ase.units import kB
from ase.io.trajectory import Trajectory
from ase.calculators.calculator import Calculator
from ase.utils import basestring
from ase.parallel import world, broadcast
from ase.utils import devnull


__all__ = ["BasinHopping"]


@nb.jit('f8[:,:](f8[:], f8[:,:], f8, f8, i8)')
def _fast_tfmc_step(masses, forces, drmax, T, seed):
    """
    Jitted version of the tfMC stepping function.
    """
    np.random.seed(seed)
    natoms = len(masses)
    displacements = np.zeros_like(forces)
    mass_min = masses.min()
    for i in range(natoms):
        dr = drmax*(mass_min/masses[i])**0.25
        for j in range(3):
            P_acc = 0.0
            P_ran = 1.0
            gamma = forces[i][j]*dr/(2.0*T)
            gamma_exp = np.exp(gamma)
            gamma_expi = 1.0/gamma_exp
            # generate displacements according to the tfMC distribution
            while P_acc < P_ran:
                xi = 2.0*np.random.ranf() - 1.0
                P_ran = np.random.ranf()
                if xi < 0:
                    P_acc = np.exp(2.0*xi*gamma)*gamma_exp - gamma_expi
                    P_acc /= gamma_exp - gamma_expi
                elif xi > 0:
                    P_acc = gamma_exp - np.exp(2.0*xi*gamma)*gamma_expi
                    P_acc /= gamma_exp - gamma_expi
                else:
                    P_acc = 1.0
            # displace
            displacements[i][j] += xi*dr
    return displacements


class BasinHopping(object):
    """
    Basin hopping method from David J. Wales.
    """
    def __init__(self, atoms, calc, local_optimizer,
                 seed=None, temperature=1000*kB,
                 force_consistent=True,
                 move='gaussian',
                 fix_com=False, fix_rot=False,
                 dr=0.2, dr_adjust=0.02, n_dr_adjust=10,
                 drmin=0.001, drmax=0.6, target_ratio=0.5,
                 tfmc_temperature=None, tfmc_steps=1,
                 avoid_overlap=0.0, local_move=False,
                 swapping=False, nswapping=100, swapping_elements=[],
                 jumping=False, jump_max=20,
                 scale_cell=False, cell_step=0.001,
                 comm=world, txt='-', prefix='bh'):
        """
        Global structure search using the basin hopping method.

        Arguments:
            atoms: (Atoms) atoms object to be optimized.
        """
        self.atoms = atoms.copy()

        if isinstance(calc, dict):
            self.calc = calc['name'](*calc['args'], **calc['kwargs'])
        else:
            self.calc = calc
        
        self.atoms.set_calculator(self.calc)
        
        if isinstance(local_optimizer, dict):
            self.local_optimizer = local_optimizer['name']
            self.local_optimize_fmax = local_optimizer['fmax']
            self.local_optimize_steps = local_optimizer['steps']
            for i in ['name', 'fmax', 'steps']:
                local_optimizer.pop(i, None)
            if 'logfile' not in local_optimizer.keys():
                local_optimizer['logfile'] = devnull
            self.local_optimizer_kwargs = local_optimizer
        else:
            raise ValueError('Local optimizer must be a dict!')
        
        # Random state
        self.rng = np.random.RandomState(seed)
        
        # MC temperature
        self.temperature = temperature
        if tfmc_temperature is None:
            self.tfmc_temperature = self.temperature
        else:
            self.tfmc_temperature = tfmc_temperature
        self.tfmc_steps = tfmc_steps

        # Using force consistent energy?
        self.force_consistent = force_consistent

        # MC movements
        mapping = {
            'gaussian': self.gaussian,
            'uniform': self.uniform,
            'tfmc': self.tfmc,
        }
        try:
            self.move = mapping[move.lower()]
        except IndexError:
            raise ValueError('Invalid atom displace method.')

        self.scale_cell = scale_cell
        self.cell_step = cell_step

        # Fix mass center or rotation
        self.fix_com = fix_com
        self.fix_rot = fix_rot

        # Maxium step size of atom displacements
        self.dr = dr
        self.n_dr_adjust = n_dr_adjust
        self.dr_adjust = dr_adjust
        self.drmax = drmax
        self.drmin = drmin
        self.target_ratio = target_ratio

        # Avoid atom collisions
        self.avoid_overlap = avoid_overlap

        # Move one atom at a time
        self.local_move = local_move

        # Swap atom symbols
        self.swapping = swapping
        self.nswapping = nswapping
        if len(swapping_elements) < 2:
            self.swapping_elements = set(self.atoms.get_chemical_symbols())
        else:
            self.swapping_elements = set(swapping_elements)

        # Occasionally jumping
        self.jumping = jumping
        self.jump_max = jump_max

        # Communicator
        self.comm = comm
        self.rng = broadcast(self.rng, root=0, comm=self.comm)

        # output
        self.own_txt = False
        if isinstance(txt, basestring):
            if txt == '-':
                self.txt = sys.stdout
            else:
                self.txt = open(txt, 'w')
                self.own_txt = True
        else:
            self.txt = txt
        
        self.prefix = prefix

        self.local_minimum_traj = \
            Trajectory('{}_local.traj'.format(self.prefix), mode='w')

        self.lowest_miniumum_trajectory = \
            Trajectory('{}_lowest.traj'.format(self.prefix), mode='w')

        # initialize MC search states        
        self.atoms_best = self.atoms.copy()
        self.atoms_prev = self.atoms.copy()

        self.pe_best = self.atoms.get_potential_energy(
            force_consistent=self.force_consistent
        )
        self.pe_prev = self.pe_best
        self.pe_curr = self.pe_best

        self.istep = 0
        self.accept_ratio = 0
    
    def __del__(self):
        if self.own_txt:
            self.txt.close()
        self.local_minimum_traj.close()
        self.lowest_miniumum_trajectory.close()

    def run(self, steps):
        self.naccepted = 0
        self.accept_ratio = 0
        self.nswapped = 0

        for i in range(steps):
            self.istep = i

            self.move(self.dr)
            do_swap = self.swapping and (i % self.nswapping == 0)
            if do_swap:
                self.swap()
            if self.scale_cell:
                c = self.atoms.cell.copy()
                c *= self.rng.uniform(-1, 1) * self.cell_step * c
                self.atoms.set_cell(c)
            self.pe_curr = self.relax()

            if self.pe_curr < self.pe_prev:
                self.atoms_prev = self.atoms.copy()
                self.pe_prev = self.pe_curr
                if self.pe_curr < self.pe_best:
                    self.pe_best = self.pe_curr
                    self.atoms_best = self.atoms.copy()
                    self.lowest_miniumum_trajectory.write(self.atoms_best)
                self.naccepted += 1
                if do_swap:
                    self.nswapped += 1
                self.local_minimum_traj.write(self.atoms_prev)
            elif self.metropolis(self.pe_curr, self.pe_prev, self.temperature):
                self.atoms_prev = self.atoms.copy()
                self.pe_prev = self.pe_curr
                self.naccepted += 1
                if do_swap:
                    self.nswapped += 1
                self.local_minimum_traj.write(self.atoms_prev)
            else:
                old_positions = self.atoms_prev.get_positions()
                old_numbers = self.atoms_prev.get_atomic_numbers()
                self.atoms.set_positions(old_positions)
                self.atoms.set_atomic_numbers(old_numbers)

            self.adjust_dr()
            if self.comm.rank == 0:
                self.log()

    def swap(self):
        ntrials = 1000
        for i in range(ntrials):
            pair = self.rng.choice(len(self.atoms), 2)
            a, b = pair[0], pair[1]
            if self.atoms[a].number != self.atoms[b].number:
                self.atoms[b].number, self.atoms[a].number = \
                self.atoms[a].number, self.atoms[b].number
                break
        if i == ntrials:
            warnings.warn("Max trials for atom swapping reached.")
    
    def gaussian(self, dr):
        """
        Generate Gaussian (normal) distributed atom displacements.
        """
        displacements = self.rng.normal(0, dr, size=(len(self.atoms), 3))
        self.displace(self.atoms,
                      displacements,
                      local_move=self.local_move,
                      avoid_overlap=self.avoid_overlap)
    
    def uniform(self, dr):
        """
        Generate uniform distributed atom displacements.
        """
        displacements = self.rng.uniform(-dr, dr, size=(len(self.atoms), 3))
        self.displace(self.atoms,
                      displacements,
                      local_move=self.local_move,
                      avoid_overlap=self.avoid_overlap)
    
    def tfmc(self, dr):
        """
        Generate atom displacements according to the tfMC algorithm.
        """
        for i in range(self.tfmc_steps):
            masses = self.atoms.get_masses()
            forces = self.atoms.get_forces()
            seed = self.rng.choice(2**32 - 1)
            temperature = self.tfmc_temperature
            try:
                displacements = _fast_tfmc_step(masses, forces, dr, temperature, seed)
            except ZeroDivisionError:
                displacements = np.zeros_like(forces)
            self.displace(self.atoms,
                          displacements,
                          local_move=self.local_move,
                          avoid_overlap=self.avoid_overlap)

    def displace(self, atoms, displacements, local_move=False, avoid_overlap=0.0):
        """
        Set atom positions based on constraints.
        """
        masses = atoms.get_masses()
        com = atoms.get_center_of_mass()
        positions = atoms.get_positions()

        if local_move:
            index = self.rng.choice(displacements.shape[0])
            t = displacements.copy()
            t[index] = 0.0
            displacements -= t
        
        if avoid_overlap:
            pass  # TODO: add avoid atoms overlapping logic

        if self.fix_com:
            com_disp = np.sum(masses[:,np.newaxis]*displacements, axis=0)/np.sum(masses)
            displacements -= com_disp
        
        if self.fix_rot:
            pos = positions - com
            inertia, basis = atoms.get_moments_of_inertia(vectors=True)
            mom = masses[:,np.newaxis]*displacements
            angmom = np.dot(basis, np.cross(pos, mom).sum(axis=0))
            omega = np.dot(np.linalg.inv(basis), np.select([inertia > 0], [angmom/inertia]))
            displacements -= np.cross(omega, pos)

        positions += displacements
        self.comm.broadcast(positions, 0)
        atoms.set_positions(positions)
    
    def metropolis(self, ecurr, eprev, T):
        return np.exp(-(ecurr - eprev)/T) > self.rng.uniform()

    def relax(self):
        self.nopt = np.nan
        positions = self.atoms.get_positions().copy()
        try:
            opt = self.local_optimizer(self.atoms, **self.local_optimizer_kwargs)
            opt.run(fmax=self.local_optimize_fmax, steps=self.local_optimize_steps)
            pe = self.atoms.get_potential_energy(force_consistent=self.force_consistent)
            displacements = self.atoms.get_positions() - positions
            self.nopt = opt.get_number_of_steps()
        except:
            # Something goes wrong
            pe = np.nan
            displacements = np.zeros_like(positions)
        # Temporary restore the atom positions
        self.atoms.set_positions(positions)
        # Now do the real stuff, aware of the fix_cm and fix_rot
        self.displace(self.atoms, displacements)
        return pe

    def adjust_dr(self):
        if self.istep % self.n_dr_adjust == 1:
            self.accept_ratio = 1.0*self.naccepted/(self.istep + 1)
            if self.accept_ratio < self.target_ratio:
                self.dr = max(self.dr - self.dr_adjust, self.drmin)
            else:
                self.dr = min(self.dr + self.dr_adjust, self.drmax)
    
    def log(self):
        header = "step pe_curr pe_prev pe_best dr ratio nopt nswap\n"
        if self.istep == 0:
            self.txt.write(header)
        info = "{:6d} {:12.6f} {:12.6f} {:12.6f} {:6.2f} {:6.2f} {:6d} {}\n"
        info = info.format(
            self.istep,
            self.pe_curr,
            self.pe_prev, 
            self.pe_best, 
            self.dr, 
            self.accept_ratio, 
            self.nopt, 
            self.nswapped
        )
        self.txt.write(info)
        self.txt.flush()


if __name__ == "__main__":
    from ase.io import read
    from ase.optimize import MDMin, LBFGS, FIRE, BFGSLineSearch
    from ase.utils import devnull
    from aenet.calculator import ANNCalculator
    
    tag = sys.argv[1]
    atoms = read("../gen/" + tag + '.cfg')
    potentials = {'Cu': '../nnpots/Cu.30h-30h.nn',
                  'Zr': '../nnpots/Zr.30h-30h.nn'} 
    
    calc = ANNCalculator(potentials)

    opt = {
        'name': BFGSLineSearch,
        'fmax': 0.06,
        'steps': 200,
    }

    do_swap = False
    if len(set(atoms.get_chemical_symbols())) > 1:
        do_swap = True

    bh = BasinHopping(atoms, calc, opt,
                      force_consistent=False,
                      move='tfmc',
                      tfmc_temperature=800*kB,
                      tfmc_steps=10,
                      swapping=do_swap,
                      nswapping=1,
                      temperature=3000*kB,
                      fix_com=True,
                      fix_rot=True,
                      dr=0.6, drmin=0.2, drmax=1.0,
                      n_dr_adjust=2, dr_adjust=0.01,
                      prefix=tag)
    bh.run(10000)
