"""
Prediction of pertubated diatomic-bond energies from the 1D taylor expnsion.
"""


def taylor_first_dimer(atoms, dr):
    """Generate one dimension perturbated configurations with energies 
    predicted by the first-order Taylor expansion.

    Usually used as a simple test for incorporating force infomations to ANN
    training procedure. A more general version for Taylor expansion is available,
    see 'taylor.py' for more details.

    Arguments:
        atoms    :   Atoms object(diatomic configuration without periodic boundary)
        dr       :   displacement of the bond length 

    Return:
        A list of Atoms objects with predicted energy and forcs.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    try:
        energy = atoms.calc.results['energy']
        forces  = atoms.calc.results['forces']
    except KeyError:
        print("AtomsError: Key values of 'energy' and 'forces' attached to Atoms can't be None, please read xsf files by 'xsf.py' in 'aenet-path/python/aenet' or use ase.calculators.get_potential_energy method to get the exact values")
    else:
        if len(atoms)==2 and (not atoms.pbc.all()):
            samples = []
            delta_e = -forces[1][2]*(-dr)
            a_left = atoms.copy()
            a_left[1].position -= dr
            a_left.calc = SinglePointCalculator(a_left,energy=energy + delta_e,forces=forces)
            samples.append(a_left)
            delta_e = -forces[1][2]*dr
            a_right = atoms.copy()
            a_right[1].position += dr
            a_right.calc = SinglePointCalculator(a_right,energy=energy + delta_e,forces=forces)
            samples.append(a_right)
            return samples

