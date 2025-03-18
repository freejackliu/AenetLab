import os
import shutil
import inspect
from math import *

from ase.io import read,write
from ase.calculators.vasp import Vasp
from ase.dft.kpoints import monkhorst_pack

from aenet.xsf import write_xsf,read_xsf
from aenet.AenetLab.aenet_io import *
from aenet.geometry import check_in_parapipe, check_nomic_distance

VASP_IOFILES = ['ase-sort.dat','CHGCAR','DOSCAR','INCAR','OSZICAR',
                 'PCDAT','POTCAR','vasprun.xml','XDATCAR','CHG','CONTCAR',
                 'EIGENVAL','OUTCAR','POSCAR','REPORT','WAVECAR','IBZKPT',
                 'KPOINTS','vasp.out','ICONST']
FILES_NEEDED = ['vasprun.xml','OUTCAR','DOSCAR']


def check_error_in_vaspout(vaspout):
    with open(vaspout,'r') as vpo:
        lines = vpo.readlines()
        io_stat = 0
        for line in lines:
            if line.strip().split()[:2] == 'ERROR FEXCP:':
                print("Warning: Check your 'ENCUT' in the Vasp INCAR file! It's too small for xc-table construction. You can modify the source code (path/to/AenetLab/Calc/VaspCalc.py) to customize your own Vasp settings")
                io_stat = 4
        return io_stat


def trig_run(atoms):
    try:
        atoms.get_potential_energy()
        final_atoms = read('vasprun.xml',index=-1)
    except RuntimeError:
        print("Warning: An ionic relaxation failed due to a bad initial structure , as this creation iteration will be passed")
        io_stat = 1
    except ValueError:
        print("Warning: ASE Vasp Calculator can't load such large forces while running, this creation iteration will be passed")
        io_stat = 2
    except:
        print("Warning: ASE Vasp Calculator failed due to unknown reasons, this creation iteration will be passed")
        io_stat = 3
    else:
        io_stat = check_error_in_vaspout('vasp.out')
    return io_stat


def xml2xsfs(xml_path, index, tag=None):
    atoms_list = []
    if isinstance(index,slice):
        atoms_list = read(xml_path, format='vasp-xml', index=index)
    elif isinstance(index,list) or isinstance(index,tuple):
        for ind in index:
            atoms =  read(xml_path, format='vasp-xml', index=ind)
            atoms_list.append(atoms)
    elif isinstance(index,int):
        atoms = read(xml_path, format='vasp-xml', index=index)
        atoms_list = [atoms]
    final_list = []
    xsf_paths  = []
    for ind, atoms in enumerate(atoms_list):
        pts = check_in_parapipe(atoms, atoms.positions)
        if len(pts) == len(atoms):
            final_list.append(atoms)
        if tag:
            xsf_path = tag+'-vasp%05d'%ind+'.xsf'
            write_xsf(xsf_path,atoms)
            xsf_paths.append(xsf_path)
    if tag:
        return final_list, xsf_paths
    else:
        return final_list


def VaspEnergyCalc(calc_method):
    def getEnergy(*args,**kwargs):
        calc_method(*args,**kwargs)
        outdir = kwargs['outdir']
        vaspmode = kwargs['vasp_mode']
        vaspccfs = kwargs['vasp_custm_confs']
        cf       = kwargs['continue_flag']
        fast_relax_steps = kwargs['fast_relax_steps']
        if fast_relax_steps == 'default':
            fast_relax_steps = 0
        if cf:
            cal_paths = read_list('%s/calc_list'%outdir)
            xsf_done_paths = []
            if os.path.exists('%s/list'%outdir):
                xsf_done_paths = read_list('%s/list'%outdir)
            cals = cal_paths.copy()
            if len(cal_paths):
                xsfs = xsf_done_paths
                for cal_path in cal_paths:
                    print(cal_path)
                    atoms = read(cal_path)
                    suffix = '.'+cal_path.split('.')[-1]
                    tag = cal_path.replace(suffix,'')

                    vasp_dir = 'vasp_io/%s'%os.path.basename(tag)
                    if not os.path.isdir(vasp_dir):
                        os.makedirs(vasp_dir)
                    io_stat = vasprun(atoms,vaspmode,vaspccfs,
                            fast_relax_steps)
                    paths = []
                    if not io_stat:
                        traj_atoms, traj_paths = xml2xsfs('vasprun.xml', 
                               -1, tag=tag)
                        for traj_i in zip(traj_atoms, traj_paths):
                            #if not check_nomic_distance(traj_i[0]):
                            paths.append(traj_i[1])

                    xsfs += paths
                    write_list(outdir+'/list',onlybase(xsfs))
                    cals.remove(cal_path)
                    os.remove(cal_path)
                    write_list('%s/calc_list'%outdir,cals)
                    os.system('mv vasprun.xml vasp_io/')
                    for fn in FILES_NEEDED:
                        if fn != 'vasprun.xml':
                             os.system('mv %s %s'%(fn,vasp_dir))
                    print('Done\n')
                print('Energy calculation done\n')
            else:
                print('Energy calculation already done\n')
        else:
            print('"VaspEnergyCalc" is shut down, set continue_flag=True to switch it on.\n')
    return getEnergy


def VaspAIMDCalc(*args,**kwargs):
    from aenet.select import get_index
    atoms = args[0]
    nsamples = args[1]
    vaspmode = kwargs['vasp_mode']
    vaspccfs = kwargs['vasp_custm_confs']
    fast_relax_steps = kwargs['fast_relax_steps']
    if not fast_relax_steps or fast_relax_steps == 'default':
        fast_relax_steps = 0
    nstart = fast_relax_steps
    if not vaspccfs:
        raise Exception("'VaspCustomized' can not be Nonetype, read the manual for more details")
    if 'nsw' not in list(vaspccfs.keys()):
        raise Exception("'nsw' should be specified in 'VaspCustomized'")
    nstop  = vaspccfs['nsw'] 
    indexpick = get_index([nstart,nstop],nsamples)
    #print(indexpick)

    vasprun(atoms,vaspmode,vaspccfs,fast_relax_steps)
    images = xml2xsfs('vasprun.xml', indexpick)
    for fn in FILES_NEEDED:
        os.system('mv %s vasp_io/'%fn)
    return images


def vasprun(atoms,vasp_mode,vasp_custm_confs=None,fast_relax_steps=0):
    if not os.path.isdir('vasp_io'):
        os.makedirs('vasp_io')
    for iofile in VASP_IOFILES:
        if os.path.exists(iofile):
            os.system('rm -f %s'%iofile)

    amin = 0.1
    cell_lengths = atoms.get_cell_lengths_and_angles()[0:3]
    for cell_length in cell_lengths:
        if cell_length > 50:
            amin = 0.01
    vasp_recom_confs = {
            "xc"         : "PBE",
            "setups"     : "recommended",
            "gga_compat" : False,
            "lasph"      : True,
            "encut"      : 420,
            "prec"       : "Accurate",
            "istart"     : 0,
            "icharg"     : 2,
            "ismear"     : 0,
            "sigma"      : 0.1,
            "lreal"      : "AUTO",
            "kgamma"     : True,
            "lwave"      : False,
            "lcharg"     : False,
            "nsim"       : 10,
            "isym"       : 0,
            "ncore"      : 8,
            "ediff"      : 1e-5
            }

    ##################################################
    #                                                #
    #                 customize                      #
    #                                                #
    ##################################################
    if vasp_mode == 'customized':
        calc = Vasp(**vasp_custm_confs)
        atoms.set_calculator(calc)
        io_stat = trig_run(atoms)

    ##################################################
    #                                                #
    #          bulk-single point energy calc         #
    #                                                #
    ##################################################
    elif vasp_mode == 'bulk':
        calc = Vasp(**vasp_recom_confs)
        atoms.set_calculator(calc)
        atoms.calc.set(
                kspacing = 0.3
                )
        if vasp_custm_confs:
            atoms.calc.set(**vasp_custm_confs)
        io_stat = trig_run(atoms)

    ##################################################
    #                                                #
    #          bulk-single point energy calc with    #
    #                   spin polarization            #
    #                                                #
    ##################################################
    elif vasp_mode == 'bulk-spin':
        mag = []
        for i in atoms.symbols:
            if i == 'Fe':
                mag.append(2.0)
            else:
                mag.append(0.0)
        atoms.set_initial_magnetic_moments(mag)
        calc = Vasp(**vasp_recom_confs)
        atoms.set_calculator(calc)
        if vasp_custm_confs:
            atoms.calc.set(**vasp_custm_confs)
        io_stat = trig_run(atoms)

    ##################################################
    #                                                #
    #      bulk2-single point energy multi-calc      #
    #                                                #
    ##################################################
    elif vasp_mode == 'bulk2':
        calc = Vasp(**vasp_recom_confs)
        atoms.set_calculator(calc)
        atoms.calc.set(
            lcharg   = True,
            kspacing = 0.8,
            algo     = 'Normal',
            nelm     = 80,
            ediff    = 1e-3,
            amin     = amin
            )
        io_stat = trig_run(atoms)
        if not io_stat:
            atoms.calc.set(
                kspacing = 0.3,
                algo     = 'Normal',
                icharg   = 1,
                nelm     = 100,
                ediff    = 1e-5,
                kpar     = 2,
                amin     = amin
                )
            io_stat = trig_run(atoms)

    ##################################################
    #                                                #
    #                cluster relaxation              #
    #                                                #
    ##################################################
    elif vasp_mode == 'cluster':
        kpts = monkhorst_pack([1,1,1])
        calc = Vasp(**vasp_recom_confs)
        atoms.set_calculator(calc)
        if fast_relax_steps:
            relax_nsw = fast_relax_steps
        else:
            relax_nsw = 30
        atoms.calc.set(
            nsw    = relax_nsw,
            ibrion = 2,
            potim  = 0.1,
            isif   = 2,
            algo   = 'Normal',
            prec   = 'Normal',
            encut  = 330,
            ediff  = 1e-2,
            kpts   = kpts
            #kspacing = 0.5
            )
        io_stat = trig_run(atoms)
        if not io_stat:
            final_atoms = read('vasprun.xml',index=-1)
            #if not os.path.isdir('xml'):
            #    os.makedirs('xml')
            #    ind = 0
            #else:
            #    fns = os.listdir('xml')
            #    if len(fns):
            #        ns = [int(fn.split('.')[0]) for fn in fns]
            #        ind = max(ns)+1
            #    else:
            #        ind = 0
            if not check_nomic_distance(final_atoms):
            #    shutil.move("vasprun.xml","xml/%i.xml"%ind)
                final_atoms.set_calculator(calc)
                final_atoms.calc.set(
                    ibrion = -1,
                    nsw    = 0,
                    isif   = 2,
                    prec   = 'Accurate',
                    algo   = 'Normal',
                    encut  = 420,
                    ediff  = 1e-5,
                    kpts   = kpts
                    #kspacing = 0.3
                    )
                io_stat = trig_run(final_atoms)
            else:
                print("Warning: Ionic relaxation failed!")
                io_stat = 1


    ##################################################
    #                                                #
    #                 bulk relaxation                #
    #                                                #
    ##################################################
    elif vasp_mode == 'bulk-relax':
        calc = Vasp(**vasp_recom_confs)
        atoms.set_calculator(calc)
        if fast_relax_steps:
            relax_nsw = fast_relax_steps
        else:
            relax_nsw = 500
        import numpy as np
        p = np.random.normal(0, 5)
        p = max(min(p,10), -10)
        p *= 10

        atoms.calc.set(
            nsw    = relax_nsw,
            ibrion = 2,
            potim  = 0.1,
            isif   = 3,
            pstress= p,
            algo   = 'Normal',
            prec   = 'Normal',
            ediff  = 1e-2,
            ediffg = -0.5,
            encut  = 320,
            kspacing = 0.5
            )
        io_stat = trig_run(atoms)
        if not io_stat:
            final_atoms = read('vasprun.xml',index=-1)
            final_atoms.calc.set(
                    nsw    = 0,
                    ibrion = -1,
                    isif   = 2,
                    algo   = 'Normal',
                    prec   = 'Accurate',
                    encut  = 420,
                    ediff  = 1e-5,
                    kspacing = 0.2
                )
            io_stat = trig_run(final_atoms)


    ##################################################
    #                                                #
    #       ab-initio MD ---- Langevin NVT ensemble  #
    #                                                #
    ##################################################
    elif vasp_mode == 'Langevin-NVT':
        calc = Vasp(**vasp_recom_confs)
        atoms.set_calculator(calc)
        atoms.calc.set(
            ibrion = 0,
            isif   = 2,
            mdalgo = 3
            )
        custm_confs = ['nsw','tebeg','langevin_gamma','potim']
        for custm_conf in custm_confs:
            if custm_conf not in list(vasp_custm_confs.keys()):
                raise Exception("Vasp tag lost : '%s', check 'vasp_custm_confs' in the configuration file."%custm_conf)
        atoms.calc.set(**vasp_custm_confs)
        io_stat = trig_run(atoms)


    ##################################################
    #                                                #
    #       ab-initio MD ---- Langevin NPT ensemble  #
    #                   (unstable)                   #
    ##################################################
    elif vasp_mode == 'Langevin-NPT':
        '''
        calc = Vasp(**vasp_custm_confs)
        atoms.set_calculator(calc)
        atoms.calc.set(
                ibrion = 0,
                isif   = 3,
                potim  = 2,
                mdalgo = 3
                )
        custm_confs = ['nsw','pstress','tebeg','langevin_gamma_l','pmass']
        for custm_conf in custm_confs:
            if custm_conf not in list(vasp_custm_confs.keys()):
                raise Exception("Vasp tag lost : '%s', check 'vasp_custm_confs' in the configuration file."%custm_conf)
        '''
        raise NotImplementedError("Vasp NPT thermostat is not stable for automated setting, please use 'Customized' style to set input tags manually")

    ##################################################
    #                                                #
    #    ab-initio MD ---- Nose-Hoover NVT ensemble  #
    #          (recommended in AIMD tasks)           #
    ##################################################
    elif vasp_mode == 'Nose-NVT':
        calc = Vasp(**vasp_recom_confs)
        atoms.set_calculator(calc)
        atoms.calc.set(
            ibrion = 0,
            ediff  = 1.e-2,
            mdalgo = 2
            )
        custm_confs = ['nsw','tebeg','smass','potim']
        for custm_conf in custm_confs:
            if custm_conf not in list(vasp_custm_confs.keys()):
                raise Exception("Vasp tag lost : '%s', check 'vasp_custm_confs' in the configuration file."%custm_conf)
        atoms.calc.set(**vasp_custm_confs)
        if 'kspacing' in vasp_custm_confs:
            if vasp_custm_confs['kspacing'] > 1.:
                kpts = monkhorst_pack([1,1,1])
                atoms.calc.set(
                        kspacing = None,
                        kpts = kpts
                        )
        io_stat = trig_run(atoms)
    
    if not os.path.isdir('vasp-xml'):
        os.makedirs('vasp-xml')
        shutil.copy('vasprun.xml','vasp-xml/1.xml')
    else:
        n_list = [int(n.split('.')[0]) for n in os.listdir('vasp-xml')]
        max_n = max(n_list)
        shutil.copy('vasprun.xml','vasp-xml/%i.xml'%(max_n+1))
    for iofile in VASP_IOFILES:
        if os.path.exists(iofile) and iofile not in FILES_NEEDED:
            os.system('mv %s vasp_io/'%iofile)

    return io_stat




