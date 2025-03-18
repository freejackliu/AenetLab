import numpy as np
import os
import inspect
from aenet.xsf import write_xsf,read_xsf
from ase.io import write
from ase.io.lammpsdata import write_lammps_data
from aenet.AenetLab.aenet_io import * #read, write and check methods
from aenet.AenetLab.Calc.VaspCalc import VaspEnergyCalc


def env_init(initdir, outdir):
    CONF_FD_NAMES = ('seed','calc_list','list','.isdataset','.creation_done')
    if not os.path.exists('%s/list'%initdir):
        files_list = os.listdir(initdir)
        for name in CONF_FD_NAMES:
            if name in files_list:
                files_list.remove(name)
        write_list(os.path.join(initdir,'list'),files_list)
    
    files_list = read_list('%s/list'%initdir)
    make_empty_dir(outdir)
    return files_list


def res_seed(outdir, seed_in):
    if seed_in != 'default':
        seed = '%s\n'%seed_in
    else:
        seed = '%s\n'%None
    write_isfl('%s/seed'%outdir,seed)


def style_run(style_name, style_func, atoms, nsamples, 
        confs, args_list, outdir, tag, perc, specorder, uc_xsfs, c_xsfs):
    scope ={style_name:style_func,
            'atoms':atoms,
            'nsamples':nsamples}
    s='images=%s(atoms,nsamples'%style_name
    for arg in args_list:
        conf = confs[arg]
        if conf != 'default':
            if isinstance(conf,str):
                s+=",%s='%s'" % (arg,conf)
            else:
                s+=",%s=%s" % (arg,conf)
    s+=')'
    outfmt = confs['output_format']
    exec(s,scope)
    print('   %s generate %d structures.(%s)'%(tag,len(scope['images']),perc))
    for ind,image in enumerate(scope['images']):
        if style_name != 'abinitio_MD':
            if outfmt == 'default':
                outfmt = 'poscar'
            name1 = '%s-%s%05d.%s'%(tag, style_name, ind, outfmt)
            name2 = os.path.join(outdir, name1)
            uc_xsfs.append(name1)
            if outfmt != 'data':
                write(name2, image)
            else:
                write_lammps_data(name2, image, specorder=specorder)
        else:
            if outfmt == 'default':
                outfmt = 'xsf'
            name1 = '%s-%s%05d.%s'%(tag, style_name, ind, outfmt)
            name2 = os.path.join(outdir, name1)
            c_xsfs.append(name1)
            if outfmt == 'xsf':
                write_xsf(name2, image)
            elif outfmt == 'data':
                write_lammps_data(name2, image, specorder=specorder)
            else:
                write(name2, image)


def create_run(initdir, outdir, nsamples, confs, create_style, create_func, Z_of_type, specorder):
    creation_done = False
    cdf = '%s/.creation_done'%outdir
    if os.path.exists(cdf):
        creation_done = read_isfl(cdf)
    if not creation_done:
        init_paths = env_init(initdir, outdir)
        seed_in = confs['seed']
        res_seed(outdir, seed_in)
        isdataset = confs['isdataset']
        if isdataset in ["default", True]:
            write_isfl('%s/.isdataset'%outdir, True)
        index = confs['index']
        print('Create the "%s" style dataset'%create_style)
        uc_xsfs, c_xsfs, n_init, al_l, tag_l = [], [], 0, [], []
        args_list = inspect.getargspec(create_func).args[2:]
        for init_file in init_paths:
            ar, tag = aread(init_file, Z_of_type, index)
            if not isinstance(ar,list):
                al = [ar]
            else:
                al = ar.copy()
            n_init += len(al)
            al_l.append(al)
            tag_l.append(tag)
        n_count = 0
        for ind, al in enumerate(al_l):
            for id, atoms in enumerate(al):
                if not isinstance(ar,list):
                    tag = tag_l[ind]
                else:
                    tag = tag_l[ind]+'_%05d'%id
                n_count += 1
                perc = '%d/%d'%(n_count,n_init)
                style_run(create_style, create_func, atoms, nsamples,
                        confs, args_list, outdir, tag, perc, specorder, uc_xsfs, c_xsfs)
        if len(uc_xsfs):
            write_list('%s/calc_list'%outdir,uc_xsfs)
            print('Creation of "%s" style dataset done, energy calculation or the next creation continued\n'%create_style)
        else:
            write_list('%s/list'%outdir,c_xsfs)
            print('Creation of abinitio-MD dataset done')
        write_isfl(cdf,True)
    else:
        print('Creation of "%s" style dataset already done, energy calculation or the next creation continued\n'%create_style)
 

def select_run(initdir, outdir, nsamples, confs, Z_of_type, specorder):
    from aenet.select import select

    creation_done = False
    cdf = '%s/.creation_done'%outdir
    select_style = confs['style']
    if os.path.exists(cdf):
        creation_done = read_isfl(cdf)
    if not creation_done:
        init_paths = env_init(initdir, outdir)
        seed_in = confs['seed']
        delay   = confs['delay']      
        params  = confs['params']
        outfmt  = confs['output_format']
        isdataset = confs['isdataset']
        res_seed(outdir, seed_in)
        if seed_in == 'default':
            seed_in = None
        if isdataset in ["default", True]:
            write_isfl('%s/.isdataset'%outdir, True)
        if outfmt == 'default':
            if 'xml2xsfs' in list(confs.keys()):
                outfmt = 'xsf'
            else:
                outfmt = 'poscar'
        if delay == 'default':
            delay = 0
        print('Run "%s" style selection'%select_style)
        ar_pool = []
        uc_xsfs_pool = []
        ct = 0
        for init_file in init_paths:
            ar, tag = aread(init_file,Z_of_type,index_list=[delay,None])
            for ind,ar_i in enumerate(ar):
                uc_xsfs_pool.append(outdir+'/'+tag+'-selection%05d.%s'%(delay+ct+ind,outfmt))
                ar_pool.append(ar_i)
            ct += len(ar)
        uc_xsfs, images = select(uc_xsfs_pool,ar_pool,nsamples,select_style,seed_in, **params)
        uc_xsfs_name = []
        for uc_xsf, image in zip(uc_xsfs,images):
            uc_xsfs_name.append(os.path.basename(uc_xsf))
            if outfmt != 'data':
                if 'xml2xsfs' in list(confs.keys()):
                    write_xsf(uc_xsf,image)
                else:
                    write(uc_xsf,image)
            else:
                write_lammps_data(uc_xsf,image,specorder=specorder)
        if 'xml2xsfs' in list(confs.keys()):
            write_list('%s/list'%outdir,uc_xsfs_name)
        else:
            write_list('%s/calc_list'%outdir,uc_xsfs_name)
        write_isfl(cdf,True)
        print('Selection done')
    else:
        print('"%s" style selection already done, energy calculation or the next creation continued\n'%select_style)


def create_ljdimer_dataset(pair,E_atom,nsamples=10,drange=[1.4,2.8],outdir='LJDimer_Xsfs'):
    from ase.calculators.lj import LennardJones
    from ase import Atoms
    
    creation_done = False
    cdf = os.path.join(outdir,'.creation_done')
    if os.path.exists(cdf):
        creation_done = read_isfl(cdf)
    if not creation_done:
        print('LJ Dimer test started')
        make_empty_dir(outdir)
        E_dimer = sum(E_atom)
        d_samples = np.linspace(drange[0],drange[1],nsamples)
        ScatterData = []
        for d_sample in d_samples:
            molecule = Atoms(pair, positions=[(0., 0., 0.), (0., 0., d_sample)])
            molecule.calc = LennardJones()
            E_coh = molecule.get_potential_energy()
            molecule.calc.results['energy'] = E_coh + E_dimer
            write_xsf('%s/lj-dimer-%.6f.xsf'%(outdir,d_sample),molecule)
            ScatterData.append([d_sample,E_coh])
        ScatterData = np.array(ScatterData)
        write_isfl(cdf,True)
        print('LJ Dimer test done')
        return ScatterData
    else:
        print('LJ Dimer test already done')
        return None


@VaspEnergyCalc
def create_replace_dataset(initdir,nsamples,confs,Z_of_type,specorder,
        continue_flag,outdir,vasp_mode,vasp_custm_confs,fast_relax_steps):
    from aenet.replace import replace

    create_style = 'replace'
    create_func  = replace
    create_run(initdir,outdir,nsamples,confs,create_style,create_func,Z_of_type,specorder)


@VaspEnergyCalc
def create_disturb_dataset(initdir,nsamples,confs,Z_of_type,specorder,
        continue_flag,outdir,vasp_mode,vasp_custm_confs,fast_relax_steps):
    from aenet.disturb import disturb
    
    create_style = 'disturb'
    create_func  = disturb
    create_run(initdir,outdir,nsamples,confs,create_style,create_func,Z_of_type,specorder) 


@VaspEnergyCalc
def create_vorinsert_dataset(initdir,nsamples,confs,Z_of_type,specorder,
        continue_flag,outdir,vasp_mode,vasp_custm_confs,fast_relax_steps):
    from aenet.vor_insert import vor_insert

    create_style = 'vor_insert'
    create_func = vor_insert
    create_run(initdir,outdir,nsamples,confs,create_style,create_func,Z_of_type,specorder)


@VaspEnergyCalc
def create_convinsert_dataset(initdir,nsamples,confs,Z_of_type,specorder,
        continue_flag,outdir,vasp_mode,vasp_custm_confs,fast_relax_steps):
    from aenet.conv_insert import conv_insert
        
    create_style = 'conv_insert'
    create_func = conv_insert
    create_run(initdir,outdir,nsamples,confs,create_style,create_func,Z_of_type,specorder)


@VaspEnergyCalc
def select_dump_dataset(initdir,nsamples,confs,Z_of_type,specorder,
        continue_flag,outdir,vasp_mode,vasp_custm_confs,fast_relax_steps):
    
    select_run(initdir,outdir,nsamples,confs,Z_of_type,specorder)


def select_xml_dataset(initdir,nsamples,confs,Z_of_type,specorder,continue_flag,outdir):

    confs.update({'xml2xsfs' : True})
    select_run(initdir,outdir,nsamples,confs,Z_of_type,specorder)


def create_abinitio_MD_dataset(initdir,nsamples,confs,Z_of_type,specorder,
        continue_flag,outdir,vasp_mode,vasp_custm_confs,fast_relax_steps):
    from aenet.MD import abinitio_MD

    create_style = 'abinitio_MD'
    create_func = abinitio_MD
    confs.update({'vasp_mode' : vasp_mode,
                  'vasp_custm_confs' : vasp_custm_confs,
                  'fast_relax_steps' : fast_relax_steps})
    if nsamples == 'auto':
        nstart = confs['fast_relax_steps']
        if nstart == 'default':
            nstart = 0
        if 'nsw' in list(confs['vasp_custm_confs'].keys()):
            nstop = confs['vasp_custm_confs']['nsw']
        nsamples = nstop - nstart
    create_run(initdir,outdir,nsamples,confs,create_style,create_func,Z_of_type,specorder)


@VaspEnergyCalc
def create_opt_dataset(initdir,nsamples,confs,Z_of_type,specorder,
        continue_flag,outdir,vasp_mode,vasp_custm_confs,fast_relax_steps):
    from aenet.opt import dft_opt
    
    create_style = 'dft_opt'
    create_func = dft_opt
    if vasp_mode not in ['bulk-relax','cluster']:
        raise ValueError("Invalid vasp mode : %s, only 'bulk-relax' and 'cluster' are valid for 'DFTOpt' style."%vasp_mode)
    create_run(initdir,outdir,nsamples,confs,create_style,create_func,Z_of_type,specorder)


def create_1Dtaylor(initdir,E_atom,dr,outdir):
    from aenet.taylor_1D import taylor_first_dimer

    make_empty_dir(outdir)
    init_paths = read_list('%s/list'%initdir)
    test_index = np.random.randint(len(init_paths))
    istest = np.array([False for _ in init_paths])
    istest[test_index] = True
    print('Taylor expansion started')

    E_dimer = sum(E_atom)
    aug_list = []
    #tay_list = []
    ScatterData = []
    tags_seen = set()
    lr_dict = {0:'left',1:'right'}
    for k, v in enumerate(init_paths):
        tag = os.path.basename(v)[0:-4]
        if tag in tags_seen:
            raise RuntimeError("Each xsf file in list must have unique name!")
        tags_seen.add(tag)
        molecule = read_xsf(v)
        aug_list.append([os.path.basename(v), bool(istest[k])])
        if not istest[k]:
            taylor_samples = taylor_first_dimer(molecule,dr)
            if taylor_samples is not None:
                for i, sample in enumerate(taylor_samples):
                    name1 = tag+'-dimer'+lr_dict[i]+'.xsf'
                    name2 = os.path.join(outdir, name1)
                    aug_list.append([name1, bool(istest[k])])
                    ScatterData.append([sample.positions[1][2],
                        sample.calc.results['energy']-E_dimer])
                    write_xsf(name2, sample)
    write_info('%s/tayinfo'%outdir,aug_list)
    print('Taylor expansion done')
    return ScatterData


def create_3Dtaylor(init_paths,drmax,nsamples,taylor_confs,outdir):
    from aenet.taylor import taylor_first
    from math import ceil
    
    if taylor_confs['fcut'] == 'default':
        fcut = [0.3,3.0]
    else:
        if isinstance(taylor_confs['fcut'],list) or isinstance(taylor_confs['fcut'],tuple):
            fcut = taylor_confs['fcut']
        else:
            raise TypeError("Expected type of 'fcut' : list or tuple ,e.g. [0.3,3.0]")
    if taylor_confs['nfirst'] == 'default':
        nfirst = len(init_paths)
    else:
        if isinstance(taylor_confs['nfirst'],int):
            nfirst = min(taylor_confs['nfirst'],len(init_paths))
            nfirst = max(taylor_confs['nfirst'],0)
        else:
            raise TypeError("Expected type of 'nfirst' : int")
    if taylor_confs['seed'] == 'default':
        seed = None
    else:
        seed = taylor_confs['seed']
    if taylor_confs['tp'] == 'default':
        test_percent = 0.1
    else:
        if isinstance(taylor_confs['tp'],str):
            test_percent = float(taylor_confs['tp'].strip('%'))/100
        elif isinstance(taylor_confs['tp'],float):
            if 0. <= taylor_confs['tp'] <= 1.:
                test_percent = taylor_confs['tp']
            else:
                raise Exception("'test_percent' should be in range of [0,1]")
        else:
            raise TypeError("Expected type of 'test_percent' : float or str , e.g. 15% or 0.15")
    n_test = ceil(test_percent*len(init_paths))
    
    make_empty_dir(outdir)
    rng = np.random.RandomState(seed)
    istest = np.array([False for _ in init_paths])
    istest[0:n_test] = True
    rng.shuffle(istest)
    print('Taylor expansion started')

    aug_list, aug_list2 = [] , []
    tags_seen = dict()
    for k, v in enumerate(init_paths):
        tag = os.path.basename(v)[0:-4]
        tag0 = tag.split('-c')[0]
        if tag0 in tags_seen.keys():
            # Each xsf in tayinfo/list must have unique name!
            tags_seen[tag0] += 1 
        else:
            tags_seen[tag0] = 0
        if len(tag.split('-c')) < 2:
            tag2 = tag + '-c%03d'%tags_seen[tag0]
            v_t = v.replace(tag,tag2)
            os.rename(v,v_t)
        else:
            tag2 = tag
            v_t = v
        atoms = read_xsf(v_t)
        #aug_list.append([os.path.basename(v_t), bool(istest[k])])
        aug_list2.append([v_t, bool(istest[k])])
        if not istest[k] and k < nfirst:
            taylor_samples = taylor_first(atoms,drmax,nsamples,fcut=fcut,seed=seed)
            if taylor_samples is not None:
                for i, sample in enumerate(taylor_samples):
                    name1 = tag2 + '-taylor%03d.xsf'%i
                    name2 = os.path.join(outdir,name1)
                    aug_list.append((name1, bool(istest[k])))
                    aug_list2.append((os.path.join(outdir,name1), bool(istest[k])))
                    write_xsf(name2, sample)
    write_info('%s/tayinfo'%outdir,aug_list)
    return aug_list2  

