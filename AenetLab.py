import os
import shutil
import subprocess
import numpy as np
from aenet.AenetLab.aenet_io import *

class AenetLab:
    def __init__(self,textnew=None):
        self.init = False
        textold = load_template()
        if textnew:
            new_list = [list(i) for i in walk(textnew)]
            textjson = deep_update(textold,new_list)
        else:
            textjson = textold.copy()
        try:
            AenetLab.Init(self,textjson)
        except:
            raise Exception('AenetLab init failed! Check your configuration json file.\n')
        else:
            self.init  = True
            write_isfl('.islab',True)
            print('AenetLab obj is created successfully!\n')
    
    def __repr__(self):
        types_list = list(self.types.keys())
        types_text = ''
        stp_dict   = {}
        for kl in types_list:
            types_text += kl
            stp_dict[kl] = self.types[kl]['stp_style']
        s = "AenetLab(Types='%s', CreateStyle='%s'," % (types_text, self.crt['style'])
        s += " StpStyle='%s', TrainMethod='%s')" % (stp_dict, self.trn['method'])
        return s

    def Init(self,textjson):
        self.types = textjson['Types']
        self.units = textjson['Units']
        self.rrgn  = textjson['Rrange']
        self.tp    = textjson['TestPercent']
        self.tseed = textjson['TestSeed']

        self.crt   = textjson['Create']
        self.crtd  = textjson['CreateDetails']

        self.tay   = textjson['Taylor']
        self.tayd  = textjson['TaylorDetails']

        self.gen   = textjson['Generate']
        self.stpd  = textjson['StpDetails']

        self.trn   = textjson['Train']
        self.trnd  = textjson['TrainDetails']

        self.vis   = textjson['Visual']
 
        if os.path.exists('.taskid'):
            self.taskid = read_isfl('.taskid')
        else:
            self.taskid = 1
            write_isfl('.taskid',1)

        self.Z_of_type = {}
        self.specorder = list(self.types.keys())
        self.old_d = textjson
        
        write_isfl('tseed',self.tseed)
        for elem in self.types.keys():
            atomic_number,E_atom = get_value(elem)
            self.Z_of_type[self.types[elem]['lammps_number']] = atomic_number
            if self.types[elem]['E_atom'] == "default":
                if E_atom:
                    self.types[elem]['E_atom'] = E_atom
                else:
                    raise Exception("\
    No monatomic energy data available for %s\
        Tips: specify its value in the configuration json file in lab dir(template.json) \
    or info-loading json file in the src dir(atomic_info.json)."%elem) 

    def set(self,args_keyvalues):
        if type(args_keyvalues) not in [list,tuple]:
            raise TypeError("Invalid args for 'set', type list or tuple is expected")
        if self.init:
            new_d = deep_update(self.old_d,args_keyvalues)
            AenetLab.Init(self,new_d)
        else:
            raise Exception("AenetLab initialization failed, please load the json file first\n")


    def create(self):
        if not self.init:
            raise Exception('AenetLab obj is not initialized\n')
        
        crt_style  = self.crt['style']
        crtd       = self.crtd[crt_style]

        if crt_style == 'LJDimer':
            from aenet.AenetLab.create import create_ljdimer_dataset,create_1Dtaylor
            
            pair = list(self.types.keys())[0]*2
            E_atom  = [self.types[elem]['E_atom'] for elem in self.types.keys()]
            nsamples= 10
            drange  = [1.4,2.8]
            outdir  = crtd['outdir']
            data = create_ljdimer_dataset(pair,E_atom,nsamples,drange,outdir)


        elif crt_style == 'Replace':
            from aenet.AenetLab.create import create_replace_dataset

            initdir  = crtd['initdir']
            outdir = crtd['outdir']
            nsamples  = crtd['nsamples']
            replace_confs = {
                    'command'           : crtd['command'],
                    'inner'             : crtd['inner'],
                    'style'             : crtd['style'],
                    'output_format'     : crtd['outfmt'],
                    'index'             : crtd['index'],
                    'isdataset'         : crtd['isdataset'],
                    'seed'              : crtd['seed'],
                    'include_initial'   : crtd['inc_init']
                    }

            continue_flag = self.crtd['Vasp']['continue_flag']
            vasp_mode = self.crtd['Vasp']['mode']
            vasp_custm_confs = self.crtd['VaspCustomized']
            fast_relax_steps = self.crtd['Vasp']['fast_relax_steps']
            if vasp_custm_confs == {}:
                vasp_custm_confs = None
            if (fast_relax_steps is None) or (fast_relax_steps is 'default'):
                fast_relax_steps = 0

            create_replace_dataset(initdir,nsamples,replace_confs,self.Z_of_type,
                    self.specorder,continue_flag=continue_flag,outdir=outdir,
                    vasp_mode=vasp_mode,vasp_custm_confs=vasp_custm_confs,
                    fast_relax_steps=fast_relax_steps)


            
        elif crt_style == 'Disturb':
            from aenet.AenetLab.create import create_disturb_dataset

            initdir  = crtd['initdir']
            outdir = crtd['outdir']
            nsamples  = crtd['nsamples']
            disturb_confs = {
                    'deformation_limits': crtd['defm_lim'],
                    'rattle_size'       : crtd['ratt_size'],
                    'no_delete'         : crtd['no_delete'],
                    'delete_percent'    : crtd['del_perc'],
                    'replicate'         : crtd['replicate'],
                    'replicate_uplimit' : crtd['rep_uplim'],

                    'bond_range'        : crtd['bond_range'],
                    'peak_threshold'    : crtd['peak_thres'],
                    'eos_interv'        : crtd['eos_interv'],
                    'output_format'     : crtd['outfmt'],
                    'index'             : crtd['index'],
                    'isdataset'         : crtd['isdataset'],
                    'seed'              : crtd['seed'],
                    'include_initial'   : crtd['inc_init']       }
            continue_flag = self.crtd['Vasp']['continue_flag']
            vasp_mode = self.crtd['Vasp']['mode']
            vasp_custm_confs = self.crtd['VaspCustomized']
            fast_relax_steps = self.crtd['Vasp']['fast_relax_steps']
            if vasp_custm_confs == {}:
                vasp_custm_confs = None
            if (fast_relax_steps is None) or (fast_relax_steps is 'default'):
                fast_relax_steps = 0

            create_disturb_dataset(initdir,nsamples,disturb_confs,
                    self.Z_of_type,self.specorder,
                    continue_flag=continue_flag,outdir=outdir,
                    vasp_mode=vasp_mode,vasp_custm_confs=vasp_custm_confs,
                    fast_relax_steps=fast_relax_steps)

        elif crt_style == 'VorInsert':
            from aenet.AenetLab.create import create_vorinsert_dataset
            
            initdir  = crtd['initdir']
            outdir = crtd['outdir']
            nsamples  = crtd['nsamples']
            vorinsert_confs = {
                    'insert_elem'     : crtd['ins_elem'],
                    'strategy'        : crtd['strategy'],
                    'only_selected'   : crtd['only_selected'],
                    'max_rins'        : crtd['max_rins'],
                    'cnst_rins'       : crtd['cnst_rins'],
                    'cnst_num'        : crtd['cnst_num'],
                    'dmin'            : crtd['dmin'],
                    'dmax'            : crtd['dmax'],
                    'nbins'           : crtd['nbins'],
                    'density'         : crtd['density'],
                    'output_format'   : crtd['outfmt'],
                    'index'           : crtd['index'],
                    'isdataset'       : crtd['isdataset'],
                    'seed'            : crtd['seed'],
                    'include_initial' : crtd['inc_init']}
            continue_flag = self.crtd['Vasp']['continue_flag']
            vasp_mode = self.crtd['Vasp']['mode']
            vasp_custm_confs = self.crtd['VaspCustomized']
            fast_relax_steps = self.crtd['Vasp']['fast_relax_steps']
            if vasp_custm_confs == {}:
                vasp_custm_confs = None
            if (fast_relax_steps is None) or (fast_relax_steps == 'default'):
                fast_relax_steps = 0

            create_vorinsert_dataset(initdir,nsamples,vorinsert_confs,
                     self.Z_of_type,self.specorder,
                     continue_flag=continue_flag,outdir=outdir,
                     vasp_mode=vasp_mode,vasp_custm_confs=vasp_custm_confs,
                     fast_relax_steps=fast_relax_steps)

        elif crt_style == 'ConvexInsert':
            from aenet.AenetLab.create import create_convinsert_dataset
            
            initdir  = crtd['initdir']
            outdir = crtd['outdir']         
            nsamples  = crtd['nsamples']
            clustercut_confs = {
                    'insert_elem'     : crtd['ins_elem'],
                    'insert_number'   : crtd['ins_num'],
                    'dmin'            : crtd['dmin'],
                    'rattle_size'     : crtd['ratt_size'],
                    'area_range'      : crtd['area_range'],
                    'vacuum_width'    : crtd['vacuum_width'],
                    'output_format'   : crtd['outfmt'],
                    'index'           : crtd['index'],
                    'isdataset'       : crtd['isdataset'],
                    'seed'            : crtd['seed'],
                    'include_initial' : crtd['inc_init']}
            continue_flag = self.crtd['Vasp']['continue_flag']
            vasp_mode = self.crtd['Vasp']['mode']
            vasp_custm_confs = self.crtd['VaspCustomized']
            fast_relax_steps = self.crtd['Vasp']['fast_relax_steps']
            if vasp_custm_confs == {}:
                vasp_custm_confs = None
            if (fast_relax_steps is None) or (fast_relax_steps is 'default'):
                fast_relax_steps = 0

            create_convinsert_dataset(initdir,nsamples,clustercut_confs,
                    self.Z_of_type,self.specorder,
                    continue_flag=continue_flag,outdir=outdir,
                    vasp_mode=vasp_mode,vasp_custm_confs=vasp_custm_confs,
                    fast_relax_steps=fast_relax_steps)

        elif crt_style == 'DumpSelect':
            from aenet.AenetLab.create import select_dump_dataset

            initdir = crtd['initdir']
            outdir = crtd['outdir']
            nsamples = crtd['nsamples']
            dump_confs = {
                    'style'           : crtd['style'],
                    'params'          : {
                        'symbols' : list(self.types.keys()),
                        'nmax'  : self.stpd['Spherical_Chebyshev']['nmax'],
                        'lmax'  : self.stpd['Spherical_Chebyshev']['lmax'],
                        'rcut'    : self.rrgn['rcut'],
                        'nn_path' : crtd['nn_path'],
                        'etor'    : crtd['etor'],
                        'niter'   : crtd['niter'] if crtd['niter'] != 'default' else 0
                        },
                    'delay'           : crtd['delay'],
                    'output_format'   : crtd['outfmt'],
                    'isdataset'       : crtd['isdataset'],
                    'seed'            : crtd['seed']}
            continue_flag = self.crtd['Vasp']['continue_flag']
            vasp_mode = self.crtd['Vasp']['mode']
            vasp_custm_confs = self.crtd['VaspCustomized']
            fast_relax_steps = self.crtd['Vasp']['fast_relax_steps']
            if vasp_custm_confs == {}:
                vasp_custm_confs = None
            if (fast_relax_steps is None) or (fast_relax_steps is 'default'):
                fast_relax_steps = 0

            select_dump_dataset(initdir,nsamples,dump_confs,self.Z_of_type,
                    self.specorder,continue_flag=continue_flag,outdir=outdir,
                    vasp_mode=vasp_mode,vasp_custm_confs=vasp_custm_confs,
                    fast_relax_steps=fast_relax_steps)

        elif crt_style == 'Xml2xsfs':
            from aenet.AenetLab.create import select_xml_dataset

            initdir = crtd['initdir']
            outdir  = crtd['outdir']
            nsamples = crtd['nsamples']
            xml_confs = {
                    'style'           : crtd['style'],
                    'params'          : {
                        'symbols' : list(self.types.keys()),
                        'nmax'  : self.stpd['Spherical_Chebyshev']['nmax'],
                        'lmax'  : self.stpd['Spherical_Chebyshev']['lmax'],
                        'rcut'    : self.rrgn['rcut'],
                        'nn_path' : crtd['nn_path'],
                        'etor'    : crtd['etor'],
                        'niter'   : crtd['niter'] if crtd['niter'] != 'default' else 0
                        },
                    'delay'           : crtd['delay'],
                    'output_format'   : crtd['outfmt'],
                    'isdataset'       : crtd['isdataset'],
                    'seed'            : crtd['seed']}
            continue_flag = self.crtd['Vasp']['continue_flag']
            select_xml_dataset(initdir,nsamples,xml_confs,self.Z_of_type,
                    self.specorder,continue_flag,outdir)

        elif crt_style == 'AbinitioMD':
            from aenet.AenetLab.create import create_abinitio_MD_dataset

            initdir = crtd['initdir']
            outdir  = crtd['outdir']
            nsamples = crtd['nsamples']
            abinitiomd_confs = {
                    'nn_opt'          : crtd['nn_opt'],
                    'nn_path'         : crtd['nn_path'],
                    'output_format'   : crtd['outfmt'],                 
                    'index'           : crtd['index'],
                    'isdataset'       : crtd['isdataset'],
                    'seed'            : crtd['seed'],
                    'include_initial' : crtd['inc_init']}
            continue_flag = self.crtd['Vasp']['continue_flag']
            vasp_mode = self.crtd['Vasp']['mode']
            vasp_custm_confs = self.crtd['VaspCustomized']
            fast_relax_steps = self.crtd['Vasp']['fast_relax_steps']
            if vasp_custm_confs == {}:
                vasp_custm_confs = None 
            if (fast_relax_steps is None) or (fast_relax_steps is 'default'):
                fast_relax_steps = 0
            create_abinitio_MD_dataset(initdir,nsamples,abinitiomd_confs,
                    self.Z_of_type,self.specorder,
                    continue_flag=continue_flag,outdir=outdir,
                    vasp_mode=vasp_mode,vasp_custm_confs=vasp_custm_confs,
                    fast_relax_steps=fast_relax_steps)

        elif crt_style == 'DFTOpt':
            from aenet.AenetLab.create import create_opt_dataset
            
            initdir = crtd['initdir']                                       
            outdir  = crtd['outdir']
            nsamples = crtd['nsamples']
            
            dftopt_confs = {
                    'nn_opt'          : crtd['nn_opt'],
                    'nn_path'         : crtd['nn_path'],
                    'cell_filter'     : crtd['cell_filter'],
                    'output_format'   : crtd['outfmt'],
                    'index'           : crtd['index'],
                    'isdataset'       : crtd['isdataset'],
                    'seed'            : crtd['seed'],
                    'include_initial' : crtd['inc_init']
                    }
            continue_flag = self.crtd['Vasp']['continue_flag']
            vasp_mode = self.crtd['Vasp']['mode']
            vasp_custm_confs = self.crtd['VaspCustomized']
            fast_relax_steps = self.crtd['Vasp']['fast_relax_steps']
            if vasp_custm_confs == {}:
                vasp_custm_confs = None 
            if (fast_relax_steps is None) or (fast_relax_steps is 'default'):
                fast_relax_steps = 0
            create_opt_dataset(initdir,nsamples,dftopt_confs,
                    self.Z_of_type,self.specorder,
                    continue_flag=continue_flag,outdir=outdir,
                    vasp_mode=vasp_mode,vasp_custm_confs=vasp_custm_confs,
                    fast_relax_steps=fast_relax_steps)
  

    def taylor3D(self):
        def _rewrite_list(initpaths):
            dirpaths = set()
            for initpath in initpaths:
                dirpaths.add(os.path.dirname(initpath))
            for dirpath in dirpaths:
                listpath = os.path.join(dirpath,'list')
                if os.path.exists(listpath):
                    filepaths = read_list(listpath)
                    old_filepaths = set(filepaths)
                    new_filepaths = set()
                    for filepath in os.listdir(dirpath):
                        if filepath.split('.')[-1] == 'xsf':
                            new_filepaths.add(filepath)
                    if not old_filepaths == new_filepaths:
                        write_list(listpath,list(new_filepaths))
        
        def Taylor3D(tay, seed, tp, initpaths, tay_dir):
            from aenet.AenetLab.create import create_3Dtaylor
            taylor_confs = {'fcut':tay['fcut'],'nfirst':tay['nfirst'],'seed':seed,'tp':tp}
            drmax = tay['drmax']
            nsamples = tay['nsamples']
            auglist=create_3Dtaylor(initpaths,drmax,nsamples,taylor_confs,tay_dir)
            _rewrite_list(initpaths)
            return auglist

        from aenet.AenetLab.connect import final_pool_fromlabs,get_set_fromlab
        outdir = self.tay['outdir']
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        tay_dir = '%s/%d_Taylorset'%(outdir,self.taskid)

        do_taylor = self.tay['do_taylor']
        dimension = self.tay['dimension']
        if dimension not in [1,3,'default']:
            raise Exception("Only 1,3 or 'default' are allowed when parsing 'dimension'")
        innerpaths = list(self.tay['innerpaths'].keys())
        extpaths   = list(self.tay['extpaths'].keys())
        
        if do_taylor:
            if os.path.exists('%s/.taylor_done'%tay_dir):
                print('================================')
                print('Taylor expansion already done\n')
            else:
                if dimension == "default":
                    dimension = 3
                if dimension == 1:
                    ####### Only for Test#########
                    from aenet.AenetLab.create import create_1Dtaylor
                    tay_dir = '01-Taylor/LJTaylor'
                    dr = self.tayd['1D']['dr']
                    _ = create_1Dtaylor(initdir,E_atom,dr,tay_dir)
                elif dimension == 3:
                    tay_extpaths,notay_extpaths = [],[]
                    tay_innerpaths,notay_innerpaths = [],[]
                    absinp_list = get_set_fromlab(os.getcwd())
                    for extpath in extpaths:
                        absexp = os.path.abspath(extpath)
                        if self.tay['extpaths'][extpath]:
                            tay_extpaths.append(absexp)
                        else:
                            notay_extpaths.append(absexp)
                    for innerpath in innerpaths:
                        absinp = os.path.abspath(innerpath)
                        if self.tay['innerpaths'][innerpath]:
                            tay_innerpaths.append(absinp)
                        else:
                            notay_innerpaths.append(absinp)
                        if absinp in absinp_list:
                            absinp_list.remove(absinp)
                    tay_innerpaths += absinp_list
                    
                    if innerpaths == []:
                        tay_initpaths_dic = final_pool_fromlabs(self.tp,tay_extpaths)
                        notay_initpaths_dic = final_pool_fromlabs(self.tp,notay_extpaths,onlyext=True)
                    else:
                        tay_initpaths_dic = final_pool_fromlabs(self.tp,tay_extpaths,inner=tay_innerpaths)
                        notay_initpaths_dic = final_pool_fromlabs(self.tp,notay_extpaths,inner=notay_innerpaths)

                    tay_initpaths = list(tay_initpaths_dic.keys())
                    notay_initpaths = list(notay_initpaths_dic.keys())
                    llist = list(notay_initpaths_dic.values())
                    tayd = self.tayd['3D']
                    tay_auglist=Taylor3D(tayd,self.tseed,self.tp,tay_initpaths,os.path.abspath(tay_dir))
                    notay_auglist = list(zip(notay_initpaths,llist))
                    aug_list = tay_auglist + notay_auglist
                    write_info('%s/info'%tay_dir,aug_list)
                write_isfl('%s/.taylor_done'%tay_dir,True)
                print('================================')
                print('Taylor expansion done\n')
        else:
            if os.path.exists('%s/.taylor_done'%tay_dir):
                print('================================')
                print('Ignore Taylor expansion and connection\n')
            else:
                if not os.path.exists(tay_dir):
                    os.makedirs(tay_dir)
                abs_ext = []
                for extpath in extpaths:
                    abs_ext.append(os.path.abspath(extpath))
                initpaths_dic = final_pool_fromlabs(self.tp,abs_ext)
                initpaths = list(initpaths_dic.keys())
                llist = list(initpaths_dic.values())
                aug_list = list(zip(initpaths,llist))
                write_info('%s/info'%tay_dir,aug_list)
                write_isfl('%s/.taylor_done'%tay_dir,True)
                print('================================')
                print('Ignore Taylor expansion! The connection was done\n')


    def generate(self,onlywrite=False):
        from aenet.AenetLab.connect import final_pool_fromTaylor
        from aenet.AenetLab.generate import write_gen_infile

        if not self.init:
            raise Exception('AenetLab obj is not initialized\n')

        outdir = self.gen['outdir']
        if not os.path.isdir(outdir):
            os.makedirs(outdir)
        elements   = list(self.types.keys())
        stp_styles = [self.types[elem]['stp_style'] for elem in self.types.keys()]
        gen_done   = False
        gen_oldnum = -1
        do_tay     = self.tay['do_taylor']
        gen_infile = self.gen['infile']
        gen_info   = self.gen['info']
        gen_output = self.gen['output']
        gen_timing = self.gen['timing']
        gen_debug  = self.gen['debug']

        E_atom  = [self.types[elem]['E_atom'] for elem in elements]
        E_unit  = self.units['E_unit']
        stp     = [f'{e}.stp' for e in elements]
        tad     = {'TIMING':False,'DEBUG':False}
        if os.path.exists('%s/.isdone'%outdir):
            gen_done = read_isfl('%s/.isdone'%outdir)
        if gen_timing:
            tad['TIMING'] = True
        if gen_debug:
            tad['DEBUG'] = True

        logic_dic = final_pool_fromTaylor('01-Taylor/%d_Taylorset'%self.taskid)
        xsf_paths = list(logic_dic.keys())
        n = len(xsf_paths)
        max_dirindex = get_max_dirindex(outdir,'gen')
        if max_dirindex == self.taskid - 1:
            if do_tay or self.taskid == 1:
                for ind,stp_style in enumerate(stp_styles):
                    if stp_style == 'Spherical_Chebyshev':
                        from aenet.AenetLab.generate import write_SC_stp
                        elem = elements[ind]
                        SC_args = self.stpd[stp_style]
                        SC_args['rcut'] = self.rrgn['rcut']
                        SC_args['rmin'] = self.rrgn['rmin']
                        SC_args['outd'] = outdir
                        SC_args['envs'] = elements
                        SC_args['e']    = elem
                        write_SC_stp(**SC_args)
            
                    elif self.stpstyle == 'Spherical':
                        #TODO : 'Spherical' style is not tested here!
                        pass

                    elif self.stpstyle == 'Chebyshev':
                        #FIXME : 'Chebyshev' style in some old versions(<    =2.4.4) of aenet is not reliable! 
                        pass

                    elif self.stpstyle in ['behler2011','behler2011r','be    hler2011c','behler2011p']:
                        #TODO : 'behler' style is not tested here!
                        pass

                write_gen_infile(outdir,gen_infile,gen_output,elements,E_atom,E_unit,stp,tad,n,xsf_paths)
                print('================================')
                print('Generation started')
                if not onlywrite:
                    gd_0 = subprocess.check_call('cd %s;'%outdir+'generate.x %s > %s\n'%(gen_infile,gen_info),shell=True)
                    print('Generation done\n')
                else:
                    gd_0 = 0
                    print('Only write infiles for generation\n')
                gen_dir = '%s/%d_gen'%(outdir,self.taskid)
                if not os.path.exists(gen_dir):
                    os.makedirs(gen_dir)
                for name in os.listdir(outdir):
                    npath = os.path.join(outdir,name)
                    if name != gen_output and not os.path.isdir(npath):
                        shutil.move(npath,gen_dir)
                if not gd_0:                        
                    write_isfl('%s/.isdone'%outdir,True)
            else:
                print('===============================')
                if not onlywrite:
                    print('Generation already done\n')
                else:
                    print('Nothing else needs to be written for generation\n')
        elif max_dirindex == self.taskid:
            print('===============================')
            if not onlywrite:
                print('Generation already done\n')
            else:
                print('Nothing else needs to be written for generation\n')
        return logic_dic


    def train(self, onlywrite=False, earlystop=False,
            index=None, epoch=None, ignore_id=False):
        from aenet.AenetLab.connect import final_pool_fromTaylor
        from aenet.AenetLab.train import write_trn_infile,write_trn_setsplit,read_energy_error_info,continue_energy_error_info

        if not self.init:
            raise Exception('AenetLab obj is not initialized\n')

        outdir = self.trn['outdir']
        
        if not os.path.isdir(outdir):
            os.makedirs(outdir)

        do_tay  = self.tay['do_taylor']
        if self.trn['initdir'] is None:
            initdir = self.gen['outdir']
            trn_set = self.gen['output']
        else:
            initdir = self.trn['initdir']
            trn_set = self.gen['output']+'.scaled'
        trn_set_path = os.path.abspath(os.path.join(initdir,trn_set))
        trn_done    = False
        trn_infile  = self.trn['infile']
        trn_info    = self.trn['info']
        trn_tp      = self.tp
        trn_steps   = self.trn['iterations']
        trn_method  = self.trn['method']
        trn_margs   = self.trnd[trn_method]
        trn_options = {'TIMING':self.trn['timing'],'DEBUG':self.trn['debug'],
                'SAVE_ENERGIES':self.trn['save_energies'],'NN_RSEED':self.trn['nn_rseed'],
                'SCR_PATH':self.trn['scr_path']}
        if not self.tay['do_taylor']:
            trn_options['MAXENERGY'] = self.trn['maxenergy']

        done_file = '%s/.isdone'%outdir
        trn_done_exist = False
        if os.path.exists(done_file):
            trn_done_exist = True
            trn_done  = read_isfl(done_file)

        netargs = []
        elems = list(self.types.keys())
        for elem in elems:
            elem_dic = self.types[elem]                                     
            netargs.append([elem,elem_dic['nnodes'],elem_dic['acfuns']])

        if not earlystop:
            trn_set_symb = os.path.join(outdir,trn_set)
            if not os.path.exists(trn_set_symb):
                subprocess.call('ln -s %s %s'%(trn_set_path,trn_set_symb),shell=True)
            logic_dic = final_pool_fromTaylor('01-Taylor/%d_Taylorset'%self.taskid)
            logic_list = [v for v in logic_dic.values()]
            _ = write_trn_infile(outdir,trn_infile,trn_set,trn_tp,trn_steps,trn_options,trn_method,trn_margs,netargs)

        if onlywrite:
            if self.trn['initdir'] is None:
                write_trn_setsplit(outdir,logic_list)
            else:
                subprocess.check_call('cp %s/train.trnsplit %s'%(self.trn['initdir'],self.trn['outdir']))
            print('===============================')
            print('Only write infiles for training\n')
        else:
            def _get_max_index(outdir,elems): 
                max_index = 0
                for na in os.listdir(outdir):
                    na_split_list = na.split('-')
                    if na_split_list[0].split('.')[0] in elems and len(na_split_list)>2:
                        max_index = max(max_index,int(na_split_list[-1]))
                return max_index

            def _renumber_nn_steps(nstop,outdir,max_index):
                 nn_nas = set()
                 for na in os.listdir(outdir):
                     na_split = na.split('.nn-')
                     if len(na_split)==2:
                         nn_nas.add(na_split[0]+'.nn-')
                 nn_nas = list(nn_nas)
                 for nn_na in nn_nas:
                     for i in range(max_index,1,-1):
                         old_name = nn_na + '%05d'%i
                         new_name = nn_na + '%05d'%(nstop+i)
                         subprocess.check_call('cd %s;'%outdir+'mv %s %s'%(old_name,new_name),shell=True)              

            max_dirindex = get_max_dirindex(outdir,'nnpots') # from aenet_io
            if not ignore_id:
                assert max_dirindex + 1 == self.taskid
            max_index = _get_max_index(outdir,elems)
               
            scr_path = self.trn['scr_path']
            if scr_path:
                scr_path = os.path.join(outdir,scr_path)
                if not os.path.isdir(scr_path):
                    os.makedirs(scr_path)

            if index:
                real_ind = index
            else:
                real_ind = self.taskid

            fcm_flag = False
            for nf in os.listdir(outdir):
                nf_p = os.path.join(outdir,nf)
                if (not os.path.isdir(nf_p)) and ('train' in nf):
                    fcm_flag = True

            clean_command = 'find . -type f -name "TEST*" -delete;'+\
                            'find . -type f -name "TRAIN*" -delete;'  
            hide_details_command = 'find . -maxdepth 1 -type f -name "*.nn-*" | xargs mv -t %d_step_details;'%self.taskid +\
                    'find . -maxdepth 1 -type f -name "%s" | xargs mv -t %d_step_details;'%(trn_info,self.taskid)
            move_nnfiles_command = 'find . -maxdepth 1 -type f -name "*.nn" | xargs mv -t %d_nnpots;'%self.taskid
            finalization_command = 'find . -type f -name "%s*" -delete;'%trn_set +\
                    'find . -type f -name "train.trnsplit" -delete;'
            if fcm_flag:
                finalization_command += 'mv %s/train* %s/%d_step_details'%(outdir,outdir,real_ind)
            #tianhe2 command style
            run_train_command = 'srun train.x %s > %s;'%(trn_infile,trn_info)
            #Other command styles such as mpirun are also supported
            #Just modify MPI executable and its keywords in the 'run_train_command' batch above
 
            if earlystop:
                if os.path.exists('%s/%s'%(outdir,trn_info)):
                    sd_dir = '%s/%d_step_details'%(outdir,self.taskid)
                    nn_dir = '%s/%d_nnpots'%(outdir,self.taskid)
                    make_empty_dir(nn_dir)
                    if os.path.exists('%s/%s'%(sd_dir,trn_info)):
                        continue_energy_error_info(sd_dir,outdir,trn_info)
                    if not os.path.exists('%s/.stopstep'%outdir):
                        stopstep = max_index
                    else:
                        stopstep = read_isfl('%s/.stopstep'%outdir)
                    if stopstep:
                        _renumber_nn_steps(stopstep,outdir,max_index)
                    subprocess.call('cd %s;'%outdir+'%s'%hide_details_command+\
                        '%s'%move_nnfiles_command+'%s'%clean_command+'rm -f .stopstep',shell=True)
                    
                    _, train_values, test_values = read_energy_error_info(sd_dir,trn_info)
                    if not epoch:
                        min_epoch = np.argmin(test_values)
                    else:
                        min_epoch = epoch
                    min_train = train_values[min_epoch]
                    min_test  = test_values[min_epoch]
                    find_minima_command = "for i in `ls %s | grep '.nn-%05d'"%(sd_dir,min_epoch)+\
                            ' | awk -F".nn-%05d"'%min_epoch+" '{print $1}'`;"+\
                            ' do cp %s/$i.nn-%05d %s/$i.nn; done'%(sd_dir,min_epoch,nn_dir)
                    subprocess.call(find_minima_command,shell=True)
                    write_isfl('%s/.isdone'%outdir,True)
                    if not ignore_id:
                        write_isfl('.taskid',self.taskid+1)
                else:
                    if not index:
                        raise TypeError('Please specify index of nnpots directory!')
                    else:
                        if not (index >= 1 or index == -1):
                            raise IndexError('Nnpots index out of range( int>=1 or == -1)')
                        elif index == -1:
                            index = max_dirindex
                    sd_dir = '%s/%d_step_details'%(outdir,index)
                    nn_dir = '%s/%d_nnpots'%(outdir,index)
                    make_empty_dir(nn_dir)
                    _, train_values, test_values = read_energy_error_info(sd_dir,trn_info)
                    if not epoch:
                        min_epoch = np.argmin(test_values)
                    else:
                        min_epoch = epoch
                    min_train = train_values[min_epoch]
                    min_test  = test_values[min_epoch]
                    find_minima_command = "for i in `ls %s | grep '.nn-%05d'"%(sd_dir,min_epoch)+\
                            ' | awk -F".nn-%05d"'%min_epoch+" '{print $1}'`;"+\
                            ' do cp %s/$i.nn-%05d %s/$i.nn; done'%(sd_dir,min_epoch,nn_dir)
                    subprocess.call(find_minima_command,shell=True)
                with open('%s/info'%nn_dir,'w') as fin:
                    print('Earlystop at:',file=fin)
                    print('epoch        TRAIN(rmse)         TEST(rmse)',file=fin)
                    print('%5d        %e        %e'%(min_epoch,min_train,min_test),file=fin)
                print('===============================')
                print('Earlystop at: %d    RMSE(test): %e'%(min_epoch,min_test))
                print('Earlystop done')
                if do_tay:
                    subprocess.call(finalization_command,shell=True) 
                os._exit(0)

            if not trn_done:
                print('===============================')
                ff = False
                if not os.path.exists('%s/.stopstep'%outdir):
                    ff_stop = False
                    if os.path.exists('%s/%s'%(outdir,trn_info)):
                        ff_stop = True
                    if not ff_stop:
                        if self.trn['initdir'] is None:
                            write_trn_setsplit(outdir,logic_list)
                        else:
                            shutil.copy('%s/train.trnsplit'%self.trn['initdir'],\
                                    '%s/train.trnsplit'%self.trn['outdir'])
                    ff = True
                    stopstep = max_index
                    step_dir = '%d_step_details'%(max_dirindex+1)
                    make_empty_dir('%s/%s'%(outdir,step_dir))
                    write_isfl('%s/.stopstep'%outdir,stopstep)
                    if ff_stop:
                        print('Training( index %d ) continued'%(max_dirindex+1))
                    else:
                        print('Training( index %d ) started'%(max_dirindex+1))
                else:
                    stopstep = read_isfl('%s/.stopstep'%outdir)
                    if stopstep:
                        _renumber_nn_steps(stopstep,outdir,max_index)
                    stopstep += max_index
                    write_isfl('%s/.stopstep'%outdir,stopstep)
                    print('Training( index %d ) continued'%(max_dirindex+1))
                sd_dir = '%s/%d_step_details'%(outdir,max_dirindex+1)
                if os.path.exists('%s/%s'%(sd_dir,trn_info)):
                    continue_energy_error_info(sd_dir,outdir,trn_info)
                if ff:
                    subprocess.check_call('cd %s;'%outdir+clean_command+\
                             run_train_command,shell=True)
                else:
                    subprocess.check_call('cd %s;'%outdir+hide_details_command+\
                        clean_command+run_train_command,shell=True)
                    max_index = _get_max_index(outdir,elems)
                    _renumber_nn_steps(stopstep,outdir,max_index)
                if os.path.exists('%s/%s'%(sd_dir,trn_info)):
                    continue_energy_error_info(sd_dir,outdir,trn_info)
                os.makedirs('%s/%d_nnpots'%(outdir,max_dirindex+1))
                subprocess.call('cd %s;'%outdir+hide_details_command+\
                        move_nnfiles_command+'rm -f .stopstep',shell=True)
                write_isfl('%s/.isdone'%outdir,True)
                print('Training done\n')
                if do_tay:
                    subprocess.call(finalization_command,shell=True)
                if not ignore_id:
                    write_isfl('.taskid',self.taskid+1)
            else:
                print('===============================')
                print('A new training(index %d) started'%(max_dirindex+1))
                if self.trn['initdir'] is None: 
                    write_trn_setsplit(outdir,logic_list)
                else:
                    subprocess.check_call('cp %s/train.trnsplit %s'%(self.trn['initdir'],self.trn['outdir']))
                write_isfl('%s/.isdone'%outdir,False)
                step_dir = '%d_step_details'%(max_dirindex+1)
                make_empty_dir('%s/%s'%(outdir,step_dir))
                subprocess.check_call('cd %s;'%outdir+'rm -f train.time train.restart train.rngstate;'+run_train_command,shell=True)
                subprocess.call('cd %s;'%outdir+hide_details_command+\
                        'mkdir %d_nnpots;'%(max_dirindex+1)+move_nnfiles_command,shell=True)
                write_isfl('%s/.isdone'%outdir,True)
                print('Training done\n')
                if do_tay:
                    subprocess.call(finalization_command,shell=True)
                if not ignore_id:
                    write_isfl('.taskid',self.taskid+1)
