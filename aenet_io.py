import os
import numpy as np
from ase.io import write,read
from ase.io.lammpsdata import read_lammps_data
from aenet.xsf import read_xsf
from aenet.read_dump import read_lammps_dump_text
str2bool = {'True':True,'False':False}
add_keys = [
        ['CreateDetails','VaspCustomized'],
        ['Taylor','innerpaths'],
        ['Taylor','extpaths']
        ]
types_template = {
        "lammps_number" : 1,
        "E_atom"        : "default",
        "stp_style"     : "Spherical_Chebyshev",
        "nnodes"        : [30,30],
        "acfuns"        : ['swish','swish']
        }


def onlybase(paths):
    names = [os.path.basename(path) for path in paths]
    return names


def read_list(list_path,relative=False):
    a_list_path = os.path.abspath(list_path)
    dir_path = os.path.abspath(os.path.dirname(list_path))
    with open(a_list_path,'r') as fl:
        first_line = fl.readline()
        try:
            _ = int(first_line.strip())
        except:
            if not relative:
                xsf_paths = [os.path.join(dir_path,first_line.strip())]
            else:
                xsf_paths = [first_line.strip()]
        else:
            xsf_paths = []
        if not relative:
            xsf_paths += [os.path.join(dir_path,line.strip()) for line in fl.readlines()]
        else:
            xsf_paths += [line.strip() for line in fl.readlines()]
        return xsf_paths


def write_list(list_path,rpaths):
    with open(list_path,'w') as fl:
        fl.write('%d\n'%len(rpaths))
        for rpath in rpaths:
            fl.write('%s\n'%rpath)


def read_info(info_path):
    with open(info_path,'r') as fi:
        _ = int(fi.readline().strip())
        lines = fi.readlines()
        path_bool = {}
        for line in lines:
            path_bool[line.strip().split()[0]]=str2bool[line.strip().split()[1]]
        return path_bool


def write_info(info_path,aug_list):
    with open(info_path,'w') as fi:
        fi.write('%d\n'%len(aug_list))
        for ag in aug_list:
            fi.write('%s %s\n'%(ag[0],ag[1]))


def read_isfl(isfl_path):
    with open(isfl_path,'r') as ifl:
        str_txt = ifl.readline().strip()
        if str_txt in ['True','False']:
            return str2bool[str_txt]
        elif str_txt == 'None':
            return None
        else:
            return int(str_txt)


def write_isfl(isfl_path,bl):
    with open(isfl_path,'w') as ifl:
        ifl.write('%s\n'%bl)


def find_env_path():
    #get AenetLab env-path:
    env_path = set()
    for path in os.getenv('PATH').split(':'):
        if path[-1] != '/':
            if path.split('/')[-1] == 'AenetLab':
                env_path.add(path)
        else:
            if path.split('/')[-2] == 'AenetLab':
                env_path.add(path) 
    env_path = list(env_path) 
    if len(env_path) > 1: 
        raise Exception("More than one env-path for AenetLab has been found")
    elif len(env_path) == 0:
        raise Exception("Can't find env-path for AenetLab")
    else:
        env_path = env_path[0]
        return env_path


def load_atomic_info():
    import json
    env_path = find_env_path()
    #load json:
    json_file = os.path.join(env_path,'jsons/atomic_info.json')
    with open(json_file,'r') as aif:
        textjson = json.load(aif)
        return textjson


def load_template():
    import json
    env_path = find_env_path()      
    #load json: 
    json_file = os.path.join(env_path,'jsons/template.json')
    with open(json_file,'r') as tf: 
        textjson = json.load(tf)
        return textjson


def load_nnfiles(nnpath):
    potentials = {}
    for n in os.listdir(nnpath):
        nsp = n.split('.')
        if nsp[-1]=='nn':
            elem = nsp[0]
            potentials[elem] = os.path.join(nnpath,n)
    return potentials


def walk(dic):
    for k, v in dic.items():
        if isinstance(v,dict):
            for tup in walk(v):
                yield (k,) + tup
        else:
            yield k,v


def check_add_keys(keyvalue):
    for add_key in add_keys:
        is_add_key = True
        for i, key in enumerate(add_key):
            if key != keyvalue[i]:
                is_add_key = False
        if is_add_key:
            return True


def deep_update(old_d,new_list):
    new_d = old_d.copy()
    old_elem = ''
    for ind, keyvalue in enumerate(new_list):
        try:
            fc = keyvalue[0]
        except TypeError:
            raise TypeError("Invalid args for 'set', type list or tuple is expected")
        else:
            if len(keyvalue) == 1:
                raise Exception("Invalid length of 'key-value' list, expected 2 got 1")
            elif len(keyvalue) == 2:
                new_d[fc] = keyvalue[-1]
            else:
                if fc != 'Types':
                    if check_add_keys(keyvalue):
                        dic_list = []
                        d = new_d[fc]
                        for ik in range(1,len(keyvalue)-2):
                            dic_list.append(d)
                            d=d[keyvalue[ik]]
                        d[keyvalue[-2]] = keyvalue[-1]
                        for ik in range(len(dic_list)-2,-1,-1):
                            dic_list[ik][keyvalue[ik+1]]=d
                            d = dic_list[ik]
                        new_d[fc] = dic_list[0]
                    else:
                        dic_list = []
                        d = new_d[fc]
                        for ik in range(1,len(keyvalue)-1):
                            dic_list.append(d)
                            d=d[keyvalue[ik]]
                        v = keyvalue[-1]
                        for ik in range(len(dic_list)-1,-1,-1):
                            dic_list[ik][keyvalue[ik+1]]=v          
                            v = dic_list[ik]
                        new_d[fc] = dic_list[0]
                else:
                    elem,keyw = keyvalue[1],keyvalue[2]
                    if old_elem != elem:
                        new_d[fc][elem] = types_template.copy()
                    if keyw in types_template.keys():
                        new_d[fc][elem][keyw] = keyvalue[3]
                    old_elem = elem
    return new_d


def get_value(elem):
    textjson = load_atomic_info()
    atomic_number = textjson[elem][0]
    E_atom = textjson[elem][1]
    return atomic_number,E_atom


def get_max_dirindex(outdir,keytag):
    max_dirindex = 0
    for na in os.listdir(outdir):
        na_split_list = na.split('_')
        if len(na_split_list)==2 and na_split_list[-1]==keytag:
            max_dirindex = max(max_dirindex,int(na_split_list[0]))
    return max_dirindex

     
def ase_lammps_read(init_file,Z_of_type,index,sf):
    if isinstance(init_file,str):
        f = open(init_file,'r')
    keys = Z_of_type.keys()
    if sf == 'data':
        atoms = read_lammps_data(f,Z_of_type=Z_of_type,style='atomic')
    elif sf == 'dump':
        specorder = [Z_of_type[key] for key in keys]
        atoms = read_lammps_dump_text(f,index,specorder=specorder)
    elif sf == 'bindump':
        raise Exception("AenetLab can not load binary dump file now")
    #    specorder = [Z_of_type[key] for key in Z_of_type.keys()]
    #    atoms = read_lammps_dump_binary(init_file,specorder=specorder)
    return atoms


def aread(init_file,Z_of_type,index_list):
    tag = os.path.basename(init_file)
    if np.isscalar(index_list):
        if index_list == "default":
            index = -1
        elif index_list == ":":
            index = slice(0,None,1)
        else:
            index =index_list
    elif isinstance(index_list,list):
        if len(index_list) == 1:
            index = slice(index_list[0])
        elif len(index_list) == 2:
            index = slice(index_list[0],index_list[1])
        elif len(index_list) == 3:
            index = slice(index_list[0],index_list[1],index_list[2])
        else:
            raise ValueError("Invalid length of 'index_list' arg,check the doc of slice obj in python") 
    p_suffix = set(tag.split('.'))
    lammps_suffix = set(['dump','data','bindump'])
    vasp_suffix = set(['xml'])
    if set.isdisjoint(p_suffix,lammps_suffix):
        if set.isdisjoint(p_suffix,vasp_suffix):
            if tag.split('.')[-1] == 'xsf':
                ar = [read_xsf(init_file)]
            else:
                ar = read(init_file,index=index)
        else:
            ar = read(init_file,format='vasp-xml',index=index)
        tag = tag.split('.'+tag.split('.')[-1])[0]
    else:
        assert len(p_suffix & lammps_suffix) == 1
        suffix = p_suffix & lammps_suffix
        sf = list(suffix)[0]
        ar = ase_lammps_read(init_file,Z_of_type,index,sf)
        p_tag = ''.join((x for x in tag.split('.'+sf)))
        if p_tag == tag:
            tag =  ''.join((x for x in p_tag.split(sf+'.')))
        else:
            tag = p_tag
    return ar, tag


def make_empty_dir(dirname):
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    else:
        os.system('rm -rf %s'%dirname)
        os.makedirs(dirname)

