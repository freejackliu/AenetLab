import os
import numpy as np
from math import ceil
from aenet.AenetLab.aenet_io import read_list,read_isfl,read_info

def get_set_fromlab(labpath):
    names = os.listdir(labpath)
    origin_paths = []
    for na in names:
        dirpath = os.path.join(labpath,na)
        dataflag_path = os.path.join(dirpath,'.isdataset')
        if os.path.isdir(dirpath) and os.path.exists(dataflag_path):
            if read_isfl(dataflag_path):
                origin_paths.append(dirpath)
    return origin_paths


def pool_fromlab(lab_path):
    origin_paths = get_set_fromlab(lab_path)
    xsf_paths = []
    if len(origin_paths):
        for origin_path in origin_paths:
            xsf_paths_i = pool_fromset(origin_path)
            xsf_paths += xsf_paths_i
    return xsf_paths


def pool_fromset(set_path):
    xsf_paths = []
    list_p = os.path.join(set_path,'list')
    if os.path.exists(list_p):
        fnames = read_list(list_p)
    else:
        fnames = os.listdir(set_path)
    for fname in fnames:
        if fname.split('.')[-1]=='xsf':
            xsf_path = os.path.join(set_path,fname)
            xsf_paths.append(xsf_path)
    return xsf_paths


def pool_fromtay(tay_path,ldic):
    lklist = list(ldic.keys())
    final_dic, xsfs_names = ldic.copy(), []
    for lk in lklist:
        xsfs_names.append(os.path.basename(lk))
    taydic = read_info(os.path.join(tay_path,'tayinfo'))
    for tk, tv in taydic.items():
        if xsfs_names.count(tk) == 0:
            final_dic[os.path.join(tay_path,tk)] = tv
        elif xsfs_names.count(tk) == 1:
            final_dic[lklist[xsfs_names.index(tk)]] = tv
        else:
            raise Exception("Each xsf in 'list' must have an unique name!")
    return final_dic


def decide_test(tp,xsfs):
    logic_dic = dict()
    if os.path.exists('tseed'):
        seed = read_isfl('tseed')
    else:
        raise Exception('Cannot find the seed file for trainset splitting')
    if isinstance(tp,str):
        test_percent = float(tp.strip('%'))/100
    elif isinstance(tp,float):
        if 0. <= tp <= 1.:
            test_percent = tp
        else:
            raise Exception("'test_percent' should be in range of [0,1]")
    else:
        raise TypeError("Expected type of 'test_percent' : float or str , e.g. 15% or 0.15")
    rng = np.random.RandomState(seed)
    n_test = ceil(len(xsfs)*test_percent)
    istest = np.array([False for _ in xsfs])
    istest[0:n_test] = True
    rng.shuffle(istest)
    for k, xsf in enumerate(xsfs):
        logic_dic[xsf] = bool(istest[k])
    return logic_dic

def final_pool_fromTaylor(taylor_path):
    logic_dic = read_info(os.path.join(taylor_path,'info'))
    return logic_dic

def final_pool_fromlabs(tp,extpaths,inner=None,onlyext=False):
     tay_path = []
     for extpath in extpaths:
         if os.path.exists(os.path.join(extpath,'tayinfo')):
             tay_path.append(extpath)
     if len(tay_path) > 1:
         raise RuntimeError("Only one Taylor expansion dataset can be loaded, here got %d"%len(tay_path))

     xsf_paths, logic_dic = [], {}
     if inner:
         for inp in inner:
             xsf_paths_i = pool_fromset(inp)
             xsf_paths += xsf_paths_i
     else:
         if not onlyext:
             xsf_paths = pool_fromlab(os.getcwd())
     if not len(extpaths):
         logic_dic = decide_test(tp,xsf_paths)
         if len(tay_path):
             logic_dic = pool_fromtay(tay_path[0],logic_dic)
     else:
         for extpath in extpaths:
             if os.path.exists(os.path.join(extpath,'.islab')):
                 xsf_paths_i = pool_fromlab(extpath)
                 xsf_paths += xsf_paths_i
             else:
                 xsf_paths_i = pool_fromset(extpath)
                 xsf_paths += xsf_paths_i
         logic_dic = decide_test(tp,xsf_paths)
         if len(tay_path):
             logic_dic = pool_fromtay(tay_path[0],logic_dic)
     return logic_dic



