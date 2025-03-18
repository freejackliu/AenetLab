# Fortran is case-insensitive, thus the keyword can be in both types. 
# For a tidy layout, we all use capital ones.
import os
import numpy as np
from aenet.xsf import *


template_trnin = """TRAININGSET {}
TESTPERCENT {}
ITERATIONS {}
{}
METHOD
{}

NETWORKS
{}
{}
"""

Logic_py2fort = {False:'.false.',True:'.true.'}

def write_trn_infile(outd,infile,trnset,tp,iter,options,mn,margs,netargs):
    if isinstance(tp,str):
        tp_final = tp.strip('%')
    elif isinstance(tp,float):
        if 0. <= tp <= 1.:
            tp_final = tp*100
        else:
            raise Exception('"test_percent" should be in range of [0,1]')
    else:
        raise TypeError('Expected type of "test_percent" : float or str , e.g. 0.15 or "15%"')

    options_tmp = []
    for option_key in list(options.keys()):
        key_v = options[option_key]
        if isinstance(key_v,bool):
            options_tmp.append('%s %s'%(option_key,Logic_py2fort[key_v]))
        elif key_v == None:
            pass
        else:
            options_tmp.append('%s %s'%(option_key,key_v))
    options_final = '\n'+'\n'.join(options_tmp)+'\n'

    m_final = mn
    for argkey in list(margs.keys()):
        if margs[argkey] != 'default':
            m_final += ' %s %s'%(argkey,margs[argkey])

    pf_dic = {}
    netargs_tmp = []
    max_namelen = len('file-name')
    for netarg in netargs:
        elem    = netarg[0]
        hl_st   = ['%sh'%hln for hln in netarg[1]]
        netfile = '.'.join([elem,'-'.join(hl_st),'nn'])
        hl_n    = len(netarg[1])
        n_a     = ' '.join(['%s:%s'%(netarg[1][i],netarg[2][i]) for i in range(hl_n)])
        pf_dic[elem]=netfile
        if len(netfile)>max_namelen:
            max_namelen = len(netfile)
        netargs_tmp.append([elem,netfile,hl_n,n_a])
    netargs_tmp2 = []
    for netarg_tmp in netargs_tmp:
        netargs_tmp2.append('  '+'%-7s'%netarg_tmp[0]+' '*(max_namelen-len(netarg_tmp[1]))+\
                '%s'%netarg_tmp[1]+'   %-6s'%netarg_tmp[2]+'  %s'%netarg_tmp[3])
    netargs_final = '\n'.join(netargs_tmp2)

    netd = f'! atom   %-{max_namelen+3}s'%'network  '+'hidden\n'+\
           f'! types  %-{max_namelen+3}s'%'file-name'+'layers  nodes:activation'

    op = template_trnin.format(trnset,tp_final,iter,options_final,m_final,netd,netargs_final)
    with open(f"{outd}/{infile}", "w") as o:
        o.write(op)
    return pf_dic

def write_trn_setsplit(outd,logic_list):
    import struct
    with open(f"{outd}/train.trnsplit","wb") as ts:
        ts.write(struct.pack("i",len(logic_list)))
        for logic_i in logic_list:
            ts.write(struct.pack("i",logic_i))

def read_energy_error_info(outd,trn_info,mode='rmse'):
    info_path = '%s/%s'%(outd,trn_info)
    with open(info_path,'r') as ip:
        lines = ip.readlines()
        energy_error_batch = False
        nn_steps = []
        train_values = []
        test_values  = []
        for line in lines:
            if line.split():
                if line.split()[0] == 'epoch':
                    energy_error_batch = True
                if energy_error_batch and line.split()[0] != 'epoch':
                    try:
                        nn_steps.append(int(line.split()[0]))
                        if mode == 'rmse':
                            train_values.append(float(line.split()[2]))
                            test_values.append(float(line.split()[4]))
                        elif mode == 'mae':
                            train_values.append(float(line.split()[1]))
                            test_values.append(float(line.split()[3]))
                        elif mode == 'both':
                            train_values.append([float(line.split()[1]),float(line.split()[2])])
                            test_values.append([float(line.split()[3]),float(line.split()[4])])
                    except ValueError:
                        energy_error_batch = False
        return nn_steps, train_values, test_values

def continue_energy_error_info(old,new,trn_info):
    ns_old, tr_old, te_old = read_energy_error_info(old,trn_info,mode='both')
    ns_new, tr_new, te_new = read_energy_error_info(new,trn_info,mode='both')
    s = ''
    count = 0
    for old_i in zip(ns_old, tr_old, te_old):
        s += '%6d'%old_i[0]+'    %E'%old_i[1][0]+'    %E'%old_i[1][1]+\
                '    %E'%old_i[2][0]+'    %E <\n'%old_i[2][1]
        count += 1
    for new_i in zip(ns_new[1:], tr_new[1:], te_new[1:]):
        s += '%6d'%(new_i[0]+count-1)+'    %E'%new_i[1][0]+'    %E'%new_i[1][1]+\
                '    %E'%new_i[2][0]+'    %E <\n'%new_i[2][1]
    with open('%s/%s'%(new,trn_info),'r') as newf:
        lines = newf.readlines()
        energy_error_batch = False
        _ = []
        ch_start = -1
        ch_end   = -1
        for i,line in enumerate(lines):
            if line.split():
                if line.split()[0] == 'epoch':
                    ch_start = i+1
                    energy_error_batch = True
                if energy_error_batch and line.split()[0] != 'epoch':
                    try:
                        _.append(int(line.split()[0]))
                    except ValueError: 
                        energy_error_batch = False
                        ch_end = i
    with open('%s/%s'%(new,trn_info),'w') as bakf:
        for i,line in enumerate(lines):
            if i<ch_start:
                bakf.write(lines[i])
            elif i==ch_start:
                bakf.write(s)
            elif i>=ch_end:
                if ch_end != -1:
                    bakf.write(lines[i])

