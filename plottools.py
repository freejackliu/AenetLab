import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from aenet.AenetLab.aenet_io import load_nnfiles


def _make_plotdir(dirpath):
    if not os.path.isdir(dirpath):
        os.makedirs(dirpath)

def _prepare(aelab,index):
    index_list = []
    trn_outdir = aelab.trn['outdir']
    plot_outdir = aelab.vis['outdir']
    for na in os.listdir(trn_outdir):
        na_slt = na.split('_')
        if len(na_slt)==2 and na_slt[-1]=='nnpots':
            index_list.append(int(na_slt[0]))
    if not len(index_list):
        raise NameError("No directory is named as the '?_nnpots' format")
    index_list = sorted(index_list)
    if isinstance(index,int):
        if index == -1:
            il = index_list[index]
        elif index > 0:
            il = index_list[index-1]
        else:
            raise IndexError('Nnpots index out of expected range( int>=1 or =-1 )')
        nn_files_path = os.path.join(trn_outdir,'%d_nnpots'%il)
        plot_path = os.path.join(plot_outdir,'%d_plot'%il)
        _make_plotdir(plot_path)
    else:
        raise TypeError("'index' must be int type")
    return il, nn_files_path, plot_path

def _prepare_without_index(aelab):
    from aenet.AenetLab.aenet_io import read_isfl,write_isfl

    plot_outdir = aelab.vis['outdir']
    _make_plotdir(plot_outdir)
    wi_file = os.path.join(plot_outdir,'.wi')
    if os.path.exists(wi_file):
        il = read_isfl(wi_file)
    else:
        il = 1
    write_isfl(wi_file,il+1)
    plot_path = os.path.join(plot_outdir,'wi%d_plot'%il)
    _make_plotdir(plot_path)
    return il, plot_path

def check_squmat(mat):
    squ_flag = True
    try:
        nrow = len(mat)
        ncol = len(mat[0])
    except:
        raise TypeError('Invalid Mat form, usually happens when ncol<2')
    else:
        for i in range(len(mat)):
            if len(mat[i]) != ncol:
                squ_flag = False
        return squ_flag, nrow, ncol

def get_bincen(bins):
    centers = np.zeros(len(bins)-1)
    for i in range(len(bins)-1):
        centers[i] = (bins[i]+bins[i+1])/2
    return centers

def save_dat(mat, path, label=None, bins2d=False):
    isqumat, nrow, ncol = check_squmat(mat)
    if isqumat:
        with open(path,'w') as f:
            if label:
                for label_i in label:
                    f.write('%s   '%label_i)
                f.write('\n')
            if bins2d:
                assert np.isnan(mat[0][0])
            for i in range(nrow):
                for j in range(ncol):
                    #if not np.isnan(mat[i][j]):
                    f.write('%e%3s'%(mat[i][j], ''))
                f.write('\n')

def plot_lj_dimer(aelab):
    pass#TODO

def plot_dimer(aelab,index=-1):
    from ase import Atoms
    from aenet.calculator import ANNCalculator

    il, nn_files_path, plot_path = _prepare(aelab,index)

    # plot args
    N = 200
    elems = list(aelab.types.keys())
    tags  = ''.join(elems)
    rcut  = aelab.rrgn['rcut']
    print('elems:',elems,'rcut:',rcut)
    L_unit = aelab.units['L_unit']
    E_unit = aelab.units['E_unit']
    pairs = []
    E_pairs = []
    for i,elem1 in enumerate(elems):
        for j,elem2 in enumerate(elems):
            if j>=i:
                pairs.append(''.join([elem1,elem2]))
                E_pairs.append(aelab.types[elem1]['E_atom']+aelab.types[elem2]['E_atom'])
    print('epairs:',E_pairs,'pairs:',pairs)
    d_line = np.linspace(0.001,rcut,N) 
    e_line = np.zeros((len(pairs),len(d_line)))
    mat = np.zeros((len(d_line),len(pairs)+1))
    mat[:,0] = d_line
    label = ['distance']
    
    potentials = load_nnfiles(nn_files_path)
    calc=ANNCalculator(potentials=potentials)
    for i,pair in enumerate(pairs):
        for j in range(N):
            pair_test = Atoms(pair, positions=[(0., 0., 0.), (0., 0., d_line[j])])
            pair_test.set_calculator(calc)
            e_line[i][j] = pair_test.get_potential_energy()-E_pairs[i]/2
        mat[:,i+1] = e_line[i,:]    
        label.append('E_'+pair)

    dat_path = '%s/%s-dimer.dat'%(plot_path,'%d_nnpots'%il)
    fig_path = '%s/%s-dimer.png'%(plot_path,'%d_nnpots'%il)
    save_dat(mat,dat_path,label)
    plt.figure(figsize=(6,6))
    ax=plt.subplot(111)
    for i,pair in enumerate(pairs):
        ax.plot(d_line,e_line[i],label=pair)
    plt.ylim((-10,25))
    plt.xlabel('Bond Length / %s'%L_unit)
    plt.ylabel('Cohesive Energy / %s'%E_unit)
    plt.legend()
    plt.savefig(fig_path)
    print('Plotting done')


def plot_error(aelab, E_err, Fa_err, angle_err,
     E_range=None, Fa_range=None, angle_range=None, index=None):
    if index:
        il, _, plot_path = _prepare(aelab, index)
    else:
        il, plot_path = _prepare_without_index(aelab)
    E_nbins = 100
    F_nbins = (100,50)
    F_range = None
    if Fa_range and angle_range:
        F_range = np.array([Fa_range, angle_range])

    E_unit = aelab.units['E_unit']
    F_unit = aelab.units['F_unit']
    angle_unit = aelab.units['angle_unit']
    E_hist, E_bins = np.histogram(E_err, bins=E_nbins, range=E_range,density=True)
    F_hist, Fa_bins, angle_bins = np.histogram2d(Fa_err, angle_err, bins=F_nbins, range=F_range, normed=True)

    E_fig = plt.figure(figsize=(10,6)) 
    E_ax = E_fig.add_subplot(111)
    E_binc = get_bincen(E_bins)
    E_ax.plot(E_binc, E_hist, '-o')
    E_label = ['E_bins','r-freq']
    E_mat = np.zeros((len(E_binc),2))
    E_mat[:,0] = E_binc
    E_mat[:,1] = E_hist
    plt.tick_params(labelsize=15)
    plt.xlabel('Absolute Atomic Energy Error (%s/atom)'%E_unit,fontsize=20)
    plt.ylabel('Relative Frequence',fontsize=20)
    E_figpath = os.path.join(plot_path, '%d_E_dist.png'%il)
    E_matpath = os.path.join(plot_path, '%d_E_dist.dat'%il)
    plt.savefig(E_figpath) 
    save_dat(E_mat, E_matpath, label=E_label)

    F_fig = plt.figure(figsize=(7,10)) 
    F_ax = F_fig.add_subplot(111)
    try:
        F_ax.hist2d(Fa_err, angle_err, bins=F_nbins, range=F_range, density=True)
    except AttributeError:
        #old matplotlib version 
        F_ax.hist2d(Fa_err, angle_err, bins=F_nbins, range=F_range, normed=True)
    Fa_binc = get_bincen(Fa_bins)
    angle_binc = get_bincen(angle_bins)
    F_label = ['[:,0]=Fa_bins','[0,:]=angle_bins','[1:,1:]=r-freq']
    F_mat = np.zeros((len(Fa_binc)+1,len(angle_binc)+1))
    F_mat[0,0] = np.nan
    F_mat[1:,0] = Fa_binc
    F_mat[0,1:] = angle_binc
    F_mat[1:,1:] = F_hist 
    plt.tick_params(labelsize=15)
    plt.xlabel('Absolute Atomic Force Error (%s)'%F_unit,fontsize=20)
    plt.ylabel('Error in Direction (%s)'%angle_unit,fontsize=20)
    F_figpath = os.path.join(plot_path, '%d_F_dist.png'%il)
    F_matpath = os.path.join(plot_path, '%d_F_dist.dat'%il)
    plt.savefig(F_figpath)
    save_dat(F_mat, F_matpath, label=F_label, bins2d=True)

    print('Plotting done')


def plot_eos(outdir, E_dft, E_ann, V_cell, labels):
    dft_dat_path = os.path.join(outdir,'dft.dat')
    ann_dat_path = os.path.join(outdir,'ann.dat')
    vol_dat_path = os.path.join(outdir,'vol.dat')
    plt.figure(figsize=(6,6))  
    color_list = ['r','g','b','y','c','m']

    for ind, label in enumerate(labels):
        plt.plot(V_cell[ind],E_ann[ind],label=label+'-ANN',color=color_list[ind],linewidth=1.5)
        plt.scatter(V_cell[ind],E_dft[ind],label=label+'-DFT',color=color_list[ind],marker='^')

    #plt.tick_params(labelsize=15)
    plt.xlabel('Cell Volume (%s^3)'%'Angs')
    plt.ylabel('Energy (%s/atom)'%'eV')
    plt.legend()
    fig_path = os.path.join(outdir,'eos.png')
    plt.savefig(fig_path)

    save_dat(E_dft,dft_dat_path)
    save_dat(E_ann,ann_dat_path)
    save_dat(V_cell,vol_dat_path)

    print('Plotting done')

