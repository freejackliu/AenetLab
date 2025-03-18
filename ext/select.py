import numpy as np
import sys
import warnings

def cur_select(m, nselect, niter=0):
    """Select columns of m using the CUR method."""
    nselect = min(m.shape[1], nselect)
    f = np.sum(m**2)
    p = np.sum(m**2, axis=0)/f
    selected = (-p).argsort()[0:nselect]
    mp = m[:, selected] @ np.linalg.pinv(m[:, selected]) @ m
    err = np.sqrt(np.sum((m - mp)**2)/f)
    rng = np.random.RandomState()
    for i in range(niter):
        new_selected = rng.choice(m.shape[1], nselect, p=p, replace=False)
        mp = m[:, new_selected] @ np.linalg.pinv(m[:, new_selected]) @ m
        new_err = np.sqrt(np.sum((m - mp)**2)/f)
        if (new_err < err):
            selected = new_selected
            err = new_err
    selected.sort()

    return selected, err


def load_many_nnfiles(dpath):
    import os
    from aenet.AenetLab.aenet_io import load_nnfiles

    nn_pool = []
    for n in os.listdir(dpath):
        nn_dir = os.path.join(dpath, n)
        if os.path.isdir(nn_dir):
            try:
                potentials = load_nnfiles(nn_dir)
            except:
                continue
            else:
                nn_pool.append(potentials)
    return nn_pool


def get_index(pool_range,nsamples):
    interval = None
    if isinstance(nsamples,str):
        if nsamples[0] == 'i':
            interval = int(nsamples[1:])
            if interval > pool_range[1]-pool_range[0]:
                warnings.warn("Interval %d is oversized for the pool range %s, only the first structure can be loaded."%(interval,pool_range))
            i_slice  = slice(0,None,interval)
            selected = np.arange(pool_range[0],pool_range[1])[i_slice]
        elif nsamples == 'auto':
            interval = 1
            i_slice  = slice(0,None,interval)
            selected = np.arange(pool_range[0],pool_range[1])[i_slice]
        else:
            raise Exception("'i' tag should be placed at initials")
    elif isinstance(nsamples,int):
        if nsamples > 0:
            if nsamples > pool_range[1]-pool_range[0]:
                raise Exception("Nsamples %d is out of pool range %s"%(nsamples,pool_range))
            selected = np.linspace(pool_range[0],pool_range[1]-1,nsamples,dtype=int)
            if len(selected) > 1:
                interval = selected[1]-selected[0]
        elif nsamples == -1:
            selected = [pool_range[1]-1]
    else:
        raise TypeError("Invalid value type for argument : 'nsamples', str or int expected.")
    if interval:
        print("   The interval of selection is %d"%interval)
    else:
        print("   There is no interval. Only one index is selected : %d"%selected[0])

    return list(selected)


def error_select(tags_pool,images_pool,nsamples,nn_path,etor):
    from aenet.calculator import ANNCalculator
    import pandas as pd

    tags = []
    images = []
    rmses = []
    nn_pool = load_many_nnfiles(nn_path)
    for ind, image in enumerate(images_pool):
        atoms = image.copy()
        pe = []
        try:
            pe.append(atoms.get_potential_energy())
        except:
            pass
        for nn_i in nn_pool:
             atoms.calc = ANNCalculator(potentials=nn_i)
             pe.append(atoms.get_potential_energy())
             atoms.calc.release()
        pe = np.asarray(pe)
        mean = np.mean(pe)
        rmse = np.sqrt(np.mean((pe - mean) ** 2))
        rmse = 1000*rmse/len(atoms)
        print("image: {:s} rmse: {:16.10f} meV".
              format(tags_pool[ind], rmse))
        sys.stdout.flush()
        if rmse >= etor:
            tags.append(tags_pool[ind])
            images.append(images_pool[ind])
            rmses.append(rmse)
        del atoms
    rmses = np.asarray(rmses)
    data_dict = {
            'rmses' : rmses,
            'tags'  : tags,
            'images': images
            }
    data = pd.DataFrame(data_dict)
    data.sort_values(by='rmses',ascending=False,inplace=True)
    if nsamples < len(rmses):
        offset = nsamples
    else:
        offset = len(rmses)
    ftags = data['tags'].values[:offset]
    fimages = data['images'].values[:offset]
    return ftags, fimages

def random_select(tags_pool,images_pool,nsamples,seed):
    tags = []
    images = []
    rng = np.random.RandomState(seed)
    selected = rng.choice(np.arange(len(images_pool)), nsamples)
    for i in selected:
        tags.append(tags_pool[i])
        images.append(images_pool[i])

    return tags, images


def index_select(tags_pool,images_pool,nsamples):
    tags = []
    images = []
    selected = get_index([0,len(images_pool)],nsamples)
    for i in selected:
        tags.append(tags_pool[i])
        images.append(images_pool[i])

    return tags, images


def CUR_select(tags_pool,images_pool,nsamples,SCargs):
    from aenet.libenv import SphericalChebyshev

    symbols = SCargs['symbols']
    nmax, lmax, rcut = SCargs['nmax'], SCargs['lmax'], SCargs['rcut']
    basis = SphericalChebyshev(symbols, nmax, lmax, rcut)
    nrows = len(images_pool)
    ncols = (nmax + 1)*(lmax + 1)*min(max(len(symbols), 1), 2)
    
    features = np.zeros((nrows, ncols), dtype=np.float64)
    for i in range(nrows):
        ("Evaluation index : (%i/%i)"%(i,nrows))
        fp = basis.evaluate(images_pool[i]).mean(axis=0)
        features[i, :] = fp[:]
    scaled_features = (features - features.mean(axis=0))/features.std(axis=0)
    m = scaled_features.T
    niter = max(0, SCargs['niter'])
    selected, error = cur_select(m, nsamples, niter)
    tags = []
    images = []
    for i in selected:
        tags.append(tags_pool[i])
        images.append(images_pool[i])
    print("Approximation error = {}".format(error))

    return tags, images


def select(tags_pool,images_pool,nsamples,select_style,seed=None,nn_path=None,etor=30,**SCargs):
    if select_style == "error":
        tags, images = error_select(tags_pool,images_pool,nsamples,nn_path,etor)
    elif select_style == "random":
        tags, images = random_select(tags_pool,images_pool,nsamples,seed)
    elif select_style == "index":
        tags, images = index_select(tags_pool,images_pool,nsamples)
    elif select_style == "CUR":
        tags, images = CUR_select(tags_pool,images_pool,nsamples,SCargs)
    else:
        raise Exception("Invalid 'select_style' : '%s'"%select_style)

    return tags, images
