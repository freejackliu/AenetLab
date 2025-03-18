import numpy as np

def print_dirpath(dir_dic,f):
    for dir_tag, dir_path in dir_dic.items():
        f.write('%s : %s\n'%(dir_tag, dir_path))
    f.write('\n')

def print_unit_info(E_unit,F_unit,angle_unit,f):
    f.write('Units info:\n')
    f.write('E      %s\n'%E_unit)
    f.write('F      %s\n'%F_unit)
    f.write('angle  %s\n'%angle_unit)
    f.write('\n')

def print_error_header(f,verb=True):
    if verb:
        f.write('%21s'%'delta_abs'+'%18s\n'%'delta_angle')

def print_energy_error(E_dft,E_ann,f,verb=True):
    ae = abs(E_dft-E_ann)
    if verb:
        f.write('%-7s'%'Etot'+'     %E\n'%ae)
    return ae

def print_forces_error(types,Fs_dft,Fs_ann,angle_unit,f,verb=True,visual=False):
    afe, fe_list = 0, []
    age, ge_list = 0, []
    for i in range(len(Fs_dft)):
        x = np.linalg.norm(Fs_dft[i])
        y = np.linalg.norm(Fs_ann[i])
        afe += abs(x-y)
        if angle_unit == 'degree':
            ge = np.arccos(np.dot(Fs_dft[i],Fs_ann[i])/x/y)*180/np.pi
            if verb:
                f.write('%-7s'%('F_%d'%i)+'%3s'%types[i]+'  %E'%abs(x-y)+
                '    %.4f\n'%ge)
            age += abs(ge)
        elif angle_unit == 'rad':
            ge = np.arccos(np.dot(Fs_dft[i],Fs_ann[i])/x/y)
            if verb:
                f.write('%-7s'%('F_%d'%i)+'%3s'%types[i]+'  %E'%abs(x-y)+
                '    %.4f\n'%ge)
            age += abs(ge)
        else:
            raise Exception("Invalid angle unit : %s"%angle_unit)
        if visual:
            fe_list.append(abs(x-y))
            ge_list.append(ge)
    if visual:
        return afe,age,fe_list,ge_list
    else:
        return afe,age

def print_MAE(style,delta_sum,Natom,f):
    if style in ['energy','force']:
        mae = delta_sum/Natom
        f.write('%6s MAE : %E per atom\n'%(style,mae))
    elif style == 'angle':
        mae = delta_sum/Natom
        f.write('%6s MAE : %f per atom\n'%(style,mae))
    else:
        raise Exception("Unexpected style '%s'"%style)
