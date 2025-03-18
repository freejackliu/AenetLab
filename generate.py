# Fortran is case-insensitive, thus the keyword can be in both types. 
# For a tidy layout, we all use capital ones.

template_stp = """DESCR
{}
END DESCR

ATOM {}

ENV {}
{}

RMIN {}d0

{}
"""

template_genin = """OUTPUT {}

TYPES
{}
{}

SETUPS
{}
{}
FILES
{}
{}
"""


def write_SC_stp(outd,envs,e,rmin,nmax,lmax,rcut):
    description = f"Setup of {e}, written by AenetLab"
    nenv = len(envs)
    env = '\n'.join(envs)
    fp = f"BASIS type=Spherical_Chebyshev\n" + \
         f"nmax={nmax} lmax={lmax} Rc={rcut}d0"
    op = template_stp.format(description, e, nenv, env, rmin, fp)
    with open(f"{outd}/{e}.stp", "w") as o:
        o.write(op)


def write_gen_infile(outd,infile,output,envs,es,eu,stp,tad,nf,fp):
    nenv  = len(envs)
    esau  = []
    enstp = []
    if eu == 'eV':
        for ind,env in enumerate(envs):
            esau.append('%-3s'%env+' %7.4f'%es[ind]+' | %s'%eu)
            enstp.append('%-3s'%env+' %s'%stp[ind])
    #TODO:more E units
    else:
        raise RuntimeError("AeLab only recognizes E_unit of 'eV'")
    tad_final = '\n'
    for ind,key in enumerate(tad.keys()):
        if tad[key]:
            keys = list(tad.keys())
            tad_final+='%s\n'%keys[ind]
    es_final    = '\n'.join(esau)
    enstp_final = '\n'.join(enstp)
    fp_final    = '\n'.join(fp)
    op = template_genin.format(output, nenv, es_final, enstp_final, tad_final, nf, fp_final)
    with open(f"{outd}/{infile}", "w") as o:
        o.write(op)

