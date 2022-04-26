from . import templates
import importlib.resources as pkg_resources
import os
import shutil
from pathlib import Path
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input files and options')
    parser.add_argument('--out-dir', metavar='o', type=str, help='path to run directory')    

    args = parser.parse_args()

    outdir = Path(args.out_dir).resolve()
    
    
    with pkg_resources.path(templates, "config.ini") as cfg:
        config_file = cfg

    with pkg_resources.path(templates, "bbh_prior.prior")  as pr:
        prior_file = pr

        

    if not os.path.isdir(outdir):
        print("Making directory: {}".format(outdir))
        os.makedirs(outdir)
        shutil.copy(config_file, outdir)
        shutil.copy(prior_file, outdir)
    else:
        raise Exception("Directory already exists: {}".format(args.out_dir))
    
    
    
        
