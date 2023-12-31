import importlib.resources as pkg_resources
import importlib_resources
import os
import shutil
from pathlib import Path
import argparse

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input files and options')
    parser.add_argument('--out-dir', metavar='o', type=str, help='path to run directory')    

    args = parser.parse_args()

    outdir = Path(args.out_dir).resolve()
    my_resources = importlib_resources.files("vitamin")
    config_file = (my_resources / "default_files"/ "init_config.ini")
    prior_file = (my_resources / "default_files"/ "bbh_prior.prior")
    

    if not os.path.isdir(outdir):
        print("Making directory: {}".format(outdir))
        os.makedirs(outdir)
        shutil.copy(config_file, os.path.join(outdir, "config.ini"))
        shutil.copy(prior_file, outdir)
    else:
        raise Exception("Directory already exists: {}".format(args.out_dir))
    
    
    
        
