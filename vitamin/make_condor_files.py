import argparse
from .vitamin_parser import InputParser
import os
import random

def create_dirs(dirs):
    for i in dirs:
        if not os.path.isdir(i):
            try: 
                os.makedirs(i)
            except:
                print >> sys.stderr, "Could not create directory {}".format(i)
                sys.exit(1)
    print("All directories exist")

def write_subfile(sub_filename,p,comment):
    print(sub_filename)
    with open(sub_filename,'w') as f:
        f.write('# filename: {}\n'.format(sub_filename))
        f.write('universe = vanilla\n')
        f.write('executable = {}\n'.format(p["exec"]))
        #f.write('enviroment = ""\n')
        f.write('getenv  = True\n')
        #f.write('RequestMemory = {} \n'.format(p["memory"]))
        f.write('log = {}/{}_$(cluster).log\n'.format(p["log_dir"],comment))
        f.write('error = {}/{}_$(cluster).err\n'.format(p["err_dir"],comment))
        f.write('output = {}/{}_$(cluster).out\n'.format(p["out_dir"],comment))
        args = ""
        if p["train"]:
            args += "--gen_train True "
        if p["val"]:
            args += "--gen_val True "
        if p["test"]:
            args += "--gen_test True "
        if p["real_test"]:
            args += "--gen_real_test True "
        if p["real_noise"]:
            args += "--gen_rnoise True "
        args += "--start_ind $(start_ind) --num_files {} --config-file {}".format( p["files_per_job"], p["config_file"])
        f.write('arguments = {}\n'.format(args))
        f.write('accounting_group = ligo.dev.o4.cbc.explore.test\n')
        f.write('queue\n')


def make_train_dag(config, run_type = "train"):


    p = {}
    p["root_dir"] = config["output"]["output_directory"]
    p["condor_dir"] = "{}/condor".format(p["root_dir"])
    p["log_dir"] = os.path.join(p["condor_dir"], "log")
    p["err_dir"] = os.path.join(p["condor_dir"], "err")
    p["out_dir"] = os.path.join(p["condor_dir"], "out")
    p["config_file"] = vitamin_config.config_file
    p["exec"] = os.path.join(p["root_dir"], "run_vitamin_condor.py_change")

    p["train"] = False
    p["val"] = False
    p["test"] = False
    p["real_test"] = False
    p["real_noise"] = False
    p[run_type] = True
    p["files_per_job"] = 20

    for direc in [p["condor_dir"], p["log_dir"], p["err_dir"], p["out_dir"]]:
        if not os.path.exists(direc):
            os.makedirs(direc)

    comment = "{}_run".format(run_type)
    run_sub_filename = os.path.join(p["condor_dir"], "{}.sub".format(comment))
    write_subfile(run_sub_filename,p,comment)


    dag_filename = "{}/{}.dag".format(p["condor_dir"],comment)
    if run_type == "train":
        num_files = int(config["data"]["n_training_data"]/config["data"]["file_split"])
        num_jobs = int(num_files/p["files_per_job"])
        if num_jobs == 0:
            num_jobs = 1
            p["files_per_job"] = num_files
    elif run_type == "real_noise":
        num_files = int(1)
        num_jobs = int(1)
    elif run_type == "val":
        num_files = int(config["data"]["n_validation_data"]/config["data"]["file_split"])
        num_jobs = 1
    elif run_type == "test":
        num_jobs = config["data"]["n_test_data"]
    elif run_type == "real_test":
        num_jobs = 1

    with open(dag_filename,'w') as f:
        seeds = []
        for i in range(num_jobs):
            seeds.append(random.randint(1,1e9))
        for i in range(num_jobs):
            comment = "File_{}".format(i)
            uid = seeds[i]
            jobid = "{}_{}_{}".format(comment,i,uid)
            job_string = "JOB {} {}\n".format(jobid,run_sub_filename)
            retry_string = "RETRY {} 1\n".format(jobid)
            if run_type == "train":
                vars_string = 'VARS {} start_ind="{}"\n'.format(jobid,int(i*p["files_per_job"]))
            else:
                vars_string = 'VARS {} start_ind="{}"\n'.format(jobid,int(i))
            f.write(job_string)
            f.write(retry_string)
            f.write(vars_string)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input files and options')
    parser.add_argument('--ini-file', metavar='i', type=str, help='path to ini file')

    args = parser.parse_args()
    vitamin_config = InputParser(args.ini_file)
    
    make_train_dag(vitamin_config, run_type = "train")
    make_train_dag(vitamin_config, run_type = "val")
    make_train_dag(vitamin_config, run_type = "test")

    if vitamin_config["data"]["use_real_noise"]:
        make_train_dag(vitamin_config, run_type = "real_noise")

