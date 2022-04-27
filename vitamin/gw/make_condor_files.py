import argparse
from ..vitamin_parser import InputParser
import os
import random
import stat

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
        args = "$(start_ind) $(sampler)"
        f.write('arguments = {}\n'.format(args))
        f.write('accounting_group = ligo.dev.o4.cbc.explore.test\n')
        f.write('queue\n')


def make_train_dag(config, run_type = "training"):

    p = {}
    p["root_dir"] = config["output"]["output_directory"]
    p["condor_dir"] = os.path.join(p["root_dir"],"condor")
    p["log_dir"] = os.path.join(p["condor_dir"], "log")
    p["err_dir"] = os.path.join(p["condor_dir"], "err")
    p["out_dir"] = os.path.join(p["condor_dir"], "out")
    p["config_file"] = config.config_file
    p["exec"] = os.path.join(p["condor_dir"], "run_{}.sh".format(run_type))

    p["run_type"] = run_type
    p["files_per_job"] = 20

    for direc in [p["condor_dir"], p["log_dir"], p["err_dir"], p["out_dir"]]:
        if not os.path.exists(direc):
            os.makedirs(direc)


    if run_type == "training":
        num_files = int(config["data"]["n_training_data"]/config["data"]["file_split"])
        num_jobs = int(num_files/p["files_per_job"])
        if num_jobs == 0:
            num_jobs = 1
            p["files_per_job"] = num_files
        samplers = [0]
    elif run_type == "real_noise":
        num_files = int(1)
        num_jobs = int(1)
        p["files_per_job"] = 1
        samplers = [0]
    elif run_type == "validation":
        num_files = int(config["data"]["n_validation_data"]/config["data"]["file_split"])
        num_jobs = 1
        p["files_per_job"] = num_files
        samplers = [0]
    elif run_type == "test":
        num_jobs = config["data"]["n_test_data"]
        p["files_per_job"] = 1
        samplers = config["testing"]["samplers"]
        samplers.remove("vitamin")
    elif run_type == "real_test":
        num_jobs = 1

    comment = "{}_run".format(run_type)
    run_sub_filename = os.path.join(p["condor_dir"], "{}.sub".format(comment))
    write_subfile(run_sub_filename,p,comment)
    make_executable(p)
    dag_filename = "{}/{}.dag".format(p["condor_dir"],comment)

    with open(dag_filename,'w') as f:
        for i in range(num_jobs):
            for j in range(len(samplers)):
                comment = "File_{}".format(i)
                uid = random.randint(1,1e9)
                jobid = "{}_{}_{}".format(comment,i,uid)
                job_string = "JOB {} {}\n".format(jobid,run_sub_filename)
                retry_string = "RETRY {} 1\n".format(jobid)
                if run_type == "train":
                    vars_string = 'VARS {} start_ind="{}"\n'.format(jobid,int(i*p["files_per_job"]))
                elif run_type == "test":
                    vars_string = 'VARS {} start_ind="{}" sampler="{}"\n'.format(jobid,int(i), samplers[j])
                else:
                    vars_string = 'VARS {} start_ind="{}"\n'.format(jobid,int(i))
                f.write(job_string)
                f.write(retry_string)
                f.write(vars_string)

def make_executable(p):
    with open(p["exec"], "w")as f:
        f.write("#!/usr/bin/bash\n")
        args = "python -m vitamin.gw.generate_data --start-ind ${{1}} --num-files {} --run-type {} --ini-file {}".format( p["files_per_job"], p["run_type"],p["config_file"])
        if p["run_type"] == "test":
            args += " --sampler ${2}"
        f.write(args + "\n")

    # make the bash script executable
    st = os.stat(p["exec"])
    os.chmod(p["exec"], st.st_mode | stat.S_IEXEC)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input files and options')
    parser.add_argument('--ini-file', metavar='i', type=str, help='path to ini file')

    args = parser.parse_args()
    vitamin_config = InputParser(args.ini_file)
    
    make_train_dag(vitamin_config, run_type = "training")
    make_train_dag(vitamin_config, run_type = "validation")
    make_train_dag(vitamin_config, run_type = "test")

    if vitamin_config["data"]["use_real_detector_noise"]:
        make_train_dag(vitamin_config, run_type = "real_noise")

