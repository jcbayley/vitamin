import configparser
import copy
import numpy as np
import json
import importlib.resources as pkg_resources
from . import templates
import bilby
import os 
from pathlib import Path


class InputParser():

    def __init__(self, config_file = None, **args):

        self.config = {}
        self.default_prior = "BBHPriorDict"
        self.default_ini = configparser.ConfigParser()
        with pkg_resources.path(templates, "config.ini") as default_config:
            self.default_ini.read(default_config)
        self.create_config(self.default_ini)
        
        self.config_file = Path(config_file).resolve()
        if config_file is not None:
            self.ini = configparser.ConfigParser()
            self.ini.read(self.config_file)
            self.create_config(self.ini)
        else:
            print("Using default config file")
            
        self.get_priors()
        self.get_masks()
        self.get_bounds()


    def __getitem__(self, key):
        return self.config[key]

    def create_config(self, ini_file):
        for key in ini_file.keys():
            self.config.setdefault(key,{})
            for key2 in ini_file[key].keys():
                if key2 in ["shared_network", "output_network"]:
                    self.config[key][key2] = ini_file[key][key2].strip("[").strip("]").split("\n")
                elif key2 in ["waveform_approximant", "run_tag"]:
                    self.config[key][key2] = str(ini_file[key][key2]).strip('"')
                elif key2 in ["output_directory","data_directory", "prior_file"]:
                    path = Path(str(ini_file[key][key2]).strip('"'))
                    self.config[key][key2] = str(path.resolve())
                else:
                    self.config[key][key2] = json.loads(ini_file[key][key2])
        

    def get_masks(self):

        self.config["masks"] = {}

        self.config["masks"]["inf_ol_mask"], self.config["masks"]["inf_ol_idx"], self.config["masks"]["inf_ol_len"] = self.get_param_index(self.config["model"]['inf_pars'],self.config["testing"]['bilby_pars'])
        self.config["masks"]["bilby_ol_mask"], self.config["masks"]["bilby_ol_idx"], self.config["masks"]["bilby_ol_len"] = self.get_param_index(self.config["testing"]['bilby_pars'],self.config["model"]['inf_pars'])
        
        #parameter masks
        for par in self.config["model"]["inf_pars"]:
            self.config["masks"]["{}_mask".format(par)], self.config["masks"]["{}_idx_mask".format(par)], self.config["masks"]["{}_len".format(par)] = self.get_param_index(self.config["model"]['inf_pars'],[par])

        all_period_params = ['phase','psi','phi_12','phi_jl']
        periodic_params = [p for p in self.config["model"]['inf_pars'] if p in all_period_params]
        all_nonperiod_params = ['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2']
        nonperiodic_params = [p for p in self.config["model"]['inf_pars'] if p in all_nonperiod_params]

        # periodic masks
        self.config["masks"]["periodic_mask"], self.config["masks"]["periodic_idx_mask"], self.config["masks"]["periodic_len"] = self.get_param_index(self.config["model"]['inf_pars'], periodic_params)
        self.config["masks"]["nonperiodic_mask"], self.config["masks"]["nonperiodic_idx_mask"], self.config["masks"]["nonperiodic_len"] = self.get_param_index(self.config["model"]["inf_pars"],nonperiodic_params)

        self.config["masks"]["idx_periodic_mask"] = np.argsort(self.config["masks"]["nonperiodic_idx_mask"] + self.config["masks"]["periodic_idx_mask"] + self.config["masks"]["ra_idx_mask"] + self.config["masks"]["dec_idx_mask"])

        # non periodic masks
        nonperiodic_nonm1m2_params = copy.copy(nonperiodic_params)
        nonperiodic_nonm1m2_params.remove("mass_1")
        nonperiodic_nonm1m2_params.remove("mass_2")
        self.config["masks"]["nonperiodicpars_nonm1m2_mask"], self.config["masks"]["nonperiodicpars_nonm1m2_idx_mask"], self.config["masks"]["nonperiodicpars_nonm1m2_len"] = self.get_param_index(nonperiodic_params,nonperiodic_nonm1m2_params)
        self.config["masks"]["nonperiodic_nonm1m2_mask"], self.config["masks"]["nonperiodic_nonm1m2_idx_mask"], self.config["masks"]["nonperiodic_nonm1m2_len"] = self.get_param_index(self.config["model"]["inf_pars"],nonperiodic_nonm1m2_params)

        self.config["masks"]["nonperiodic_m1_mask"], self.config["masks"]["nonperiodic_m1_idx_mask"], self.config["masks"]["nonperiodic_m1_len"] = self.get_param_index(nonperiodic_params,['mass_1'])
        self.config["masks"]["nonperiodic_m2_mask"], self.config["masks"]["nonperiodic_m2_idx_mask"], self.config["masks"]["nonperiodic_m2_len"] = self.get_param_index(nonperiodic_params,['mass_2'])



    def get_bounds(self,):
        self.config["bounds"] = {}
        for par in self.config["model"]["inf_pars"]:
            self.config["bounds"]["{}_max".format(par)] = self.config["priors"][par].maximum
            self.config["bounds"]["{}_min".format(par)] = self.config["priors"][par].minimum

    def get_priors(self):
        d = bilby.core.prior.__dict__.copy()
        d.update(bilby.gw.prior.__dict__)
        if "prior_file" in self.config["data"].keys():
            priors = d[self.default_prior](self.config["data"]["prior_file"])
            # set geocent time from other input pars
            priors['geocent_time'] = bilby.core.prior.Uniform(
                minimum=self.config["data"]["ref_geocent_time"] - self.config["data"]["duration"]/2,
                maximum=self.config["data"]["ref_geocent_time"] + self.config["data"]["duration"]/2,
                name='geocent_time', latex_label='$t_c$', unit='$s$')
        else:
            priors = s[self.default_prior]
            print("No prior file, using default prior")

        self.config["priors"] = priors
            
    def get_param_index(self, all_pars,pars,sky_extra=None):
        """ 
        Get the list index of requested source parameter types
        """
        # identify the indices of wrapped and non-wrapped parameters - clunky code
        mask = []
        idx = []

        # loop over inference params
        for i,p in enumerate(all_pars):
            
            # loop over wrapped params
            flag = False
            for q in pars:
                if p==q:
                    flag = True    # if inf params is a wrapped param set flag

            # record the true/false value for this inference param
            if flag==True:
                mask.append(True)
                idx.append(i)
            elif flag==False:
                mask.append(False)

        return mask, idx, np.sum(mask)

    
