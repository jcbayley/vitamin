import configparser
import copy
import numpy as np
import json
import importlib.resources as pkg_resources
import importlib_resources
from collections import OrderedDict
#from . import templates
import bilby
import os 
from pathlib import Path
from ..group_inference_parameters import group_outputs
import regex
from ..vitamin_parser import InputParser

class GWInputParser(InputParser):

    def __init__(self, config_file = None, **args):
        """
        Create a configuration, either default configuration or input config file
        """
        self.config = {}
        self.default_prior = "BBHPriorDict"
        self.default_ini = configparser.ConfigParser()
        my_resources = importlib_resources.files("vitamin")
        self.default_ini.read((my_resources / "default_files"/ "config.ini"))
        #with pkg_resources.path(os.path.join(__package__,"default_files"), "config.ini") as default_config:
        #    print(default_config)
        #    self.default_ini.read(default_config)
        self.create_config(self.default_ini)
        

        if config_file is not None:
            self.config_file = Path(config_file).resolve()
            if not self.config_file.is_file():
                raise FileNotFoundError(f"No file {self.config_file}")
            self.ini = configparser.ConfigParser()
            print(f"Loading: {self.config_file}")
            self.ini.read(self.config_file)
            self.create_config(self.ini)
        else:
            self.config_file = None
            print("Using default config file")
            
        self.get_priors()
        self.get_masks()
        self.get_bounds()
        self.get_param_order()        

    def get_masks(self):

        self.config["masks"] = {}

        self.config["masks"]["inf_bilby_idx"] = []
        for i, p in enumerate(self.config["model"]["inf_pars_list"]):
            for j, p1 in enumerate(self.config["testing"]["bilby_pars"]):
                if p == p1:
                    self.config["masks"]["inf_bilby_idx"].append((i,j))


        self.config["masks"]["inf_bilby_len"] = len(self.config["masks"]["inf_bilby_idx"])
        #self.config["masks"]["inf_ol_mask"], self.config["masks"]["inf_ol_idx"], self.config["masks"]["inf_ol_len"] = self.get_param_index(self.config["model"]['inf_pars_list'],self.config["testing"]['bilby_pars'])
        #self.config["masks"]["bilby_ol_mask"], self.config["masks"]["bilby_ol_idx"], self.config["masks"]["bilby_ol_len"] = self.get_param_index(self.config["testing"]['bilby_pars'],self.config["model"]['inf_pars_list'])
        
        #parameter masks
        for par in self.config["model"]["inf_pars_list"]:
            self.config["masks"]["{}_mask".format(par)], self.config["masks"]["{}_idx_mask".format(par)], self.config["masks"]["{}_len".format(par)] = self.get_param_index(self.config["model"]['inf_pars_list'],[par])

        all_period_params = ['phase','psi','phi_12','phi_jl']
        periodic_params = [p for p in self.config["model"]['inf_pars_list'] if p in all_period_params]
        all_nonperiod_params = ['mass_1','mass_2','luminosity_distance','geocent_time','theta_jn','a_1','a_2','tilt_1','tilt_2']
        nonperiodic_params = [p for p in self.config["model"]['inf_pars_list'] if p in all_nonperiod_params]

        # periodic masks
        self.config["masks"]["periodic_mask"], self.config["masks"]["periodic_idx_mask"], self.config["masks"]["periodic_len"] = self.get_param_index(self.config["model"]['inf_pars_list'], periodic_params)
        self.config["masks"]["nonperiodic_mask"], self.config["masks"]["nonperiodic_idx_mask"], self.config["masks"]["nonperiodic_len"] = self.get_param_index(self.config["model"]["inf_pars_list"],nonperiodic_params)

        self.config["masks"]["idx_periodic_mask"] = np.argsort(self.config["masks"]["nonperiodic_idx_mask"] + self.config["masks"]["periodic_idx_mask"] + self.config["masks"]["ra_idx_mask"] + self.config["masks"]["dec_idx_mask"])



    def get_bounds(self,):
        self.config["bounds"] = {}
        for par in self.config["priors"]:
            self.config["bounds"]["{}_max".format(par)] = self.config["priors"][par].maximum
            self.config["bounds"]["{}_min".format(par)] = self.config["priors"][par].minimum
        if "chirp_mass" not in self.config["priors"].keys():
            self.config["bounds"]["chirp_mass_max"] = bilby.gw.conversion.component_masses_to_chirp_mass(self.config["bounds"]["mass_1_max"], self.config["bounds"]["mass_2_max"])
            self.config["bounds"]["chirp_mass_min"] = bilby.gw.conversion.component_masses_to_chirp_mass(self.config["bounds"]["mass_1_min"], self.config["bounds"]["mass_2_min"])
        if "mass_ratio" not in self.config["priors"].keys():
            self.config["bounds"]["mass_ratio_max"] = bilby.gw.conversion.component_masses_to_mass_ratio(self.config["bounds"]["mass_1_max"], self.config["bounds"]["mass_2_max"])
            self.config["bounds"]["mass_ratio_min"] = bilby.gw.conversion.component_masses_to_mass_ratio(self.config["bounds"]["mass_1_max"], self.config["bounds"]["mass_2_min"])



    def get_priors(self):
        d = bilby.core.prior.__dict__.copy()
        d.update(bilby.gw.prior.__dict__)
        if "prior_file" in self.config["data"].keys():
            priors = d[self.default_prior](self.config["data"]["prior_file"])
            # set geocent time from other input pars
        else:
            priors = d[self.default_prior]()
            print("No prior file, using default prior")

        priors['geocent_time'].minimum += self.config["data"]["ref_geocent_time"] - self.config["data"]["duration"]/2
        priors['geocent_time'].maximum += self.config["data"]["ref_geocent_time"] - self.config["data"]["duration"]/2
        #priors['geocent_time'] = bilby.core.prior.Uniform(
        #    minimum=self.config["data"]["ref_geocent_time"] + self.config["data"]["duration"]/2 - 0.35,
        #    maximum=self.config["data"]["ref_geocent_time"] + self.config["data"]["duration"]/2 - 0.15,
        #    name='geocent_time', latex_label='$t_c$', unit='$s$')

        self.config["priors"] = priors
        self.config["testing"]["bilby_pars"] = list(self.config["priors"].keys())
        if self.config["testing"]["phase_marginalisation"]:
            self.config["testing"]["bilby_pars"].remove("phase")
            
