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
from .group_inference_parameters import group_outputs
import regex

class InputParser():

    def __init__(self, config_file = None, **args):
        """
        Create a configuration, either default configuration or input config file
        """
        self.config = {}
        self.default_prior = "Uniform"
        self.default_ini = configparser.ConfigParser()
        my_resources = importlib_resources.files("vitamin")
        self.default_ini.read((my_resources / "default_files"/ "config.ini"))
        #with pkg_resources.path(os.path.join(__package__,"default_files"), "config.ini") as default_config:
        #    print(default_config)
        #    self.default_ini.read(default_config)
        self.create_config(self.default_ini)
        

        if config_file is not None:
            self.config_file = Path(config_file).resolve()
            self.ini = configparser.ConfigParser()
            self.ini.read(self.config_file)
            self.create_config(self.ini)
        else:
            self.config_file = None
            print("Using default config file")
            
        self.get_priors()
        self.get_masks()
        self.get_bounds()
        

    def __getitem__(self, key):
        return self.config[key]

    def create_config(self, ini_file):
        for key in ini_file.keys():
            if key == "inf_pars":
                self.config[key] = OrderedDict()
            else:
                self.config.setdefault(key,{})
            for key2 in ini_file[key].keys():
                if key2 in ["shared_conv","shared_network", "output_network", "r1_conv","r2_conv","q_conv", "r1_network","r2_network","q_network"]:
                    self.config[key][key2] = [mod.strip("\n") for mod in regex.split(r"\s*,\s*(?![^(]*\))", ini_file[key][key2].strip("[").strip("]"))]
                elif key2 in ["waveform_approximant", "run_tag"]:
                    self.config[key][key2] = str(ini_file[key][key2]).strip('"')
                elif key2 in ["output_directory","data_directory", "prior_file"]:
                    path = Path(str(ini_file[key][key2]).strip('"'))
                    self.config[key][key2] = str(path.resolve())
                else:
                    self.config[key][key2] = json.loads(ini_file[key][key2])
        
        self.config["model"]["inf_pars_list"] = list(self.config["inf_pars"].keys())

    def get_masks(self):

        self.config["masks"] = {}
        #parameter masks
        for par in self.config["model"]["inf_pars_list"]:
            self.config["masks"]["{}_mask".format(par)], self.config["masks"]["{}_idx_mask".format(par)], self.config["masks"]["{}_len".format(par)] = self.get_param_index(self.config["model"]['inf_pars_list'],[par])

        #self.get_param_order()


    def get_bounds(self,):
        self.config["bounds"] = {}
        for par in self.config["priors"]:
            self.config["bounds"]["{}_max".format(par)] = self.config["priors"][par].maximum
            self.config["bounds"]["{}_min".format(par)] = self.config["priors"][par].minimum


    def get_priors(self):
        """
        Get the prior distributions from the 
        """
        d = bilby.core.prior.__dict__.copy()
        d.update(bilby.gw.prior.__dict__)
        if "prior_file" in self.config["data"].keys():
            
            priors = d[self.default_prior](self.config["data"]["prior_file"])
            # set geocent time from other input pars
        else:
            
            priors = d[self.default_prior]()
            print("No prior file, using default prior")

        self.config["priors"] = priors
        self.config["data"]["prior_pars"] = list(self.config["priors"].sample().keys())
            
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

    
    def get_param_order(self):
        grouped_params, newidx, oldidx = group_outputs(self.config["inf_pars"], self.config["bounds"])
        oldorder = list(self.config["inf_pars"].keys())
        neworder = []
        for name, group in grouped_params.items():
            neworder.extend(group.pars)

        new_order_idx = []
        for i, par in enumerate(neworder):
            new_order_idx.append(oldorder.index(par))

        reverse_order_idx = []
        for i, par in enumerate(oldorder):
            reverse_order_idx.append(neworder.index(par))
        
        self.config["masks"]["group_order_idx"] = new_order_idx
        self.config["masks"]["ungroup_order_idx"] = reverse_order_idx
