from . import out_dist
import inspect
from collections import OrderedDict

def group_outputs(config):

    inf_pars = config["inf_pars"]
    
    set_pars = set(inf_pars.values())
    
    file_dists = inspect.getmembers(out_dist, inspect.isclass)
    available_dists = OrderedDict()
    for name,cls in file_dists:
        available_dists[name] = cls

    output_dists = OrderedDict()
    for dist in set_pars:
        pars = [par for par, val in inf_pars.items() if val == dist]
        num_pars = len(pars)
        if dist in available_dists.keys():
            output_dists[dist] = available_dists[dist](pars, config)

        else:
            raise Exception("No available distribution of that Name, available names: {}".format(available_dists.keys()))

    return output_dists
    
