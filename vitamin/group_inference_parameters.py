from . import out_dist
import inspect
from collections import OrderedDict

def group_outputs(config):

    inf_pars = config["inf_pars"]
    
    set_pars = set(inf_pars.values())
    
    available_dists = inspect.getmembers(out_dist, inspect.isclass)
    available_dists = {name:cls for name,cls in available_dists}

    output_dists = OrderedDict()
    for dist in set_pars:
        pars = [par for par, val in inf_pars.items() if val == dist]
        num_pars = len(pars)
        if dist in available_dists.keys():
            output_dists[dist] = available_dists[dist](pars)
        else:
            raise Exception("No available distribution of that Name, available names: {}".format(available_dists.keys()))

    return output_dists
    
