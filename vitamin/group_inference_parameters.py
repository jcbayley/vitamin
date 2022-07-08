from . import out_dist
import inspect
from collections import OrderedDict

def group_outputs(inf_pars, bounds):
    """
    Get the output distributions for each parmaeter and groups them by name 
    """
    inf_pars = inf_pars
    
    set_pars = []
    for name, cls in inf_pars.items():
        if cls not in set_pars:
            set_pars.append(cls)

    # get the available distributions from the out_dist file
    file_dists = inspect.getmembers(out_dist, inspect.isclass)
    available_dists = OrderedDict()
    for name,cls in file_dists:
        available_dists[name] = cls
    
    #loop over input parameters and if possible add their distribution classes
    output_dists = OrderedDict()
    new_order = []
    for dist in set_pars:
        pars = [par for par, val in inf_pars.items() if val == dist]
        num_pars = len(pars)
        splitdist = dist.split("_")
        if len(splitdist) == 1:
            distname = splitdist[0]
            distindex = None
        else:
            distname = splitdist[0]
            distindex = splitdist[1]
        if dist in available_dists.keys():
            output_dists[dist] = available_dists[dist](pars, bounds, index=distindex)
            new_order.extend(pars)
        else:
            raise Exception("No available distribution of that Name, available names: {}".format(available_dists.keys()))

    input_order = list(inf_pars.keys())
    new_order_index = []
    for i, par in enumerate(new_order):
        new_order_index.append(input_order.index(par))
        
    reverse_order_index = []
    for i, par in enumerate(input_order):
        reverse_order_index.append(new_order.index(par))
        
    return output_dists, new_order_index, reverse_order_index
    
