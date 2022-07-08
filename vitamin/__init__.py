"""
VItamin
=====

VItamin: a user-friendly machine learning posterior generation library.

The aim of VItamin is to provide a user-friendly interface to perform fast parameter
estimation. It is primarily designed and built for inference of compact
binary coalescence events in interferometric data, but it can also be used for
more general problems.

The code is hosted at .
For installation instructions see
"need to add link here".

"""
from __future__ import absolute_import
import sys

from . import vitamin_model
#from . import gw
from . import callbacks
from . import plotting
#from . import templates

# Check for optional basemap installation
try:
    from mpl_toolkits.basemap import Basemap
    print("module 'basemap' is installed")
except (ModuleNotFoundError, ImportError):
    print("module 'basemap' is not installed")
    print("Skyplotting functionality is automatically disabled.")
else:
    from .skyplotting import plot_sky

from .vitamin_model import CVAE

#from . import run_vitamin

__version__ = "0.3.0"
__author__ = 'Joseph Bayley, Hunter Gabbard, Chris Messenger'
__credits__ = 'University of Glasgow'

