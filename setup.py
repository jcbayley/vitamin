#!/usr/bin/env python

from setuptools import setup, find_packages
import subprocess
import sys
import os

# check that python version is 3.5 or above
python_version = sys.version_info
if python_version < (3, 6):
    sys.exit("Python < 3.6 is not supported, aborting setup")
print("Confirmed Python version {}.{}.{} >= 3.6.0".format(*python_version[:3]))


def write_version_file(version):
    """ Writes a file with version information to be used at run time

    Parameters
    ----------
    version: str
        A string containing the current version information

    Returns
    -------
    version_file: str
        A path to the version file

    """
    try:
        git_log = subprocess.check_output(
            ['git', 'log', '-1', '--pretty=%h %ai']).decode('utf-8')
        git_diff = (subprocess.check_output(['git', 'diff', '.']) +
                    subprocess.check_output(
                        ['git', 'diff', '--cached', '.'])).decode('utf-8')
        if git_diff == '':
            git_status = '(CLEAN) ' + git_log
        else:
            git_status = '(UNCLEAN) ' + git_log
    except Exception as e:
        print("Unable to obtain git version information, exception: {}"
              .format(e))
        git_status = 'release'

    version_file = '.version'
    if os.path.isfile(version_file) is False:
        with open('vitamin/' + version_file, 'w+') as f:
            f.write('{}: {}'.format(version, git_status))

    return version_file


def get_long_description():
    """ Finds the README and reads in the description """
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, 'README.md')) as f:
        long_description = f.read()
    return long_description


# get version info from __init__.py
def readfile(filename):
    with open(filename) as fp:
        filecontents = fp.read()
    return filecontents


VERSION = '0.3.0'
version_file = write_version_file(VERSION)
long_description = get_long_description()


setup(
    name='vitamin',
    version='0.3.0',    
    description='A user-friendly machine learning Bayesian inference library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://git.ligo.org/joseph.bayley/vitamin_c',
    author='Joseph Bayley, Hunter Gabbard, Chris Messenger, Ik Siong Heng, Francesco Tonolini, Roderick Murray-Smith',
    author_email='joseph.bayley@glasgow.ac.uk',
    license='GNU General Public License v3 (GPLv3)',
    packages=find_packages(),
    package_dir={'vitamin': 'vitamin'},
    include_package_data=True,
    package_data={'vitamin': ['default_files/bbh.prior',"default_files/config.ini", "default_files/init_config.ini"],
                  'vitamin': [version_file]},

    python_requires='>=3.6', 
    install_requires=['numpy',
                      'universal-divergence',
                      'absl-py',
                      'asn1crypto',
                      'astor',
                      'astroplan',
                      'astropy',
                      'astropy-healpix',
                      'astroquery',
                      'beautifulsoup4',
                      'bilby>=1.0.0',
                      'cachetools',
                      'certifi',
                      'cffi',
                      'chardet',
                      'cloudpickle',
                      'corner',
                      'cpnest',
                      'cryptography',
                      'cycler',
                      'Cython',
                      'decorator',
                      'deepdish',
                      'dill',
                      'dqsegdb2',
                      'dynesty',
                      'emcee',
                      'entrypoints',
                      'future',
                      'gast',
                      'grpcio',
                      'gwdatafind',
                      'gwosc',
                      'gwpy',
                      'h5py',
                      'healpy',
                      'html5lib',
                      'idna',
                      'jeepney',
                      'joblib',
                      'keyring',
                      'kiwisolver',
                      'lalsuite',
                      'ligo-gracedb',
                      'ligo-segments',
                      'ligo.skymap',
                      'ligotimegps',
                      'lscsoft-glue',
                      'Markdown',
                      'matplotlib',
                      'mock',
                      'networkx',
                      'numexpr',
                      'oauthlib',
                      'opt-einsum',
                      'pandas',
                      'patsy',
                      'Pillow',
                      'protobuf',
                      'ptemcee',
                      'pyaml',
                      'pyasn1',
                      'pyasn1-modules',
                      'pycparser',
                      'pymc3',
                      'pyOpenSSL',
                      'pyparsing',
                      'python-dateutil',
                      'python-ligo-lw',
                      'pytz',
                      'PyYAML',
                      'reproject',
                      'requests',
                      'requests-oauthlib',
                      'rsa',
                      'scikit-learn',
                      'scikit-optimize',
                      'scipy',
                      'seaborn',
                      'SecretStorage',
                      'six',
                      'soupsieve',
                      'tables',
                      'tensorflow==2.6.0',
                      'keras==2.6.0',
                      'tensorflow-addons==0.13.0',
                      'tensorflow-probability==0.14.1',
                      'termcolor',
                      'tqdm',
                      'urllib3',
                      'webencodings',
                      'Werkzeug',
                      'wrapt',
                      'natsort',
                      'regex',
                      'importlib_resources'],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: >=3.6',
    ],
)

