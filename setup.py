from setuptools import setup

setup(
    name='vitamin_b',
    version='1.0.0',    
    description='A example Python package',
    url='https://github.com/hagabbar/vitamin_b',
    author='Hunter Gabbard',
    author_email='h.gabbard.1@research.gla.ac.uk',
    license='GNU General Public License v3.0',
    packages=['vitamin_b'],
    install_requires=['universal-divergence==0.2.0',
                      'absl-py==0.9.0',
                      'asn1crypto==0.24.0',
                      'astor==0.8.1',
                      'astroplan==0.5',
                      'astropy==3.2.1',
                      'astropy-healpix==0.4',
                      'astroquery==0.3.10',
                      'beautifulsoup4==4.8.0',
                      'bilby==0.5.5',
                      'cachetools==4.0.0',
                      'certifi==2019.9.11',
                      'cffi==1.12.3',
                      'chardet==3.0.4',
                      'cloudpickle==1.2.2',
                      'corner==2.0.1',
                      'cpnest==0.9.9',
                      'cryptography==2.7',
                      'cycler==0.10.0',
                      'Cython==0.29.15',
                      'decorator==4.4.0',
                      'deepdish==0.3.6',
                      'dill==0.3.1.1',
                      'dqsegdb2==1.0.1',
                      'dynesty==0.9.7',
                      'emcee==2.2.1',
                      'entrypoints==0.3',
                      'future==0.18.2',
                      'gast==0.2.2',
                      'google-auth==1.11.3',
                      'google-auth-oauthlib==0.4.1',
                      'google-pasta==0.2.0',
                      'grpcio==1.27.2',
                      'gwdatafind==1.0.4',
                      'gwosc==0.4.3',
                      'gwpy==0.15.0',
                      'h5py==2.9.0',
                      'healpy==1.12.10',
                      'html5lib==1.0.1',
                      'idna==2.8',
                      'jeepney==0.4.1',
                      'joblib==0.14.1',
                      'Keras==2.3.1',
                      'Keras-Applications==1.0.8',
                      'Keras-Preprocessing==1.1.0',
                      'keyring==19.2.0',
                      'kiwisolver==1.1.0',
                      'lalsuite==6.62',
                      'ligo-gracedb==2.4.0',
                      'ligo-segments==1.2.0',
                      'ligo.skymap==0.1.12',
                      'ligotimegps==2.0.1',
                      'lscsoft-glue==2.0.0',
                      'Markdown==3.2.1',
                      'matplotlib==3.1.3',
                      'mock==3.0.5',
                      'networkx==2.3',
                      'numexpr==2.7.0',
                      'numpy==1.18.4',
                      'oauthlib==3.1.0',
                      'opt-einsum==3.2.0',
                      'pandas==1.0.1',
                      'patsy==0.5.1',
                      'Pillow==6.1.0',
                      'protobuf==3.11.3',
                      'ptemcee==1.0.0',
                      'pyaml==20.4.0',
                      'pyasn1==0.4.8',
                      'pyasn1-modules==0.2.8',
                      'pycparser==2.19',
                      'pymc3==3.7',
                      'pyOpenSSL==19.0.0',
                      'pyparsing==2.4.6',
                      'python-dateutil==2.8.1',
                      'python-ligo-lw==1.5.3',
                      'pytz==2019.3',
                      'PyYAML==5.1.2',
                      'reproject==0.5.1',
                      'requests==2.22.0',
                      'requests-oauthlib==1.3.0',
                      'rsa==4.0',
                      'scikit-learn==0.22.2.post1',
                      'scikit-optimize==0.7.4',
                      'scipy==1.4.1',
                      'seaborn==0.9.0',
                      'SecretStorage==3.1.1',
                      'six==1.14.0',
                      'soupsieve==1.9.3',
                      'tables==3.5.2',
                      'tensorboard==2.1.1',
                      'tensorflow==2.1.0',
                      'tensorflow-estimator==2.1.0',
                      'tensorflow-probability==0.9.0',
                      'termcolor==1.1.0',
                      'Theano==1.0.4',
                      'tqdm==4.42.1',
                      'urllib3==1.25.6',
                      'webencodings==0.5.1',
                      'Werkzeug==1.0.0',
                      'wrapt==1.12.1',
                      ],

    classifiers=[
        'Development Status :: 1 - Planning',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU License',  
        'Operating System :: POSIX :: Linux',        
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
    ],
)
