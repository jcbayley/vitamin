=================================
Customizing Your Parameters Files for GW search
=================================

VItamin allows the user to tailor their run to a reasonable degree. 
Some work still needs to be done to improve the customizability. 

First, use vitamin to produce the default parameters files

.. code-block:: console

   $ python -m vitamin.initialise_directory --out-dir your_run_directory

You will now see a config.ini and a bbh.prior file.

To change the default values for each of these values, use your 
text editor of choice (e.g. vim).

An example config file is:

.. code-block:: console

   [run_info]
   run_tag = "test_run1"
   
   [output]
   output_directory = "path to output directory"
   
   [training]
   resume_training = false
   
   num_iterations = 5000
   chunk_batch = 40
   num_epoch_load = 4
   batch_size = 512
   
   plot_interval = 100
   test_interval = false
   tensorboard_log = true
   
   # learning rate modifications
   initial_learning_rate = 1e-4
   
   cycle_lr = false
   cycle_lr_start = 1000
   cycle_lr_length = 100
   
   decay_lr = false
   decay_lr_start = 1000
   deca_lr_length = 3000
   
   ramp_start = 400
   ramp_length = 600
   ramp_n_cycles = 1
   
   [data]
   data_directory = "Path to data directory"
   prior_file = "path to prior file"
   
   sampling_frequency = 1024
   duration = 1
   ref_geocent_time = 1325029268
   
   waveform_approximant = "IMRPhenomPv2"
   reference_frequency = 20
   minimum_frequency = 10
   
   n_training_data = 1e6
   n_validation_data = 1e3
   n_test_data = 128
   file_split = 1e3
   
   use_real_detector_noise = false
   save_polarisations = true
   
   psd_files = []
   
   y_normscale = 128
   detectors = ["H1","L1","V1"]
   
   [testing]
   make_paper_plots = false
   samplers = ["vitamin","nessai"]
   nsamples = 3000
   phase_marginalisation = true
   
   [model]
   
   shared_network = [Conv1D(96,64,2),
		Conv1D(96,64,2), 
		Conv1D(96,64,2),
		Conv1D(96,64,2),
		Conv1D(96,64,2)
		]
   
   r1_network = [Linear(4096),
		Linear(2048),
		Linear(1024)]

   q_network = [Linear(4096),
		Linear(2048),
		Linear(1024)]

   r2_network = [Linear(4096),
		Linear(2048),
		Linear(1024)]

   [inf_pars]
   chirp_mass = "JointChirpmassMR"
   mass_ratio = "JointChirpmassMR"	
   phase =  "VonMises"
   luminosity_distance =  "TruncatedNormal"
   geocent_time = "TruncatedNormal"
   theta_jn = "VonMises"
   psi = "VonMises"
   phi_12 = "VonMises"
   phi_jl = "VonMises"
   a_1 = "TruncatedNormal"
   a_2 = "TruncatedNormal"
   tilt_1 = "VonMises"
   tilt_2 = "VonMises"
   ra = "JointVonMisesFisher"
   dec = "JointVonMisesFisher"

  
