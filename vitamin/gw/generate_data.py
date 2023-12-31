from .create_template import GenerateTemplate
import argparse
from .gw_parser import GWInputParser
import os
import h5py
import numpy as np
import bilby

class DataGenerator():

    def __init__(self, config, run_type, write_to_file = True):
        """
        Generate training/validation/test data
        """
        self.config = config
        self.run_type = run_type
        self.write_to_file = True
        
    def create_training_val_file(self, start_ind=0):

        # Make training set directory
        if self.write_to_file:
            save_dir = os.path.join(self.config["data"]["data_directory"], self.run_type)
            os.system('mkdir -p %s' % save_dir)

        # Make directory for plots
        #os.system('mkdir -p %s/latest_%s' % (self.config["output"]['output_directory'],self.config["run_info"]['run_tag']))

        signal = GenerateTemplate(config=self.config, run_type = self.run_type, save_dir = save_dir)
        
        signal_dataset = []
        signal_inj_pars = {}
        signal_inference_pars = []
        signal_snrs = []
        for i in range(int(self.config["data"]["file_split"])):

            # generate training sample source parameter, waveform and snr
            signal.clear_attributes()

            if self.config["data"]["use_real_detector_noise"] and self.config["data"]["save_polarisations"] == False:
                signal.generate_real_noise()

            signal.get_injection_parameters()
            signal.generate_polarisations()
            if self.config["data"]["save_polarisations"]:
                wfp = [signal.waveform_polarisations["plus"], signal.waveform_polarisations["cross"]]
                signal_dataset.append(wfp)
            else:
                # only works with gaussian noise at the moment
                signal.get_detector_response()
                signal_dataset.append(signal.whitened_signals_td)
                signal_snrs.append(signal.snrs)
            for key, value in signal.save_injection_parameters.items():
                signal_inj_pars.setdefault(key, [])
                signal_inj_pars[key].append(value)


        if self.write_to_file:
            fname = os.path.join(save_dir, "data_{}_{}.h5py".format(start_ind,self.config["data"]['n_{}_data'.format(self.run_type)]))
            with h5py.File(fname, 'w') as hf:
                if self.config["data"]["save_polarisations"]:
                    hf.create_dataset('y_hplus_hcross', data=signal_dataset)
                else:
                    hf.create_dataset('y_data_noisefree', data=signal_dataset)
                inj_keys = []
                inj_vals = []
                for key, value in signal_inj_pars.items():
                    inj_keys.append(key)
                    inj_vals.append(value)
                hf.create_dataset('injection_parameters_values', data=inj_vals)
                hf.create_dataset('injection_parameters_keys', data=inj_keys)
                #hf.create_dataset('inference_parameters_keys', data=self.config["model"]["inf_pars_list"])
                hf.create_dataset('snrs', data=signal_snrs)
                hf.close()
        else:
            return signal_dataset, signal_inj_pars, signal_snrs


    def create_test_file(self, start_ind = 0, sampler = None):
        
        # Make testing set directory
        save_dir = os.path.join(self.config["data"]["data_directory"], self.run_type)
        waveform_dir = os.path.join(save_dir, "waveforms")
        sampler_dir = os.path.join(save_dir, "{}".format(sampler))
        directories = [save_dir, waveform_dir]
        if sampler is not None:
            directories.append(sampler_dir)

        for direc in directories:
            if not os.path.isdir(direc):
                os.makedirs(direc)

        fname = os.path.join(waveform_dir, "{}_{}.h5py".format("test_data",start_ind))

        signal = GenerateTemplate(config=self.config, run_type = self.run_type, save_dir = save_dir)
        signal.clear_attributes()

        if os.path.isfile(fname):
            with h5py.File(fname,'r') as hf:
                print(hf.keys())
                signal.whitened_signal_td = np.array(hf["y_data_noisefree"])
                signal.whitened_data_td = np.array(hf["y_data_noisy"])
                signal.unwhitened_data_td = np.array(hf["unwhitened_data_noisy"])
                signal.frequency_domain_strain = np.array(hf["frequency_domain_strain"])
                inj_keys = hf["injection_parameters_keys"]
                inj_vals = hf["injection_parameters_values"]
                signal.injection_parameters = {}
                signal.save_injection_parameters = {}
                for k in range(len(inj_keys)):
                    signal.save_injection_parameters[inj_keys[k].decode()] = inj_vals[k]
                    signal.injection_parameters[inj_keys[k].decode()] = inj_vals[k]
                #signal.snrs = hf["snrs"]
            signal.generate_polarisations()
            signal.get_detector_response(frequency_domain_strain = signal.frequency_domain_strain)
            self.write_to_file = False
        else:
            if self.config["data"]["use_real_detector_noise"]:
                signal.generate_real_noise()

            signal.get_injection_parameters()
            signal.generate_polarisations()
            signal.get_detector_response()

            if self.write_to_file:
                print("Generated: %s ..." % (fname))
                # Save generated testing samples in h5py format
                with h5py.File(fname,'w') as hf:
                    hf.create_dataset('y_data_noisefree', data=signal.whitened_signals_td)
                    hf.create_dataset('y_data_noisy', data=signal.whitened_data_td)
                    hf.create_dataset('unwhitened_data_noisy', data=signal.unwhitened_data_td)
                    hf.create_dataset('frequency_domain_strain', data=signal.frequency_domain_strain)
                    inj_keys = []
                    inj_vals = []
                    for key, value in signal.save_injection_parameters.items():
                        inj_keys.append(key)
                        inj_vals.append(value)
                    hf.create_dataset('injection_parameters_values', data=inj_vals)
                    hf.create_dataset('injection_parameters_keys', data=inj_keys)
                    #hf.create_dataset('inference_parameters_keys', data=self.config["model"]["inf_pars_list"])
                    hf.create_dataset('snrs', data=signal.snrs)
                    hf.close()

        if sampler is not None:
            if sampler == "grid":
                fname = os.path.join(sampler_dir, f"test_outputs_grid_{start_ind}.h5py")

                #initially need to test if dynesty/nessai posterior has been generated
                if "nessai" in self.config["testing"]["samplers"]:
                    t_sampler = "nessai"
                if "dynesty" in self.config["testing"]["samplers"]:
                    t_sampler = "dynesty"

                t_sampler_dir = os.path.join(save_dir, "{}".format(t_sampler))
            
                #filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)
                filename = os.path.join(t_sampler_dir, "{}_{}.h5py".format("test_outputs",start_ind))
                if os.path.isfile(filename) == False:
                    raise Exception(f"Samples not available for {t_sampler} for test data index {idx}")

                data_temp = {}
                inf_pars = list(self.config["inf_pars"].keys())
                with h5py.File(filename, 'r') as hf:
                    for q in inf_pars:
                        p = q + '_post'
                        data_temp[q] = hf[p][:]

                        if p == 'psi_post':
                            data_temp[p] = np.remainder(data_temp[q],np.pi)

                        # set min and max ranges for each parameter
                        minrange = np.min(data_temp[q])
                        maxrange = np.max(data_temp[q])
                        data_temp[q] = (minrange, maxrange)


                # set the ranges over which to place the grid based on the posterior from dynesty or nessai
                infpar1_range = np.linspace(data_temp[inf_pars[0]][0], data_temp[inf_pars[0]][1], self.config["testing"]["n_grid_points"])
                infpar2_range = np.linspace(data_temp[inf_pars[1]][0], data_temp[inf_pars[1]][1], self.config["testing"]["n_grid_points"])

                #################
                # compute the posterior grid
                ################
                signal.compute_posterior_grid(infpar1_range=infpar1_range, infpar2_range=infpar2_range)
                with h5py.File(fname, 'w') as hf:
                    hf.create_dataset('grid_posterior', data=signal.posterior_grid)
                    hf.create_dataset('grid_parameters', data=signal.posterior_grid_pars)
                    #hf.create_dataset('grid_prior', data=signal.prior_grid)
                    #hf.create_dataset('grid_likelihood', data=signal.likelihood_grid)
                    hf.close()
            else:
                fname = os.path.join(sampler_dir, "{}_{}.h5py".format("test_outputs",start_ind))
                signal.run_pe(sampler=sampler, start_ind = start_ind)
                with h5py.File(fname, 'w') as hf:
                    hf.create_dataset('noisy_waveform', data=signal.whitened_data_td)
                    hf.create_dataset('noisefree_waveform', data=signal.whitened_signals_td)
                    logl = getattr(signal,sampler).log_likelihood_evaluations
                    if logl is not None:
                        hf.create_dataset('log_like_eval', data=logl) 

                    # converting masses so have all representations
                    all_posterior_params = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(getattr(signal, sampler).posterior)[0]
                    if "chirp_mass" not in all_posterior_params:
                        all_posterior_params["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(all_posterior_params["mass_1"], all_posterior_params["mass_2"])
                    if "mass_ratio" not in all_posterior_params:
                        all_posterior_params["mass_ratio"] = bilby.gw.conversion.component_masses_to_mass_ratio(all_posterior_params["mass_1"], all_posterior_params["mass_2"])


                    for q,qi in all_posterior_params.items():
                        name = q + '_post'
                        print('saving PE samples for parameter {}'.format(q))
                        hf.create_dataset(name, data=np.array(qi))
                    hf.create_dataset('run_time', data=getattr(signal,"{}_runtime".format(sampler)))
                    hf.close()

    def create_training_noise_files(self, start_ind=0):

        # Make training set directory
        if self.write_to_file:
            save_dir = os.path.join(self.config["data"]["data_directory"], self.run_type)
            os.system('mkdir -p %s' % save_dir)

        # Make directory for plots
        #os.system('mkdir -p %s/latest_%s' % (self.config["output"]['output_directory'],self.config["run_info"]['run_tag']))

        signal = GenerateTemplate(config=self.config, run_type = self.run_type, save_dir = save_dir)
        
        #generate training noise sample 4096 length file with selected sampling frequency
        signal.clear_attributes()
        signal.generate_real_noise()

        if self.write_to_file:
            fname = os.path.join(save_dir, "data_{}_{}.h5py".format(start_ind,self.config["data"]['n_{}_data'.format(self.run_type)]))
            print("saving file to {}".format(fname))
            with h5py.File(fname, 'w') as hf:
                hf.create_dataset('y_noise', data=signal.noise_examples)
                hf.create_dataset('start_times', data=signal.noise_start_times)
                hf.create_dataset('durations', data=signal.noise_durations)
                hf.create_dataset('sample_rates', data=signal.noise_sample_rates)
                hf.close()
        else:
            return noise_dataset, noise_start_times, noise_durations, noise_sample_rates



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input files and options')
    parser.add_argument('--ini-file', metavar='i', type=str, help='path to ini file')
    parser.add_argument('--run-type', metavar='r', type=str, help='training/validation/test')
    parser.add_argument('--num-files', metavar='r', type=int, help='number of files to generate', default = 1000)
    parser.add_argument('--start-ind', metavar='s', type=int, help='index for file label, i.e. set which batch for running through condor', default=0)
    parser.add_argument('--sampler', type=str, help='which sampler to compare to', default="dynesty")

    args = parser.parse_args()
    vitamin_config = GWInputParser(args.ini_file)

    data = DataGenerator(vitamin_config, args.run_type, write_to_file = True)
    
    if args.run_type in ["training","validation"]:
        for i in range(args.num_files):
            data.create_training_val_file(start_ind = args.start_ind + i)
    elif args.run_type == "test":
        for i in range(args.num_files):
            data.create_test_file(start_ind=args.start_ind + i, sampler = args.sampler)
        #data.create_test_file(start_ind=args.start_ind, sampler = None)
    elif args.run_type in ["training_noise","validation_noise"]:
        for i in range(args.num_files):
            data.create_training_noise_files(start_ind = args.start_ind + i)


    
        
