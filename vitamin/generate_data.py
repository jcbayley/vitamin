from .create_template import GenerateTemplate


class DataGenerator():

    def __init__(self, config, run_type, write_to_file = True):
        """
        Generate training/validation/test data
        """
        self.config = config
        self.run_type = run_type
        self.write_to_file = True
        
    def create_training_val_file(self):

        # Make training set directory
        if self.write_to_file:
            save_dir = os.path.join(self.config["data"]["data_directory"], self.run_type)
            os.system('mkdir -p %s' % save_dir)

        # Make directory for plots
        os.system('mkdir -p %s/latest_%s' % (self.config["outputs"]['output_dir'],self.config["run_info"]['run_tag']))

        signal = GenerateTemplate(config=self.config, run_type = self.run_type, save_dir = save_dir)
        
        signal_dataset = []
        signal_inj_pars = []
        signal_snrs = []
        for i in range(0,self.config["data"]['{}_dataset_size'.format(self.run_type)],self.params['file_split']):

            # generate training sample source parameter, waveform and snr
            signal.clear_attributes()

            if self.params["use_real_detector_data"]:
                signal.generate_real_noise()

            signal_inj_pars.append(signal.injection_parameters)
            signal_snrs.append(signal.snrs)
            signal.generate_polarisations()
            if save_polarisations:
                signal_dataset.append(signal.waveform_polarisations)
            else:
                # only works with gaussian noise at the moment
                signal.get_detector_response()
                signal_dataset.append(signal.whitened_signal_td)

        if self.write_to_file:
            hf = h5py.File('%s/data_%d-%d.h5py' % (save_dir,(i+self.config["data"]['training_file_split']),self.config["data"]['{}_dataset_size',format(self.run_type)]), 'w')
            hf.create_dataset('x_data', data=signal_inj_pars)
            hf.create_dataset('y_data_noisefree', data=signal_dataset)
            hf.create_dataset('snrs', data=signal_snrs)
            hf.close()
        else:
            return signal_dataset, signal_inj_pars, signal_snrs


    def generate_test_data(self):
        
        # Make testing set directory
        save_dir = os.path.join(self.config["data"]["data_directory"], self.run_type)
        if not os.path.isdir(save_dir):
            os.system('mkdir -p %s' % save_dir)

        signal = GenerateTemplate(config=self.config, run_type = self.run_type)
        signal.clear_attributes()
        
        if self.config["data"]["use_real_detector_data"]:
            signal.generate_real_noise()
            
        signal.generate_polarisations()
        signal.get_detector_response()
        signals.run_pe()

        if self.write_to_file:
            fname = "%s/%s_%s.h5py ..." % (save_dir,self.config["testing"]['bilby_results_label'],sig_ind)
            print("Generated: %s ..." % (fname))
        
            # Save generated testing samples in h5py format
            with h5py.File(fname,'w') as hf:
                hf.create_dataset('x_data', data=signal.injection_parameters)
                hf.create_dataset('y_data_noisefree', data=signal.whitened_td_noisefree)
                hf.create_dataset('y_data_noisy', data=signal.whitened_td_noisy)
                hf.create_dataset('snrs', data=signal.snrs)
                hf.close()




        
