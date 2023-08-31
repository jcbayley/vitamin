import natsort
import os
import sys
import h5py
import numpy as np
import torch
import time
from lal import GreenwichMeanSiderealTime
from astropy.time import Time
from astropy import coordinates as coord
import bilby 
from gwpy.timeseries import TimeSeries
import copy 
from gwdatafind import find_urls
#from ..group_inference_parameters import group_outputs

class DataSet(torch.utils.data.Dataset):

    def __init__(self, input_dir, config=None, test_set = False, silent = True, val_set = False, num_epoch_load = 4, shuffle=False,channel_name="DCS-CALIB_STRAIN_C01"):
        
        self.config = config
        self.input_dir = input_dir
        self.test_set = test_set
        self.val_set = val_set
        self.silent = silent
        self.shuffle = shuffle
        self.batch_size = self.config["training"]["batch_size"]

        if self.test_set:
            self.run_type = "test"
        elif self.val_set:
            self.run_type = "validation"
        else:
            self.run_type = "training"

        #load all filenames
        self.get_all_filenames()
        # get number of data examples as give them indicies
        self.num_data = len(self.filenames)*self.config["data"]["file_split"]
        self.num_dets = len(self.config["data"]["detectors"])
        self.indices = np.arange(self.num_data)
        self.data_length = self.config["data"]["sampling_frequency"]*self.config["data"]["duration"]
        
        self.chunk_batch = self.config["training"]["chunk_batch"]
        self.chunk_size = self.batch_size*self.chunk_batch
        self.chunk_iter = 0
        self.max_chunk_num = int(np.floor((self.num_data/self.chunk_size)))

        self.num_epoch_load = self.config["training"]["num_epoch_load"]
        self.epoch_iter = 0
        # will addthis to init files eventually
        self.channel_name = channel_name
        self.unconvert_parameters = unconvert_parameters
        self.convert_parameters = convert_parameters
        self.verbose = False

        if self.config['training']['make_sig']:
            # define the start time of the timeseries
            self.start_time = self.config["data"]["ref_geocent_time"]-self.config["data"]["duration"]/2.0
            
            # Fixed arguments passed into the source model 
            waveform_arguments = dict(waveform_approximant=self.config["data"]["waveform_approximant"],
                                    reference_frequency=self.config["data"]["reference_frequency"], 
                                    minimum_frequency=self.config["data"]["minimum_frequency"],
                                    maximum_frequency=self.config["data"]["sampling_frequency"]/2.0)

            # Create the waveform_generator using a LAL BinaryBlackHole source function
            self.waveform_generator = bilby.gw.WaveformGenerator(
                duration=self.config["data"]["duration"], sampling_frequency=self.config["data"]["sampling_frequency"],
                frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
                parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
                waveform_arguments=waveform_arguments,
                start_time=self.start_time)


    def __len__(self):
        """ number of batches per epoch"""
        return(int(self.chunk_batch))
        #return int(np.floor(len(self.filenames)*self.params["tset_split"]/self.batch_size))

    def __getitem__(self):
        """
        Loads in one chunk of data where the size if set by chunk_size
        """

        X, Y_noisefree, Y_noisy, snrs, truths, psds = self.load_waveforms(self.filenames, None)
        if np.any(np.isnan(X)) or not np.all(np.isfinite(self.X)):
            print("NaN of Inf in parameters")
            failed = True
        elif np.any(np.isnan(Y_noisefree)) or not np.all(np.isfinite(Y_noisefree)):
            print("NaN or Inf in y data")
            failed = True

        return np.array(Y_noisefree), np.array(X)
            
        

    def __getitem__(self, index = 0):
        """
        get waveforms from data
        X: wavefrom parameters
        Y: waveform
        """
        if self.test_set:
            # parameters, waveform noisefree, waveform noisy, snrs
            X, Y_noisefree, Y_noisy, snrs, psds = self.X, self.Y_noisefree, self.Y_noisy, self.snrs, self.psds
            #Y_noisy = Y_noisy/self.params["y_normscale"]
        else:

            start_index = index*self.batch_size #- (self.chunk_iter - 1)*self.chunk_size
            end_index = (index+1)*self.batch_size #- (self.chunk_iter - 1)*self.chunk_size

            X, Y_noisefree = self.X[start_index:end_index], self.Y_noisefree[start_index:end_index]
            if self.config["model"]["include_psd"]:
                psds = self.psds[start_index:end_index]

            # add noise here
            if self.config["data"]["use_real_detector_noise"]:
                Y_noisefree = (Y_noisefree + self.Y_noise[start_index:end_index])/float(self.config["data"]["y_normscale"])
            elif self.config["data"]["randomise_psd_factor"] != 0:
                Y_noisefree = (Y_noisefree)/float(self.config["data"]["y_normscale"])
            elif self.config['training']['make_sig']:
                Y_noisefree = (Y_noisefree)/float(self.config["data"]["y_normscale"])
            else:
                # noise scale is a factor of two to match that of the test results out of bilby
                Y_noisefree = (Y_noisefree + np.random.normal(size=np.shape(Y_noisefree), loc=0.0, scale=2.0))/float(self.config["data"]["y_normscale"])

            return np.array(Y_noisefree), np.array(X)


    def on_epoch_end(self):
        """Updates indices after each epoch
        """
        self.epoch_iter += 1
        if self.epoch_iter > self.num_epoch_load and not self.test_set and not self.val_set:
            self.load_next_chunk()
            self.epoch_iter = 0

        if self.shuffle == True:
            np.random.shuffle(self.indices)


    def get_all_filenames(self):
        """
        Get a list of all of the filenames containing the waveforms
        """
        # Sort files by number index in file name using natsorted program
        self.filenames = np.array(natsort.natsorted(os.listdir(self.input_dir),reverse=False))
        if self.test_set:
            self.filenames = self.filenames[:self.config["data"]["n_test_data"]]

    def get_psd(self, psd_file, randomise_psd = False):
        """ Get random psd from list of psds (not yet working!!!!!) """
        psd_type = psd_file.split('/')[-1].split('_')[-1].split('.')[0]
        with open(psd_file, "r") as f:
            aparr = np.loadtxt(f)
        if randomise_psd and self.config["data"]["randomise_psd_factor"] != 0:
            # define the factor change as positive side of Gaussian with mean of 1 and some variance
            random_multipl = np.abs(np.random.normal(loc=1,scale = self.config["data"]["randomise_psd_factor"],size = np.shape(aparr)[0]))
            # for half of the components multiply by 1/fact
            random_multipl[np.random.randint(0, len(random_multipl),size = int(0.5*len(random_multipl)))] = 1./random_multipl[np.random.randint(0, len(random_multipl),size = int(0.5*len(random_multipl)))]
        if psd_type == "psd":
            if randomise_psd and self.config["data"]["randomise_psd_factor"] != 0:
                aparr[:,1] *= random_multipl
            psd_data = bilby.gw.detector.PowerSpectralDensity(psd_array=aparr[:,1], frequency_array=aparr[:,0])
        elif psd_type == "asd":
            if randomise_psd and self.config["data"]["randomise_psd_factor"] != 0:
                aparr[:,1] *= np.sqrt(random_multipl)
            psd_data = bilby.gw.detector.PowerSpectralDensity(asd_array=aparr[:,1], frequency_array=aparr[:,0]) 
        else:
            print('Could not determine whether psd or asd ...')
            exit()
        return psd_data

    def get_psd_for_ifo(self, ifos, random_psd_factor = False):
        """
        load the psd from a file or ue the default psd for a bilby ifo object
        """
        num_psd_files = len(self.config["data"]["psd_files"])
        if num_psd_files == 0:
            raise FileNotFoundError("Please specify some psd file to whiten data with")
        elif num_psd_files == 1:
            self.type_psd = self.config["data"]["psd_files"][0].split('/')[-1].split('_')[-1].split('.')[0]
            for int_idx,ifo in enumerate(ifos):
                ifo.power_spectral_density = self.get_psd(psd_file=self.config["data"]["psd_files"][0], randomise_psd = random_psd_factor)
        elif num_psd_files > 1:
            self.type_psd = self.config["data"]["psd_files"][0].split('/')[-1].split('_')[-1].split('.')[0]
            for int_idx,ifo in enumerate(ifos):
                ifo.power_spectral_density = self.get_psd(psd_file=self.config["data"]["psd_files"][int_idx], randomise_psd = random_psd_factor)
        

    def get_whitened_signal_response(self, data):

        stload = time.time()
        ifos = bilby.gw.detector.InterferometerList(self.config["data"]['detectors'])

        self.get_psd_for_ifo(ifos)

        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.config["data"]["sampling_frequency"], duration=self.config["data"]["duration"],
            start_time=self.config["data"]["ref_geocent_time"] - self.config["data"]["duration"]/2)

        if self.verbose:
            print("Time ifo setup", time.time() - stload)

        stload = time.time()

        data["x_data"] = self.randomise_extrinsic_parameters(data["x_data"])

        if self.verbose:
            print("Random par setup", time.time() - stload)
        
        all_signals = np.zeros((len(data["x_data"]), len(self.config["data"]["detectors"]), int(self.config["data"]["sampling_frequency"]*self.config["data"]["duration"])))
            
        stload = time.time()
        for inj in range(len(data["x_data"])):
            #injection_parameters = {key: data["x_data"][inj][ind] for ind, key in enumerate(self.injection_parameters)}
            injection_parameters = {key: data["x_data"][inj][ind] for ind, key in enumerate(self.injection_parameters)}
                    
            #injection_parameters["geocent_time"] += self.config["data"]["ref_geocent_time"]

            Nt = self.config["data"]["sampling_frequency"]*self.config["data"]["duration"]
            whitened_signals_td = []
            polarisations = {"plus":data["y_hplus_hcross"][inj][:,0], "cross":data["y_hplus_hcross"][inj][:,1]}
            for dt in range(len(self.config["data"]['detectors'])):
                signal_fd = ifos[dt].get_detector_response(polarisations, injection_parameters)
                whitened_signal_fd = signal_fd/ifos[dt].amplitude_spectral_density_array
                all_signals[inj,dt] = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_signal_fd)
                #whitened_signals_td.append(whitened_signal_td)
                    
            #all_signals.append(whitened_signals_td)
        if self.verbose:
            print("Response setup", time.time() - stload)

        stload = time.time()
        data["y_data_noisefree"] = np.transpose(all_signals, [0,2,1])
               
        #data["x_data"], time_correction = self.randomise_time(data["x_data"])
        data["x_data"], distance_correction = self.randomise_distance(data["x_data"], data["y_data_noisefree"])
        data["x_data"], phase_correction = self.randomise_phase(data["x_data"], data["y_data_noisefree"])
        

        y_temp_fft = np.fft.rfft(np.transpose(data["y_data_noisefree"], [0,2,1]))*phase_correction
        
        data["y_data_noisefree"] = np.transpose(np.fft.irfft(y_temp_fft),[0,2,1])*distance_correction

        if self.verbose:
            print("Rand phase/dist setup", time.time() - stload)

        del y_temp_fft
        return data

    def get_whitened_signal(self, data):

        ifos = bilby.gw.detector.InterferometerList(self.config["data"]['detectors'])
        # set the psd once using a random factor (if specified in config) to generate strain data
        data["x_data"] = self.randomise_extrinsic_parameters(data["x_data"])
        
        # get initial psd to whiten with
        self.get_psd_for_ifo(ifos, random_psd_factor = False)

        all_signals = np.zeros((len(data["x_data"]), len(self.config["data"]["detectors"]), int(self.config["data"]["sampling_frequency"]*self.config["data"]["duration"])))

        for inj in range(len(data["x_data"])):

            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=self.config["data"]["sampling_frequency"], duration=self.config["data"]["duration"],
                start_time=self.config["data"]["ref_geocent_time"] - self.config["data"]["duration"]/2)

            data["x_data"] = self.randomise_extrinsic_parameters(data["x_data"])

            injection_parameters = {key: data["x_data"][inj][ind] for ind, key in enumerate(self.injection_parameters)}
                    
            ifos.inject_signal(waveform_generator=self.waveform_generator,
                           parameters=injection_parameters)
            #injection_parameters["geocent_time"] += self.config["data"]["ref_geocent_time"]

            Nt = self.config["data"]["sampling_frequency"]*self.config["data"]["duration"]
            whitened_signals_td = []
            psds = []
            for dt in range(len(self.config["data"]['detectors'])):
                whitened_signal_fd = ifos[dt].frequency_domain_strain/ifos[dt].amplitude_spectral_density_array
                all_signals[inj,dt] = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_signal_fd)
                    

        data["y_data_noisefree"] = np.transpose(all_signals, [0,2,1])

        return data

    def get_whitened_signal_noise(self, data):

        ifos = bilby.gw.detector.InterferometerList(self.config["data"]['detectors'])
        # set the psd once using a random factor (if specified in config) to generate strain data
        data["x_data"] = self.randomise_extrinsic_parameters(data["x_data"])
        
        # get initial psd to whiten with
        self.get_psd_for_ifo(ifos, random_psd_factor = False)
    
        if not self.config["model"]["include_psd"]:
            white_psds = []
            for dt in range(len(self.config["data"]['detectors'])):
                ifos[dt].sampling_frequency=self.config["data"]["sampling_frequency"]
                ifos[dt].duration=self.config["data"]["duration"]
                ifos[dt].start_time=self.config["data"]["ref_geocent_time"] - self.config["data"]["duration"]/2
                white_psds.append(ifos[dt].amplitude_spectral_density_array)

        all_signals = []
        all_psds = []
        for inj in range(len(data["x_data"])):
            self.get_psd_for_ifo(ifos, random_psd_factor = True)

            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=self.config["data"]["sampling_frequency"], duration=self.config["data"]["duration"],
                start_time=self.config["data"]["ref_geocent_time"] - self.config["data"]["duration"]/2)

            fdstrain = []
            for dt in range(len(self.config["data"]['detectors'])):
                noise_fd = ifos[dt].strain_data.frequency_domain_strain
                fdstrain.append(noise_fd)

            # redefine the psd based on the original psds without random factor (for whitening)
            #self.get_psd_for_ifo(ifos, random_psd_factor = False)

            #injection_parameters = {key: data["x_data"][inj][ind] for ind, key in enumerate(self.injection_parameters)}
            injection_parameters = {key: data["x_data"][inj][ind] for ind, key in enumerate(self.injection_parameters)}
                    
            #injection_parameters["geocent_time"] += self.config["data"]["ref_geocent_time"]

            Nt = self.config["data"]["sampling_frequency"]*self.config["data"]["duration"]
            whitened_signals_td = []
            polarisations = {"plus":data["y_hplus_hcross"][inj][:,0], "cross":data["y_hplus_hcross"][inj][:,1]}
            psds = []
            for dt in range(len(self.config["data"]['detectors'])):
                signal_fd = ifos[dt].get_detector_response(polarisations, injection_parameters)
                psd_noise = ifos[dt].amplitude_spectral_density_array
                if self.config["model"]["include_psd"]:
                    whitened_signal_fd = (signal_fd+fdstrain[dt])/psd_noise
                else:
                    whitened_signal_fd = (signal_fd+fdstrain[dt])/white_psds[dt]
                noise_nan = np.isnan(psd_noise) | np.isinf(psd_noise)
                psd_noise[noise_nan] = 0
                psds.append(np.array([psd_noise[:-1], psd_noise[:-1]]).flatten())
                whitened_signal_td = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_signal_fd)
                whitened_signals_td.append(whitened_signal_td)
                    
            all_signals.append(whitened_signals_td)
            if self.config["model"]["include_psd"]:
                all_psds.append(np.array(psds))

        data["y_data_noisefree"] = np.transpose(all_signals, [0,2,1])
        if self.config["model"]["include_psd"]:
            data["y_psds"] = np.transpose(all_psds, [0,2,1])
               
        data["x_data"], time_correction = self.randomise_time(data["x_data"])
        data["x_data"], distance_correction = self.randomise_distance(data["x_data"], data["y_data_noisefree"])
        data["x_data"], phase_correction = self.randomise_phase(data["x_data"], data["y_data_noisefree"])
        

        y_temp_fft = np.fft.rfft(np.transpose(data["y_data_noisefree"], [0,2,1]))*phase_correction*time_correction
        
        data["y_data_noisefree"] = np.transpose(np.fft.irfft(y_temp_fft),[0,2,1])#*distance_correction
        data["y_data_noisefree"] *= distance_correction

        # concat along second axis, (channels)
        if self.config["model"]["include_psd"]:
            data["y_data_noisefree"] = np.concat([data["y_data_noisefree"],data["y_psds"]], axis = 2)

        del y_temp_fft
        return data

    def get_real_noise(self, segment_duration, segment_range, num_segments):

        # compute the number of time domain samples
        Nt = int(self.config["data"]["sampling_frequency"]*self.config["data"]["duration"])

        # Get ifos bilby variable
        ifos = bilby.gw.detector.InterferometerList(self.config["data"]["detectors"])

        start_range, end_range = segment_range
        # select times within range that do not overlap by the duration
        file_length_sec = 4096
        num_load_files = 1#int(0.1*num_segments/file_length_sec)

        num_f_seg = int((end_range - start_range)/file_length_sec)

        if num_load_files > num_f_seg:
            num_load_files = num_f_seg

        file_time = np.random.choice(num_f_seg, size=(num_load_files, len(self.config["data"]["detectors"])))*file_length_sec + start_range

        load_files = {det:[] for det in self.config["data"]["detectors"]}
        start_times = {}
        #for fle in range(num_load_files):
        fle = 0
        st1 = time.time()
        for ifo_idx,ifo in enumerate(ifos): # iterate over interferometers
            det_str = ifo.name
            gwf_url = find_urls("{}".format(det_str[0]), "{}_HOFT_C01".format(det_str), file_time[fle][ifo_idx],file_time[fle][ifo_idx] + segment_duration)
            time_series = TimeSeries.read(gwf_url, "{}:{}".format(det_str,self.channel_name))
            time_series = time_series.resample(1024)
            start_times[det_str] = np.array(time_series.t0)
            load_files[det_str] = time_series
            num_psd_files = len(self.config["data"]["psd_files"])
            if num_psd_files == 0:
                pass
            else:
                ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.config["data"]["psd_files"][ifo_idx])
            del time_series

        st2 = time.time()
        starts = []
        for ifo_idx,ifo in enumerate(ifos): # iterate over interferometers
            det_str = ifo.name
            st_t = np.random.uniform(start_times[det_str] + segment_duration,start_times[det_str] + file_length_sec - segment_duration, num_segments)
            starts.append(st_t)

        noise_samples = []
        for st_ind in range(num_segments):
            t_noise_samp = []
            for ifo_idx,ifo in enumerate(ifos): # iterate over interferometers
                det_str = ifo.name
                t_time_series = load_files[det_str].crop(starts[ifo_idx][st_ind], starts[ifo_idx][st_ind] + segment_duration)
                ifo.set_strain_data_from_gwpy_timeseries(time_series=t_time_series) # input new ts into bilby ifo
                del t_time_series
                noise_sample = ifo.strain_data.frequency_domain_strain # get frequency domain strain
                noise_sample /= ifo.amplitude_spectral_density_array # normalise to a fixed psd
                noise_sample = np.sqrt(2.0*Nt)*np.fft.irfft(noise_sample) # convert frequency to time domain
                t_noise_samp.append(noise_sample)
            noise_samples.append(t_noise_samp)

        return np.transpose(np.array(noise_samples), (0,2,1))

    def load_real_noise(self, num_segments):

        # compute the number of time domain samples
        Nt = int(self.config["data"]["sampling_frequency"]*self.config["data"]["duration"])

        # Get ifos bilby variable
        ifos = bilby.gw.detector.InterferometerList(self.config["data"]["detectors"])
        self.get_psd_for_ifos(ifos)

        if self.test_set:
            nkey = "test_noise"
        elif self.val_set:
            nkey = "validation_noise"
        else:
            nkey = "training_noise"
        
        print("loading noise from : {}".format(os.path.join(self.config["data"]["data_directory"], nkey)))
        noise_files = os.listdir(os.path.join(self.config["data"]["data_directory"], nkey))

        file_choice = np.random.choice(noise_files)
        data = {ifo.name:[] for ifo in ifos}
        #for find in file_choice:
        h5py_file = h5py.File(os.path.join(self.config["data"]["data_directory"],"{}_noise".format(self.run_type),file_choice), 'r')
        for ifo_idx,ifo in enumerate(ifos):
            data[ifo.name] = TimeSeries(h5py_file["y_noise"][ifo_idx], sample_rate=h5py_file["sample_rates"][ifo_idx], t0=h5py_file["start_times"][ifo_idx])

        rand_times = np.random.uniform(0, 4096, size = (num_segments, len(self.config["data"]["detectors"])))
        fle = 0
        return_segments = []
        st1 = time.time()
        for seg in range(num_segments):
            #file_ind = np.arange(len(self.config["data"]["detectors"]))
            #np.random.shuffle(file_ind)
            temp_segment = []
            for ifo_idx,ifo in enumerate(ifos): # iterate over interferometers
                det_str = ifo.name
                
                rand_time = np.random.uniform(data[det_str].t0.value + self.config["data"]["duration"], data[det_str].t0.value + data[det_str].duration.value - self.config["data"]["duration"])

                temp_ts = data[det_str].crop(rand_time, rand_time + self.config["data"]["duration"])
                ifo.strain_data.set_from_gwpy_timeseries(temp_ts)

                h_fd = ifo.strain_data.frequency_domain_strain
                whitened_h_fd = h_fd/ifo.amplitude_spectral_density_array
                whitened_h_td = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_h_fd)
                
                temp_segment.append(whitened_h_td)
                
            return_segments.append(temp_segment)

        return np.transpose(np.array(return_segments), (0,2,1))

        
    def load_waveforms(self, filenames, indices = None):
        """
        load all the data from the filenames paths
        args
        ---------
        filenames : list
            list of filenames as string
        indices: list
            list of indices to get data from filenames. if multiple filenames, must be same length as filenames with the indices from each

        x_data           : parameters
        y_data_noisefree : waveform without noise
        y_data_noisy     : waveform with noise
        """

        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'snrs': [], 'y_hplus_hcross': [], 'y_psds':[]}

        #idx = np.sort(np.random.choice(self.params["tset_split"],self.params["batch_size"],replace=False))
        injpar_order = []
        if not self.config['training']['make_sig'] or self.test_set:
            for i,filename in enumerate(filenames):
                try:
                    stload = time.time()
                    h5py_file = h5py.File(os.path.join(self.input_dir,filename), 'r')
                    # dont like the below code, will rewrite at some point
                    if self.test_set:
                        t_noisefree = h5py_file['y_data_noisefree']
                        t_noisy = h5py_file['y_data_noisy'][:,:int(self.config["data"]["duration"]*self.config["data"]["sampling_frequency"])]
                        data['y_data_noisefree'].append(t_noisefree)
                        data['y_data_noisy'].append(t_noisy)
                        data['snrs'].append(h5py_file['snrs'])
                        injection_parameters_keys = [st.decode() for st in h5py_file['injection_parameters_keys']]
                        injection_parameters_values = np.array(h5py_file['injection_parameters_values'])

                        self.par_idx = self.get_infer_pars(self.config["model"]["inf_pars_list"], injection_parameters_keys)
                        self.injection_parameters = injection_parameters_keys
                        # reorder injection parameters into inference parameters order
                        data["x_data"].append([injection_parameters_values])#[self.par_idx]])
                        #data["x_data"].append([injection_parameters_values[self.par_idx]])

                    else:
                        if self.config["data"]["save_polarisations"]:
                            data["y_hplus_hcross"].append(h5py_file["y_hplus_hcross"][indices[i]])
                        else:
                            data['y_data_noisefree'].append(h5py_file['y_data_noisefree'][indices[i]])
                        injection_parameters_keys = [st.decode() for st in h5py_file['injection_parameters_keys']]
                        injection_parameters_values = np.array(h5py_file['injection_parameters_values'])[:, indices[i]]

                        self.par_idx = self.get_infer_pars(self.config["model"]["inf_pars_list"], injection_parameters_keys)
                        self.injection_parameters = injection_parameters_keys
                        # reorder injection parameters into inference parameters order
                        data["x_data"].append(np.array(injection_parameters_values.T))#[self.par_idx].T))
                        #data["x_data"].append(np.array(injection_parameters_values[self.par_idx].T))


                    if not self.silent:
                        print('...... Loaded file ' + os.path.join(self.input_dir,filename))
                except OSError:
                    print('Could not load requested file: {}'.format(filename))
                    continue
            if self.verbose:
                print("File load: ", time.time() - stload)
            # concatentation all the x data (parameters) from each of the files

            if self.test_set:
                data['y_data_noisy'] = np.transpose(np.array(data['y_data_noisy']),[0,2,1])
            else:
                if self.config["data"]["save_polarisations"] == True:
                    data['y_hplus_hcross'] = np.transpose(np.concatenate(np.array(data['y_hplus_hcross']), axis=0),[0,2,1])
                else:
                    data['y_data_noisefree'] = np.transpose(np.concatenate(np.array(data['y_data_noisefree']), axis=0),[0,2,1])
        else:
            data["x_data"] = []
            injection_parameters = self.config["priors"].sample(len(filenames)*1000)
            save_injection_parameters, added_keys = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(injection_parameters)
            if "chirp_mass" not in save_injection_parameters:
                save_injection_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(save_injection_parameters["mass_1"], save_injection_parameters["mass_2"])
            if "mass_ratio" not in save_injection_parameters:
                save_injection_parameters["mass_ratio"] = bilby.gw.conversion.component_masses_to_mass_ratio(save_injection_parameters["mass_1"], save_injection_parameters["mass_2"])
            self.injection_parameters = []
            injvals = []
            for key, val in save_injection_parameters.items():
                injvals.append(val)
                self.injection_parameters.append(key)
            data["x_data"].append(np.array(injvals).T)
            self.par_idx = self.get_infer_pars(self.config["model"]["inf_pars_list"], self.injection_parameters)

        stload = time.time()
        data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()

        

        
        if self.test_set:
            truths = copy.copy(data["x_data"])[:,self.par_idx]

        if not self.silent:
            print('...... {} will be inferred'.format(infparlist))

        if self.verbose:
            print("Time transpose: ", time.time() - stload)

        y_normscale = self.config["data"]["y_normscale"]
        stload = time.time()
        if not self.test_set:
            if self.config["data"]["save_polarisations"]:
                if float(self.config["data"]["randomise_psd_factor"]) != 0:
                    data = self.get_whitened_signal_noise(data)
                elif self.config['training']['make_sig']:
                    data = self.get_whitened_signal(data)
                else:
                    data = self.get_whitened_signal_response(data)
                if self.verbose:
                    print("Time getresponse: ", time.time() - stload)
            else:
                data["x_data"], time_correction = self.randomise_time(data["x_data"])
                data["x_data"], phase_correction = self.randomise_phase(data["x_data"], data["y_data_noisefree"])
                data["x_data"], distance_correction = self.randomise_distance(data["x_data"], data["y_data_noisefree"])
        
                # apply phase, time and distance corrections
                y_temp_fft = np.fft.rfft(np.transpose(data["y_data_noisefree"],[0,2,1]))*phase_correction*time_correction
                data["y_data_noisefree"] = np.transpose(np.fft.irfft(y_temp_fft),[0,2,1])*distance_correction

                pass
                #del y_temp_fft
            if self.verbose:
                print("beforeswap:", np.shape(data["y_data_noisefree"]))
                
            data["y_data_noisefree"]= np.swapaxes(data["y_data_noisefree"], 1,2)

            if self.verbose:
                print("afterswap:", np.shape(data["y_data_noisefree"]))
                
        else:
            if self.config["model"]["include_psd"]:
                ifos = bilby.gw.detector.InterferometerList(self.config["data"]['detectors'])
                # set the psd once using a random factor (if specified in config) to generate strain data
        
                # get initial psd to whiten with
                self.get_psd_for_ifo(ifos, random_psd_factor = False)
                psds = []
                for dt in range(len(self.config["data"]['detectors'])):
                    ifos[dt].sampling_frequency=self.config["data"]["sampling_frequency"]
                    ifos[dt].duration=self.config["data"]["duration"]
                    ifos[dt].start_time=self.config["data"]["ref_geocent_time"] - self.config["data"]["duration"]/2
                    psd_noise = ifos[dt].amplitude_spectral_density_array
                    noise_nan = np.isnan(psd_noise) | np.isinf(psd_noise)
                    psd_noise[noise_nan] = 0
                    psds.append(np.array([psd_noise[:-1], psd_noise[:-1]]).flatten())
                data["y_psds"] = np.repeat(np.expand_dims(np.array(psds).T,axis=0), np.shape(data["y_data_noisy"])[0], axis=0)
                data["y_data_noisy"] = np.concat([data["y_data_noisy"],data["y_psds"]], axis = 2)

            data["y_data_noisy"] = data["y_data_noisy"]/y_normscale

            if len(np.shape(data["y_data_noisy"])) != 3:
                raise Exception(f"data array should have three dimensions, current shape: {np.shape(data['y_data_noisy'])}")

            data['y_data_noisy'] = np.transpose(np.array(data['y_data_noisy']),[0,2,1])
            #np.swapaxes(data["y_data_noisy"], 1,2)

        stload = time.time()
        data["x_data"] = data["x_data"][:,self.par_idx]
        data["x_data"] = convert_parameters(self.config, data["x_data"])

        if np.any(data["x_data"] > 1) or np.any(data["x_data"] < 0):
            print("WARNING: data out of range [0,1]")

        # reorder parameters so can be grouped into different distributions
        #data["x_data"] = data["x_data"][:, self.config["masks"]["group_order_idx"]]

        if self.verbose:
            print("Time covert pars: ", time.time() - stload)
        
        if self.config["data"]["use_real_detector_noise"] and not self.test_set:
            real_det_noise = self.load_real_noise(len(data["y_data_noisefree"]))

            # cast data to float
            data["y_data_noisy"] = np.array(real_det_noise)
            np.swapaxes(data["y_data_noisy"], 1,2)

        if len(np.shape(data["y_data_noisefree"])) != 3:
            raise Exception(f"data array should have three dimensions, current shape: {np.shape(data['y_data_noisefree'])}")


        if self.test_set:
            for key in ["y_data_noisy"]:
                if np.array(data[key]).shape[1:] != (self.num_dets, self.data_length):
                    raise Exception(f"Incorrect data shape for {key}: {np.array(data[key]).shape} should be (N,{self.num_dets}, {self.data_length})")
        
        else:
            for key in ["y_data_noisefree"]:
                if np.array(data[key]).shape[1:] != (self.num_dets, self.data_length):
                    raise Exception(f"Incorrect data shape for {key}: {np.array(data[key]).shape} should be (N,{self.num_dets}, {self.data_length})")
  

        if self.test_set:
            return data['x_data'], data['y_data_noisefree'], data['y_data_noisy'],data['snrs'], truths, data["y_psds"]
        else:
            if self.config["data"]["use_real_detector_noise"]:
                return data['x_data'], data['y_data_noisefree'], data['y_data_noisy'],data['snrs'],data["y_data_noise"], data["y_psds"]
            else:
                return data['x_data'], data['y_data_noisefree'], data['y_data_noisy'],data['snrs'], None, data["y_psds"]

        


    
    def get_infer_pars(self,inference_parameters, injection_parameters):

        # Get rid of encoding
        
        # extract inference parameters from all source parameters loaded earlier
        # choose the indicies of which parameters to infer
        par_idx = []
        for k in inference_parameters:
            if k in injection_parameters:
                for i,q in enumerate(injection_parameters):
                    if k==q:
                        par_idx.append(i)
            else:
                raise Exception("Inference parameter not available in injection parameters, please convert injection parameters")
                    
        
        return par_idx


    def randomise_extrinsic_parameters(self, x):
        "randomise the extrinsic parameters"

        new_ra = np.random.uniform(size = len(x), low = self.config["bounds"]["ra_min"], high = self.config["bounds"]["ra_max"])
        # uniform in the sin of the declination (doesnt yet change if input prior is changed)
        new_dec = np.arcsin(np.random.uniform(size = len(x), low = np.sin(self.config["bounds"]["dec_min"]), high = np.sin(self.config["bounds"]["dec_max"])))
        new_psi = np.random.uniform(size = len(x), low = self.config["bounds"]["psi_min"], high = self.config["bounds"]["psi_max"])
        new_geocent_time = np.random.uniform(size = len(x), low = self.config["bounds"]["geocent_time_min"], high = self.config["bounds"]["geocent_time_max"])

        x[:, np.where(np.array(self.injection_parameters)=="geocent_time")[0][0]] = new_geocent_time
        x[:, np.where(np.array(self.injection_parameters)=="ra")[0][0]] = new_ra
        x[:, np.where(np.array(self.injection_parameters)=="dec")[0][0]] = new_dec
        x[:, np.where(np.array(self.injection_parameters)=="psi")[0][0]] = new_psi

        return x
        
    def randomise_phase(self, x, y):
        """ randomises phase of input parameter x"""
        # get old phase and define new phase
        old_phase = x[:,np.where(np.array(self.injection_parameters)=="phase")[0]]
        new_x = np.random.uniform(size=np.shape(old_phase), low=0.0, high=1.0)
        new_phase = self.config["bounds"]['phase_min'] + new_x*(self.config["bounds"]['phase_max'] - self.config["bounds"]['phase_min'])

        # defice 
        x[:, np.where(np.array(self.injection_parameters) == "phase")[0]] = new_phase

        phase_correction = -1.0*(np.cos(new_phase-old_phase) + 1j*np.sin(new_phase-old_phase))
        phase_correction = np.tile(np.expand_dims(phase_correction,axis=1),(1,self.num_dets,int(np.shape(y)[1]/2) + 1))

        return x, phase_correction

    def randomise_time(self, x):

        #old_geocent = x[:,np.array(self.config["masks"]["geocent_time_mask"]).astype(np.bool)]
        #print(np.shape(old_geocent))
        old_geocent = x[:,np.where(np.array(self.injection_parameters) == "geocent_time")[0]]
        #print(np.shape(old_geocent))
        #sys.exit()
        new_x = np.random.uniform(size=np.shape(old_geocent), low=0.0, high=1.0)
        new_geocent = self.config["bounds"]['geocent_time_min'] + new_x*(self.config["bounds"]['geocent_time_max'] - self.config["bounds"]['geocent_time_min'])

        x[:, np.where(np.array(self.injection_parameters) == "geocent_time")[0]] = new_geocent
        fvec = np.arange(self.data_length/2 + 1)/self.config["data"]['duration']

        time_correction = -2.0*np.pi*fvec*(new_geocent-old_geocent)
        time_correction = np.cos(time_correction) + 1j*np.sin(time_correction)
        time_correction = np.tile(np.expand_dims(time_correction,axis=1),(1,self.num_dets,1))
        
        return x, time_correction
        
    def randomise_distance(self,x, y):
        #old_d = x[:, np.array(self.config["masks"]["luminosity_distance_mask"]).astype(np.bool)]
        old_d = x[:, np.where(np.array(self.injection_parameters) == "luminosity_distance")[0]]

        new_d = self.config["priors"].sample(np.shape(old_d))["luminosity_distance"]

        x[:, np.where(np.array(self.injection_parameters) == "luminosity_distance")[0]] = new_d
                                                           
        dist_scale = np.tile(np.expand_dims(old_d/new_d,axis=1),(1,np.shape(y)[1],1))

        return x, dist_scale
        
    def load_bilby_samples(self):
        """
        read in pre-computed posterior samples
        """
        i_idx = 0

        self.sampler_outputs = {}
        for sampler in self.config["testing"]['samplers']:
            if sampler == "vitamin": 
                continue
            else:
                self.sampler_outputs[sampler] = []

        self.samples_available = {}
        got_samples = False
        save_dir = os.path.join(self.config["data"]["data_directory"], "test")
        for sampler in self.config["testing"]['samplers']:
            if sampler == "vitamin":
                continue
            sampler_dir = os.path.join(save_dir, "{}".format(sampler))
            samp_available_temp = []
            XS = np.zeros((self.config["data"]["n_test_data"], self.config["testing"]["n_samples"], len(self.config["testing"]['bilby_pars'])))
            for idx in range(self.config["data"]["n_test_data"]):
                #filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)
                filename = os.path.join(sampler_dir, "{}_{}.h5py".format("test_outputs",idx))
                #dataLocations_inner = '%s_%s' % (params['pe_dir'],samp_idx_inner+'2')
                #filename_inner = '%s/%s_%d.h5py' % (dataLocations_inner,params['bilby_results_label'],i)
                # If file does not exist, skip to next file
                if os.path.isfile(filename) == False:
                    print("no output file for example: {} and sampler: {}".format(idx, sampler))
                    continue
                samp_available_temp.append(idx)
                got_samples = True
                data_temp = {}

                # Retrieve all source parameters to do inference on
                for q in self.config["testing"]['bilby_pars']:
                    p = q + '_post'
                    par_min = q + '_min'
                    par_max = q + '_max'
                    with h5py.File(filename, 'r') as hf:
                        data_temp[p] = hf[p][:]

                    if p == 'psi_post':
                        data_temp[p] = np.remainder(data_temp[p],np.pi)
                    #elif p == 'geocent_time_post':
                    #x    data_temp[p] = data_temp[p] - self.config["data"]['ref_geocent_time']

                    """
                    # Convert samples to hour angle if doing pp plot
                    if p == 'ra_post' and pp_plot:
                    data_temp[p] = convert_ra_to_hour_angle(data_temp[p], params, None, single=True)
                    data_temp[p] = (data_temp[p] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
                    """            
                    Nsamp = data_temp[p].shape[0]
                    #n = n + 1
                print('... read in {} samples from {}'.format(Nsamp,filename))

                # place retrieved source parameters in numpy array rather than dictionary
                j = 0
                XS_temp = np.zeros((Nsamp,len(data_temp)))
                for p,d in data_temp.items():
                    XS_temp[:,j] = d
                    j += 1

                rand_idx_posterior = np.linspace(0,Nsamp-1,num=self.config["testing"]['n_samples'],dtype=np.int)
                np.random.shuffle(rand_idx_posterior)

                XS[idx] = XS_temp[rand_idx_posterior, :]
                self.samples_available[sampler] = samp_available_temp

            if got_samples:
                # Append test sample posteriors to existing array of other test sample posteriors                
                self.sampler_outputs[sampler] = np.array(XS)
        
        
        
def convert_parameters(config, x_data):
    # convert the parameters from right ascencsion to hour angle
    x_data = convert_ra_to_hour_angle(x_data, config, config["model"]['inf_pars_list'])
    
    for i,k in enumerate(config["model"]["inf_pars_list"]):
        #if k.decode('utf-8')=='psi':
        if k =='psi':
            psi_idx = i
        if k =='phase':
            phi_idx = i

    # convert phi to X=phi+psi and psi on ranges [0,pi] and [0,pi/2] repsectively - both periodic and in radians  

    if "psi" in config["model"]["inf_pars_list"] and "phase" in config["model"]["inf_pars_list"] and config["model"]["psiphi_to_psix"]:
        x_data[:,psi_idx], x_data[:,phi_idx] = psiphi_to_psiX(x_data[:,psi_idx],x_data[:,phi_idx])
    
    for i,k in enumerate(config["model"]["inf_pars_list"]):
        par_min = k + '_min'
        par_max = k + '_max'
        # these two are forced du the the psi/X reparameterisation
        # Ensure that psi is between 0 and p
        if config["model"]["psiphi_to_psix"]:
            if par_min == 'psi_min':
                x_data[:,i] = x_data[:,i]/(np.pi/2.0)
                #data['x_data'][:,i] = np.remainder(data['x_data'][:,i], np.pi)
            elif par_min=='phase_min':
                x_data[:,i] = x_data[:,i]/np.pi
            else:
                pass

        if par_min == "ra_min":
            # set the ramin and ramax as the min max of the hour angle to renormalise
            ramin = convert_ra_to_hour_angle(float(config["bounds"][par_min]), config, config["model"]['inf_pars_list'])
            ramax = convert_ra_to_hour_angle(float(config["bounds"][par_max]), config, config["model"]['inf_pars_list'])
            x_data[:,i] = (x_data[:,i] - ramin) / (ramax - ramin)

        else:
            # normalise remaininf parameters between bounds
            x_data[:,i] = (x_data[:,i] - config["bounds"][par_min]) / (config["bounds"][par_max] - config["bounds"][par_min])

    return x_data

def unconvert_parameters(config, x_data):

    x_data = np.array(x_data, np.float64)
    # unnormalise to bounds
    #min_chirp, minq = m1m2_to_chirpmassq(self.config["bounds"]["mass_1_min"],self.config["bounds"]["mass_2_min"])
    #max_chirp, maxq = m1m2_to_chirpmassq(self.config["bounds"]["mass_1_max"],self.config["bounds"]["mass_2_max"])
    for i,k in enumerate(config["model"]["inf_pars_list"]):
        par_min = k + '_min'
        par_max = k + '_max'

        # Ensure that psi is between 0 and pi
        if config["model"]["psiphi_to_psix"]:
            if par_min == 'psi_min':
                x_data[:,i] = x_data[:,i]*(np.pi/2.0)
                #data['x_data'][:,i] = np.remainder(data['x_data'][:,i], np.pi)
            elif par_min=='phase_min':
                x_data[:,i] = x_data[:,i]*np.pi
            else:
                pass

        elif par_min == "ra_min":
            ramin = convert_ra_to_hour_angle(float(config["bounds"][par_min]), config, config["model"]['inf_pars_list'])
            ramax = convert_ra_to_hour_angle(float(config["bounds"][par_max]), config, config["model"]['inf_pars_list'])
            x_data[:,i] = x_data[:,i]*(ramax - ramin) + ramax

        else:
            x_data[:,i] = x_data[:,i]*(float(config["bounds"][par_max]) - float(config["bounds"][par_min])) + float(config["bounds"][par_min])



    # convert the parameters from right ascencsion to hour angle
    x_data = convert_hour_angle_to_ra(x_data, config, config["model"]['inf_pars_list'])

    # convert phi to X=phi+psi and psi on ranges [0,pi] and [0,pi/2] repsectively - both periodic and in radians   
    
    for i,k in enumerate(config["model"]["inf_pars_list"]):
        if k =='psi':
            psi_idx = i
        if k =='phase':
            phi_idx = i

    if "psi" in config["model"]["inf_pars_list"] and "phase" in config["model"]["inf_pars_list"] and config["model"]["psiphi_to_psix"]:
        x_data[:,psi_idx], x_data[:,phi_idx] = psiX_to_psiphi(x_data[:,psi_idx],x_data[:,phi_idx])

    
    return x_data



def m1m2_to_chirpmassq(m1,m2):
    chirp_mass = ((m1*m2)**(3./5))/((m1 + m2)**(1./5))
    q = m2/m1
    return chirp_mass, q

def chirpmassq_to_m1m2(chirp_mass, q):
    m1 = chirp_mass*((1+q)/(q**3))**(1./5)
    m2 = m1*q
    return m1,m2

def psiphi_to_psiX(psi,phi):
    """
    Rescales the psi,phi parameters to a different space
    Input and Output in radians
    """
    X = np.remainder(psi+phi,np.pi)
    psi = np.remainder(psi,np.pi/2.0)
    return psi,X
    
def psiX_to_psiphi(psi,X):
    """     
    Rescales the psi,X parameters back to psi, phi
    Input and Output innormalised units [0,1]
    """
    r = np.random.randint(0,2,size=(psi.shape[0],2))
    #psi *= np.pi/2.0   # convert to radians                                                                                                                             
    #X *= np.pi         # convert to radians                                                                                                                             
    phi = np.remainder(X-psi + np.pi*r[:,0] + r[:,1]*np.pi/2.0,2.0*np.pi).flatten()
    psi = np.remainder(psi + np.pi*r[:,1]/2.0, np.pi).flatten()
    return psi,phi  # convert back to normalised [0,1] 
    
def convert_ra_to_hour_angle(data, config, pars, single=False):
    """
    Converts right ascension to hour angle and back again
    """

    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(config["data"]['ref_geocent_time'], format='gps', location=greenwich)
    t = t.sidereal_time('mean', 'greenwich').radian

    # compute single instance
    if single:
        return t - data

    for i,k in enumerate(pars):
        if k == 'ra':
            ra_idx = i

    # Check if RA exist
    try:
        ra_idx
    except NameError:
        print('...... RA is fixed. Not converting RA to hour angle.')
    else:
        if type(data) == float:
            data = t - data
        else:
            # Iterate over all training samples and convert to hour angle
            for i in range(data.shape[0]):
                data[i,ra_idx] = t - data[i,ra_idx]

    return data

def convert_hour_angle_to_ra(data, config, pars, single=False):
    """
    Converts right ascension to hour angle and back again
    """
    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(config["data"]['ref_geocent_time'], format='gps', location=greenwich)
    t = t.sidereal_time('mean', 'greenwich').radian

    # compute single instance
    if single:
        return np.remainder(t - data,2.0*np.pi)

    for i,k in enumerate(pars):
        if k == 'ra':
            ra_idx = i

    # Check if RA exist
    try:
        ra_idx
    except NameError:
        print('...... RA is fixed. Not converting RA to hour angle.')
    else:
        # Iterate over all training samples and convert to hour angle
        for i in range(data.shape[0]):
            data[i,ra_idx] = np.remainder(t - data[i,ra_idx],2.0*np.pi)

    return data


