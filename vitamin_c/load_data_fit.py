import natsort
import os
import h5py
import numpy as np
import tensorflow as tf
import time
from lal import GreenwichMeanSiderealTime
from astropy.time import Time
from astropy import coordinates as coord
import bilby 
from gwpy.timeseries import TimeSeries
import copy 
from gwdatafind import find_urls

class DataLoader(tf.keras.utils.Sequence):

    def __init__(self, input_dir,  batch_size = 512, params=None, bounds=None, masks = None, fixed_vals = None, test_set = False, silent = True, chunk_batch = 40, val_set = False, num_epoch_load = 4, shuffle=False,channel_name="DCS-CALIB_STRAIN_C01"):
        
        self.params = params
        self.bounds = bounds
        self.masks = masks
        self.fixed_vals = fixed_vals
        self.input_dir = input_dir
        self.test_set = test_set
        self.val_set = val_set
        self.silent = silent
        self.shuffle = shuffle
        self.batch_size = batch_size

        #load all filenames
        self.get_all_filenames()
        # get number of data examples as give them indicies
        self.num_data = len(self.filenames)*self.params["tset_split"]
        self.num_dets = len(self.params["det"])
        self.indices = np.arange(self.num_data)

        self.chunk_batch = chunk_batch
        self.chunk_size = self.batch_size*chunk_batch
        self.chunk_iter = 0
        self.max_chunk_num = int(np.floor((self.num_data/self.chunk_size)))
        #self.max_chunk_num = 50
        
        self.num_epoch_load = num_epoch_load
        self.epoch_iter = 0
        # will addthis to init files eventually
        self.params["noiseamp"] = 1
        self.channel_name = channel_name

        if self.params["use_real_det_noise"]:
            self.noise_files = os.listdir(self.params["noise_set_dir"])

    def __len__(self):
        """ number of batches per epoch"""
        return(int(self.chunk_batch))
        #return int(np.floor(len(self.filenames)*self.params["tset_split"]/self.batch_size))

    def load_next_chunk(self):
        """
        Loads in one chunk of data where the size if set by chunk_size
        """
        if self.test_set:
            self.X, self.Y_noisefree, self.Y_noisy, self.snrs, self.truths = self.load_waveforms(self.filenames, None)
        else:
            if self.chunk_iter >= self.max_chunk_num:
                print("Reached maximum number of chunks, restarting index and shuffling files")
                self.chunk_iter = 0
                np.random.shuffle(self.filenames)
                
            start_load = time.time()
            #print("Loading data from chunk {}".format(self.chunk_iter))
            # get the data indices for given chunk
            temp_chunk_indices = self.indices[self.chunk_iter*self.chunk_size:(self.chunk_iter + 1)*self.chunk_size]
            # get the filenames which these data indices live, take set to get single file index
            temp_filename_indices = np.array(list(set(np.floor(temp_chunk_indices/self.params["tset_split"])))).astype(int)
            # rewrite the data indices as the index within each file
            temp_chunk_indices = temp_chunk_indices % self.params["tset_split"]
            # if the index falls to zero then split as the file is the next one 
            temp_chunk_indices_split = np.split(temp_chunk_indices, np.where(np.diff(temp_chunk_indices) < 0)[0] + 1)

            self.X, self.Y_noisefree, self.Y_noisy, self.snrs, self.Y_noise = self.load_waveforms(self.filenames[temp_filename_indices], temp_chunk_indices_split)

            self.chunk_size = len(self.X)
            self.chunk_batch = np.floor(self.chunk_size/self.batch_size)
            end_load = time.time()
            print("load_time chunk {}: {}".format(self.chunk_iter, end_load - start_load))

            # check for infs or nans in training data
            failed = False
            if np.any(np.isnan(self.X)) or not np.all(np.isfinite(self.X)):
                print("NaN of Inf in parameters")
                failed = True
            elif np.any(np.isnan(self.Y_noisefree)) or not np.all(np.isfinite(self.Y_noisefree)):
                print("NaN or Inf in y data")
                failed = True

            if failed is True:
                # if infs or nans are present reload and augment this chunk until there are no nans
                self.load_next_chunk()
            else:
                # otherwise iterate to next chunk
                self.chunk_iter += 1
        

    def __getitem__(self, index = 0):
        """
        get waveforms from data
        X: wavefrom parameters
        Y: waveform
        """
        if self.test_set:
            # parameters, waveform noisefree, waveform noisy, snrs
            X, Y_noisefree, Y_noisy, snrs = self.X, self.Y_noisefree, self.Y_noisy, self.snrs
            #Y_noisy = Y_noisy/self.params["y_normscale"]
        else:

            start_index = index*self.batch_size #- (self.chunk_iter - 1)*self.chunk_size
            end_index = (index+1)*self.batch_size #- (self.chunk_iter - 1)*self.chunk_size
            X, Y_noisefree = self.X[start_index:end_index], self.Y_noisefree[start_index:end_index]
            # add noise here
            if self.params["use_real_det_noise"]:
                Y_noisefree = (Y_noisefree + self.Y_noise[start_index:end_index])/self.params["y_normscale"]
            else:
                Y_noisefree = (Y_noisefree + self.params["noiseamp"]*np.random.normal(size=np.shape(Y_noisefree), loc=0.0, scale=1.0))/self.params["y_normscale"]

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

    def get_whitened_signal_response(self, data):

        ifos = bilby.gw.detector.InterferometerList(self.params['det'])
        for int_idx,ifo in enumerate(ifos):
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.params["psd_files"][0])

        ifos.set_strain_data_from_power_spectral_densities(
            sampling_frequency=self.params["sample_rate"], duration=self.params["duration"],
            start_time=self.params["ref_geocent_time"] - self.params["duration"]/2)

        data["x_data"] = self.randomise_extrinsic_parameters(data["x_data"])

        all_signals = []
            
        for inj in range(len(data["x_data"])):
            injection_parameters = {key: data["x_data"][inj][ind] for ind, key in enumerate(self.params["inf_pars"])}
                    
            injection_parameters["geocent_time"] += self.params["ref_geocent_time"]

            Nt = self.params["sample_rate"]*self.params["duration"]
            whitened_signals_td = []
            polarisations = {"plus":data["y_hplus_hcross"][inj][:,0], "cross":data["y_hplus_hcross"][inj][:,1]}
            for dt in range(len(self.params['det'])):
                signal_fd = ifos[dt].get_detector_response(polarisations, injection_parameters) 
                whitened_signal_fd = signal_fd/ifos[dt].amplitude_spectral_density_array
                whitened_signal_td = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_signal_fd)
                whitened_signals_td.append(whitened_signal_td)            
                    
            all_signals.append(whitened_signals_td)

        data["y_data_noisefree"] = np.transpose(all_signals, [0,2,1])
               
        data["x_data"], time_correction = self.randomise_time(data["x_data"])
        data["x_data"], distance_correction = self.randomise_distance(data["x_data"], data["y_data_noisefree"])
        data["x_data"], phase_correction = self.randomise_phase(data["x_data"], data["y_data_noisefree"])

        y_temp_fft = np.fft.rfft(np.transpose(data["y_data_noisefree"], [0,2,1]))*phase_correction*time_correction
        
        data["y_data_noisefree"] = np.transpose(np.fft.irfft(y_temp_fft),[0,2,1])#*distance_correction
        data["y_data_noisefree"] *= distance_correction
        del y_temp_fft
        return data

    def get_real_noise(self, segment_duration, segment_range, num_segments):

        # compute the number of time domain samples
        Nt = int(self.params["sample_rate"]*self.params["duration"])

        # Get ifos bilby variable
        ifos = bilby.gw.detector.InterferometerList(self.params["det"])

        start_range, end_range = segment_range
        # select times within range that do not overlap by the duration
        file_length_sec = 4096
        num_load_files = 1#int(0.1*num_segments/file_length_sec)

        num_f_seg = int((end_range - start_range)/file_length_sec)

        if num_load_files > num_f_seg:
            num_load_files = num_f_seg

        file_time = np.random.choice(num_f_seg, size=(num_load_files, len(self.params["det"])))*file_length_sec + start_range

        load_files = {det:[] for det in self.params["det"]}
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
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.params["psd_files"][0])
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
        Nt = int(self.params["sample_rate"]*self.params["duration"])

        # Get ifos bilby variable
        ifos = bilby.gw.detector.InterferometerList(self.params["det"])

        for ifo_idx,ifo in enumerate(ifos): # iterate over interferometers
            ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.params["psd_files"][0])


        file_choice = np.random.choice(self.noise_files, len(self.params["det"]))
        data = {ifo.name:[] for ifo in ifos}
        for find in file_choice:
            h5py_file = h5py.File(os.path.join(self.params["noise_set_dir"],find), 'r')
            for ifo_idx,ifo in enumerate(ifos):
                data[ifo.name].append(TimeSeries(h5py_file["real_noise_samples"][ifo_idx], sample_rate=h5py_file["sample_rate"][ifo_idx], t0=h5py_file["t0"][ifo_idx]))

        rand_times = np.random.uniform(0, 4096, size = (num_segments, len(self.params["det"])))
        fle = 0
        return_segments = []
        st1 = time.time()
        for seg in range(num_segments):
            file_ind = np.arange(len(self.params["det"]))
            np.random.shuffle(file_ind)
            temp_segment = []
            for ifo_idx,ifo in enumerate(ifos): # iterate over interferometers
                det_str = ifo.name
                
                rand_time = np.random.uniform(data[det_str][file_ind[ifo_idx]].t0.value + self.params["duration"], data[det_str][file_ind[ifo_idx]].t0.value + data[det_str][file_ind[ifo_idx]].duration.value - self.params["duration"])

                temp_ts = data[det_str][file_ind[ifo_idx]].crop(rand_time, rand_time + self.params["duration"])
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

        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': [], 'snrs': [], 'y_hplus_hcross': []}

        #idx = np.sort(np.random.choice(self.params["tset_split"],self.params["batch_size"],replace=False))
        for i,filename in enumerate(filenames):
            try:
                h5py_file = h5py.File(os.path.join(self.input_dir,filename), 'r')
                # dont like the below code, will rewrite at some point
                if self.test_set:
                    data['x_data'].append(h5py_file['x_data'])
                    data['y_data_noisefree'].append([h5py_file['y_data_noisefree']])
                    data['y_data_noisy'].append([h5py_file['y_data_noisy']])
                    data['rand_pars'] = h5py_file['rand_pars']
                    data['snrs'].append(h5py_file['snrs'])
                else:
                    data['x_data'].append(h5py_file['x_data'][indices[i]])
                    if self.params["save_polarisations"]:
                        data["y_hplus_hcross"].append(h5py_file["y_hplus_hcross"][indices[i]])
                    else:
                        data['y_data_noisefree'].append(h5py_file['y_data_noisefree'][indices[i]])
                    data['rand_pars'] = [i for i in h5py_file['rand_pars']]
                    #data['snrs'].append(h5py_file['snrs'][indices[i]])
                if not self.silent:
                    print('...... Loaded file ' + os.path.join(self.input_dir,filename))
            except OSError:
                print('Could not load requested file: {}'.format(filename))
                continue

        # concatentation all the x data (parameters) from each of the files
        data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
        if self.test_set:
            data['y_data_noisy'] = np.transpose(np.concatenate(np.array(data['y_data_noisy']), axis=0),[0,2,1])
        else:
            if self.params["save_polarisations"] == True:
                data['y_hplus_hcross'] = np.transpose(np.concatenate(np.array(data['y_hplus_hcross']), axis=0),[0,2,1])
            else:
                data['y_data_noisefree'] = np.transpose(np.concatenate(np.array(data['y_data_noisefree']), axis=0),[0,2,1])

        self.decoded_rand_pars, self.par_idx = self.get_infer_pars(data)
        # reorder x parameters into params['infer_pars'] order
        data["x_data"] = data['x_data'][:,self.par_idx]
        
        if self.test_set:
            truths = copy.copy(data["x_data"])

        if not self.silent:
            print('...... {} will be inferred'.format(infparlist))

        y_normscale = self.params["y_normscale"]
        if not self.test_set:
            if self.params["save_polarisations"]:
                data = self.get_whitened_signal_response(data)
            else:
                data["x_data"], time_correction = self.randomise_time(data["x_data"])
                data["x_data"], phase_correction = self.randomise_phase(data["x_data"], data["y_data_noisefree"])
                data["x_data"], distance_correction = self.randomise_distance(data["x_data"], data["y_data_noisefree"])
        
                # apply phase, time and distance corrections
                y_temp_fft = np.fft.rfft(np.transpose(data["y_data_noisefree"],[0,2,1]))*phase_correction*time_correction
                data["y_data_noisefree"] = np.transpose(np.fft.irfft(y_temp_fft),[0,2,1])*distance_correction

                pass
                #del y_temp_fft
        
            # add noise to the noisefree waveforms and normalise and normalise

            #data["y_data_noisefree"] = (data["y_data_noisefree"] + self.params["noiseamp"]*tf.random.normal(shape=tf.shape(data["y_data_noisefree"]), mean=0.0, stddev=1.0, dtype=tf.float32))/y_normscale
            #data["y_data_noisefree"] = data["y_data_noisefree"] 
        else:
            data["y_data_noisy"] = data["y_data_noisy"]/y_normscale

        data["x_data"] = self.convert_parameters(data["x_data"])



        if self.params["use_real_det_noise"] and not self.test_set:
            real_det_noise= self.load_real_noise(len(data["y_data_noisefree"]))

            data["y_data_noise"] = tf.cast(np.array(real_det_noise),dtype=tf.float32)

        # cast data to float
        data["x_data"] = tf.cast(data['x_data'],dtype=tf.float32)
        data["y_data_noisefree"] = tf.cast(data['y_data_noisefree'],dtype=tf.float32)
        data["y_data_noisy"] = tf.cast(data['y_data_noisy'],dtype=tf.float32)

        if self.test_set:
            return data['x_data'], data['y_data_noisefree'], data['y_data_noisy'],data['snrs'], truths
        else:
            if self.params["use_real_det_noise"]:
                return data['x_data'], data['y_data_noisefree'], data['y_data_noisy'],data['snrs'],data["y_data_noise"]
            else:
                return data['x_data'], data['y_data_noisefree'], data['y_data_noisy'],data['snrs'], None


    def convert_parameters(self, x_data):
        # convert the parameters from right ascencsion to hour angle
        x_data = convert_ra_to_hour_angle(x_data, self.params, self.params['inf_pars'])
        
        # convert phi to X=phi+psi and psi on ranges [0,pi] and [0,pi/2] repsectively - both periodic and in radians   
        
        for i,k in enumerate(self.params["inf_pars"]):
            #if k.decode('utf-8')=='psi':
            if k =='psi':
                psi_idx = i
            if k =='phase':
                phi_idx = i
            if k =='mass_1':
                m1_idx = i
            if k =='mass_2':
                m2_idx = i

        x_data[:,psi_idx], x_data[:,phi_idx] = psiphi_to_psiX(x_data[:,psi_idx],x_data[:,phi_idx])
        #replace m1 with chirp mass
        #x_data[:, m1_idx], x_data[:,m2_idx] = m1m2_to_chirpmassq(x_data[:,m1_idx], x_data[:,m2_idx])

        #min_chirp, minq = m1m2_to_chirpmassq(self.bounds["mass_1_min"],self.bounds["mass_2_min"])
        #max_chirp, maxq = m1m2_to_chirpmassq(self.bounds["mass_1_max"],self.bounds["mass_2_max"])
        
        # normalise to bounds
        #decoded_rand_pars, par_idx = self.get_infer_pars(data)

        for i,k in enumerate(self.params["inf_pars"]):
            par_min = k + '_min'
            par_max = k + '_max'
            # Ensure that psi is between 0 and pi
            if par_min == 'psi_min':
                x_data[:,i] = x_data[:,i]/(np.pi/2.0)
                #data['x_data'][:,i] = np.remainder(data['x_data'][:,i], np.pi)
            elif par_min=='phase_min':
                x_data[:,i] = x_data[:,i]/np.pi
            
            #elif k in "mass_1":
                #chirm mass
            #    x_data[:, i] = (x_data[:, i] - min_chirp)/(max_chirp - min_chirp)
            #elif k in "mass_2":
                # mass ratio
            #    x_data[:, i] = (x_data[:, i] - 0.125)/(1 - 0.125)
            
            else:
                x_data[:,i] = (x_data[:,i] - self.bounds[par_min]) / (self.bounds[par_max] - self.bounds[par_min])
        
        return x_data

    def unconvert_parameters(self, x_data):

        # unnormalise to bounds
        #min_chirp, minq = m1m2_to_chirpmassq(self.bounds["mass_1_min"],self.bounds["mass_2_min"])
        #max_chirp, maxq = m1m2_to_chirpmassq(self.bounds["mass_1_max"],self.bounds["mass_2_max"])
        for i,k in enumerate(self.params["inf_pars"]):
            par_min = k + '_min'
            par_max = k + '_max'

            # Ensure that psi is between 0 and pi
            if par_min == 'psi_min':
                x_data[:,i] = x_data[:,i]*(np.pi/2.0)
                #data['x_data'][:,i] = np.remainder(data['x_data'][:,i], np.pi)
            elif par_min=='phase_min':
                x_data[:,i] = x_data[:,i]*np.pi

            #elif k in "mass_1":
            #    x_data[:,i] = x_data[:, i]*(max_chirp - min_chirp) + min_chirp
            #elif k in "mass_2":
            #    x_data[:,i] = x_data[:, i]*(1 - 0.125) + 0.125
            
            else:
                x_data[:,i]=x_data[:,i]*(self.bounds[par_max] - self.bounds[par_min]) + self.bounds[par_min]


        # convert the parameters from right ascencsion to hour angle
        x_data = convert_hour_angle_to_ra(x_data, self.params, self.params['inf_pars'])

        # convert phi to X=phi+psi and psi on ranges [0,pi] and [0,pi/2] repsectively - both periodic and in radians   
        
        for i,k in enumerate(self.params["inf_pars"]):
            if k =='psi':
                psi_idx = i
            if k =='phase':
                phi_idx = i
            if k =='mass_1':
                m1_idx = i
            if k =='mass_2':
                m2_idx = i

        x_data[:,psi_idx], x_data[:,phi_idx] = psiX_to_psiphi(x_data[:,psi_idx],x_data[:,phi_idx])
        #replace m1 with chirp mass
        #x_data[:, m1_idx], x_data[:,m2_idx] = chirpmassq_to_m1m2(x_data[:,m1_idx], x_data[:,m2_idx])
        
        return x_data

    
    def get_infer_pars(self,data):

        # Get rid of encoding
        
        decoded_rand_pars = []
        for i,k in enumerate(data['rand_pars']):
            decoded_rand_pars.append(k.decode('utf-8'))
        
        # extract inference parameters from all source parameters loaded earlier
        # choose the indicies of which parameters to infer
        par_idx = []
        infparlist = ''
        for k in self.params["inf_pars"]:
            infparlist = infparlist + k + ', '
            for i,q in enumerate(decoded_rand_pars):
                if k==q:
                    par_idx.append(i)
        
        return decoded_rand_pars, par_idx


    def randomise_extrinsic_parameters(self, x):
        "randomise the extrinsic parameters"

        new_ra = np.random.uniform(size = len(x), low = self.bounds["ra_min"], high = self.bounds["ra_max"])
        # uniform in the sin of the declination (doesnt yet change if input prior is changed)
        new_dec = np.arcsin(np.random.uniform(size = len(x), low = np.sin(self.bounds["dec_min"]), high = np.sin(self.bounds["dec_max"])))
        new_psi = np.random.uniform(size = len(x), low = self.bounds["psi_min"], high = self.bounds["psi_max"])
        #new_geocent_time = np.random.uniform(size = len(x), low = self.bounds["geocent_time_min"], high = self.bounds["geocent_time_max"])

        #x[:, np.where(np.array(self.params["inf_pars"])=="geocent_time")[0][0]] = new_geocent_time
        x[:, np.where(np.array(self.params["inf_pars"])=="ra")[0][0]] = new_ra
        x[:, np.where(np.array(self.params["inf_pars"])=="dec")[0][0]] = new_dec
        x[:, np.where(np.array(self.params["inf_pars"])=="psi")[0][0]] = new_psi

        return x
        
    def randomise_phase(self, x, y):
        """ randomises phase of input parameter x"""
        # get old phase and define new phase
        old_phase = x[:,np.array(self.masks["phase_mask"]).astype(np.bool)]
        new_x = np.random.uniform(size=np.shape(old_phase), low=0.0, high=1.0)
        new_phase = self.bounds['phase_min'] + new_x*(self.bounds['phase_max'] - self.bounds['phase_min'])

        # defice 
        x[:, np.array(self.masks["phase_mask"]).astype(np.bool)] = new_phase

        phase_correction = -1.0*(np.cos(new_phase-old_phase) + 1j*np.sin(new_phase-old_phase))
        phase_correction = np.tile(np.expand_dims(phase_correction,axis=1),(1,self.num_dets,int(np.shape(y)[1]/2) + 1))

        return x, phase_correction

    def randomise_time(self, x):

        old_geocent = x[:,np.array(self.masks["geocent_mask"]).astype(np.bool)]
        new_x = np.random.uniform(size=np.shape(old_geocent), low=0.0, high=1.0)
        new_geocent = self.bounds['geocent_time_min'] + new_x*(self.bounds['geocent_time_max'] - self.bounds['geocent_time_min'])

        x[:, np.array(self.masks["geocent_mask"]).astype(np.bool)] = new_geocent

        fvec = np.arange(self.params['ndata']/2 + 1)/self.params['duration']

        time_correction = -2.0*np.pi*fvec*(new_geocent-old_geocent)
        time_correction = np.cos(time_correction) + 1j*np.sin(time_correction)
        time_correction = np.tile(np.expand_dims(time_correction,axis=1),(1,self.num_dets,1))
        
        return x, time_correction
        
    def randomise_distance(self,x, y):
        old_d = x[:, np.array(self.masks["dist_mask"]).astype(np.bool)]
        new_x = np.random.uniform(size=tf.shape(old_d), low=0.0, high=1.0)
        new_d = self.bounds['luminosity_distance_min'] + new_x*(self.bounds['luminosity_distance_max'] - self.bounds['luminosity_distance_min'])
        
        x[:, np.array(self.masks["dist_mask"]).astype(np.bool)] = new_d
                                                           
        dist_scale = np.tile(np.expand_dims(old_d/new_d,axis=1),(1,np.shape(y)[1],1))

        return x, dist_scale





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
    



###############
## Old Load data script
###############
        
def load_data(params,bounds,fixed_vals,input_dir,inf_pars,test_data=False,silent=False):
    """ Function to load either training or testing data.
    """

    # Get list of all training/testing files and define dictionary to store values in files
    if type("%s" % input_dir) is str:
        dataLocations = ["%s" % input_dir]
    else:
        print('ERROR: input directory not a string')
        exit(0)

    # Sort files by number index in file name using natsorted program
    filenames = sorted(os.listdir(dataLocations[0]))
    filenames = natsort.natsorted(filenames,reverse=False)

    # If loading by chunks, randomly shuffle list of training/testing filenames
    if params['load_by_chunks'] == True and not test_data:
        nfiles = np.min([int(params['load_chunk_size']/float(params['tset_split'])),len(filenames)])
        files_idx = np.random.randint(0,len(filenames),nfiles) 
        filenames= np.array(filenames)[files_idx]
        if not silent:
            print('...... shuffled filenames since we are loading in by chunks')

    # Iterate over all training/testing files and store source parameters, time series and SNR info in dictionary
    data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': [], 'snrs': []}
    for filename in filenames:
        if test_data:
            # Don't load files which are not consistent between samplers
            for samp_idx_inner in params['samplers'][1:]:
                inner_file_existance = True
                dataLocations_inner = '%s_%s' % (params['pe_dir'],samp_idx_inner+'1')
                filename_inner = '%s/%s_%d.h5py' % (dataLocations_inner,params['bilby_results_label'],int(filename.split('_')[-1].split('.')[0]))
                # If file does not exist, skip to next file
                inner_file_existance = os.path.isfile(filename_inner)
                if inner_file_existance == False:
                    break

            if inner_file_existance == False:
                print('File not consistent beetween samplers')
                continue
        try:
            data['x_data'].append(h5py.File(dataLocations[0]+'/'+filename, 'r')['x_data'][:])
            data['y_data_noisefree'].append(h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisefree'][:])

            if test_data:
                data['y_data_noisy'].append(h5py.File(dataLocations[0]+'/'+filename, 'r')['y_data_noisy'][:])
            data['rand_pars'] = h5py.File(dataLocations[0]+'/'+filename, 'r')['rand_pars'][:]
            data['snrs'].append(h5py.File(dataLocations[0]+'/'+filename, 'r')['snrs'][:])
            if not silent:
                print('...... Loaded file ' + dataLocations[0] + '/' + filename)
        except OSError:
            print('Could not load requested file')
            continue
    if np.array(data['y_data_noisefree']).ndim == 3:
        data['y_data_noisefree'] = np.expand_dims(np.array(data['y_data_noisefree']),axis=0)
        data['y_data_noisy'] = np.expand_dims(np.array(data['y_data_noisy']),axis=0)
    data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
    data['y_data_noisefree'] = np.transpose(np.concatenate(np.array(data['y_data_noisefree']), axis=0),[0,2,1])
    if test_data:
        data['y_data_noisy'] = np.transpose(np.concatenate(np.array(data['y_data_noisy']), axis=0),[0,2,1])
    data['snrs'] = np.concatenate(np.array(data['snrs']), axis=0)

    # Get rid of weird encoding bullshit
    decoded_rand_pars = []
    for i,k in enumerate(data['rand_pars']):
        decoded_rand_pars.append(k.decode('utf-8'))

    # convert ra to hour angle
    if test_data and len(data['x_data'].shape) == 1:
        data['x_data'] = np.expand_dims(data['x_data'], axis=0)
    if test_data:
        data['x_data'] = convert_ra_to_hour_angle(data['x_data'], params, decoded_rand_pars)
    else:
        data['x_data'] = convert_ra_to_hour_angle(data['x_data'], params, params['rand_pars'])

    # Normalise the source parameters
    #if test_data:
    #    print(data['x_data'])
    #    exit()
    for i,k in enumerate(decoded_rand_pars):
        par_min = k + '_min'
        par_max = k + '_max'

        # Ensure that psi is between 0 and pi
        if par_min == 'psi_min':
            data['x_data'][:,i] = np.remainder(data['x_data'][:,i], np.pi)

        # normalize by bounds
        data['x_data'][:,i]=(data['x_data'][:,i] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])

    # extract inference parameters from all source parameters loaded earlier
    idx = []
    infparlist = ''
    for k in inf_pars:
        infparlist = infparlist + k + ', '
        for i,q in enumerate(decoded_rand_pars):
            m = q
            if k==m:
                idx.append(i)
    data['x_data'] = tf.cast(data['x_data'][:,idx],dtype=tf.float32)
    if not silent:
        print('...... {} will be inferred'.format(infparlist))

    return data['x_data'], tf.cast(data['y_data_noisefree'],dtype=tf.float32), tf.cast(data['y_data_noisy'],dtype=tf.float32), data['snrs']

def convert_ra_to_hour_angle(data, params, pars, single=False):
    """
    Converts right ascension to hour angle and back again
    """

    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(params['ref_geocent_time'], format='gps', location=greenwich)
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
        # Iterate over all training samples and convert to hour angle
        for i in range(data.shape[0]):
            data[i,ra_idx] = t - data[i,ra_idx]

    return data

def convert_hour_angle_to_ra(data, params, pars, single=False):
    """
    Converts right ascension to hour angle and back again
    """
    greenwich = coord.EarthLocation.of_site('greenwich')
    t = Time(params['ref_geocent_time'], format='gps', location=greenwich)
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


def load_samples(params,sampler,pp_plot=False, bounds=None):
    """
    read in pre-computed posterior samples
    """
    if type("%s" % params['pe_dir']) is str:
        # load generated samples back in
        dataLocations = '%s_%s' % (params['pe_dir'],sampler+'2')
        print('... looking in {} for posterior samples'.format(dataLocations))
    else:
        print('ERROR: input samples directory not a string')
        exit(0)

    # Iterate over requested number of testing samples to use
#    for i in range(params['r']):
    i = i_idx = 0
    for samp_idx,samp in enumerate(params['samplers'][1:]):
        if samp == sampler:
            samp_idx = samp_idx
            break
    while i_idx < params['r']:
        filename = '%s/%s_%d.h5py' % (dataLocations,params['bilby_results_label'],i)
        for samp_idx_inner in params['samplers'][1:]:
            inner_file_existance = True
            if samp_idx_inner == samp_idx:
                inner_file_existance = os.path.isfile(filename)
                if inner_file_existance == False:
                    break
                else:
                    continue

            dataLocations_inner = '%s_%s' % (params['pe_dir'],samp_idx_inner+'2')
            filename_inner = '%s/%s_%d.h5py' % (dataLocations_inner,params['bilby_results_label'],i)
            # If file does not exist, skip to next file
            inner_file_existance = os.path.isfile(filename_inner)                
            if inner_file_existance == False:
                break

        if inner_file_existance == False:
            i+=1
            if i > 500:
                sys.exit()
            print('File does not exist for one of the samplers: {}'.format(filename))
            continue
        #if not os.path.isfile(filename):
        #    print('... unable to find file {}. Exiting.'.format(filename))
        #    exit(0)

        print('... Loading test sample -> ' + filename)
        data_temp = {}
        n = 0

        # Retrieve all source parameters to do inference on
        for q in params['bilby_pars']:
            p = q + '_post'
            par_min = q + '_min'
            par_max = q + '_max'
            data_temp[p] = h5py.File(filename, 'r')[p][:]
            
            if p == 'psi_post':
                data_temp[p] = np.remainder(data_temp[p],np.pi)
            elif p == 'geocent_time_post':
                data_temp[p] = data_temp[p] - params['ref_geocent_time']
            """
            # Convert samples to hour angle if doing pp plot
            if p == 'ra_post' and pp_plot:
                data_temp[p] = convert_ra_to_hour_angle(data_temp[p], params, None, single=True)
            data_temp[p] = (data_temp[p] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
            """            
            Nsamp = data_temp[p].shape[0]
            n = n + 1
        print('... read in {} samples from {}'.format(Nsamp,filename))

        # place retrieved source parameters in numpy array rather than dictionary
        j = 0
        XS = np.zeros((Nsamp,n))
        for p,d in data_temp.items():
            XS[:,j] = d
            j += 1
        print('... put the samples in an array')

        # Append test sample posteriors to existing array of other test sample posteriors
        rand_idx_posterior = np.linspace(0,Nsamp-1,num=params['n_samples'],dtype=np.int)
        np.random.shuffle(rand_idx_posterior)
        rand_idx_posterior = rand_idx_posterior[:params['n_samples']]
        
        if i == 0:
            XS_all = np.expand_dims(XS[rand_idx_posterior,:], axis=0)
        else:
            XS_all = np.vstack((XS_all,np.expand_dims(XS[rand_idx_posterior,:], axis=0)))
        print('... appended {} samples to the total'.format(params['n_samples']))
        i+=1; i_idx+=1
    return XS_all
