import numpy as np
import bilby
from sys import exit
import os, glob, shutil
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
from scipy import integrate, interpolate
import scipy
import lalsimulation
import lal
import time
import h5py
import json
from scipy.ndimage.interpolation import shift
import argparse
from gwpy.timeseries import TimeSeries
from gwdatafind import find_urls


class GenerateTemplate():

    def __init__(self, config, run_type = "train", save_dir = None):
        """
        Create a BBH signal template in real noise or Gaussian noise. Run parameter estimation on signal
        """
        self.config = config
        self.save_dir = save_dir
        self.run_type = run_type
        #if self.config["data"]["use_real_detector_noise"]:
        #    self.real_noise_seg = self.config["data"]["{}_real_noise_seg".format(run_type.strip("_noise"))] 
        self.duration = self.config["data"]["duration"]
        if self.run_type == "test":
            # if test set add 2s to w5Caveform for use with lalinference
            self.duration += 2

        self.channel_name = "DCS-CALIB_STRAIN_C01"
        #self.get_prior()
        
    #def get_prior(self):
    #    self.prior = bilby.gw.prior.BBHPriorDict(self.config["data"]["prior_file"])
 
    def get_injection_parameters(self):
        """
        Get the injection parameters from the prior distribution
        Also get the parameters for inference to save
        """
        self.injection_parameters = self.config["priors"].sample()
        self.save_injection_parameters, added_keys = bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters(self.injection_parameters)
        if "chirp_mass" not in self.save_injection_parameters:
            self.save_injection_parameters["chirp_mass"] = bilby.gw.conversion.component_masses_to_chirp_mass(self.save_injection_parameters["mass_1"], self.save_injection_parameters["mass_2"])
        if "mass_ratio" not in self.save_injection_parameters:
            self.save_injection_parameters["mass_ratio"] = bilby.gw.conversion.component_masses_to_mass_ratio(self.save_injection_parameters["mass_1"], self.save_injection_parameters["mass_2"])


    def clear_attributes(self):
        """ Remove attributes to regenerate data
        """
        for key in ["injection_parameters","injection_parameters_list", "waveform_polarisations", "whitened_signals_td", "ifos", "waveform_generator", "snrs", "noise_samples", "noise_start_times", "noise_durations", "noise_sample_rates"]:
            if hasattr(self, key):
                delattr(self, key)

    def generate_polarisations(self):
        """ Generates the polarisations of signal to be combined later.
        """

        if self.config["data"]["sampling_frequency"]>4096:
            print('EXITING: bilby doesn\'t seem to generate noise above 2048Hz so lower the sampling frequency')
            exit(0)

        # compute the number of time domain samples
        self.Nt = int(self.config["data"]["sampling_frequency"]*self.duration)
        
        # define the start time of the timeseries
        self.start_time = self.config["data"]["ref_geocent_time"]-self.config["data"]["duration"]/2.0
        
        # Fixed arguments passed into the source model 
        waveform_arguments = dict(waveform_approximant=self.config["data"]["waveform_approximant"],
                                  reference_frequency=self.config["data"]["reference_frequency"], 
                                  minimum_frequency=self.config["data"]["minimum_frequency"],
                                  maximum_frequency=self.config["data"]["sampling_frequency"]/2.0)

        # Create the waveform_generator using a LAL BinaryBlackHole source function
        self.waveform_generator = bilby.gw.WaveformGenerator(
            duration=self.duration, sampling_frequency=self.config["data"]["sampling_frequency"],
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
            start_time=self.start_time)
        
        # extract waveform from bilby
        self.waveform_generator.parameters = self.injection_parameters
        self.waveform_polarisations = self.waveform_generator.frequency_domain_strain()
        #time_signal = self.waveform_generator.time_domain_strain()
        
    def get_detector_response(self, frequency_domain_strain = None):
        """
        Gets the whitened signal from polarisations
        """

        
        # initialise the interferometers
        ifos = bilby.gw.detector.InterferometerList(self.config["data"]['detectors'])
        
        # If user is specifying PSD file (bit of a messy section, will update)
        num_psd_files = len(self.config["data"]["psd_files"])
        if num_psd_files == 0:
            pass
        elif num_psd_files == 1:
            self.type_psd = self.config["data"]["psd_files"][0].split('/')[-1].split('_')[-1].split('.')[0]
            for int_idx,ifo in enumerate(ifos):
                if self.type_psd == 'psd':
                    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=self.config["data"]["psd_files"][0])
                elif self.type_psd == 'asd':
                    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.config["data"]["psd_files"][0])
                else:
                    print('Could not determine whether psd or asd ...')
                    exit()

        elif num_psd_files == len(ifos):
            self.type_psd = self.config["data"]["psd_files"][0].split('/')[-1].split('_')[-1].split('.')[0]
            for int_idx,ifo in enumerate(ifos):
                if self.type_psd == 'psd':
                    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=self.config["data"]["psd_files"][int_idx])
                elif self.type_psd == 'asd':
                    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.config["data"]["psd_files"][int_idx])
                else:
                    print('Could not determine whether psd or asd ...')
                    exit()

        else:
            print("Num psd files not valid: should be 0, 1 or number of detectors")

        # if strain privided set the strain from this
        if frequency_domain_strain is not None:
            for int_idx,ifo in enumerate(ifos):
                ifo.set_strain_data_from_frequency_domain_strain(frequency_domain_strain[int_idx],
                                                                 sampling_frequency=self.config["data"]["sampling_frequency"],
                                                                 duration=self.duration,
                                                                 start_time=self.start_time)
            
        else:
            if self.config["data"]["use_real_detector_noise"]:
                #load_files, start_times, durations, sample_rates = get_real_noise(params = params, channel_name = "DCS-CALIB_STRAIN_C01", real_noise_seg = real_noise_seg)
                if not hasattr(self, "noise_examples"):
                    raise Exception("Please load some real data using get_real_noise")
                else:
                    # need to test this peice of code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
                    for int_idx,ifo in enumerate(ifos):
                        rand_idx = int(np.random.uniform(0, len(self.noise_examples[int_idx]) - self.config["data"]["sampling_frequency"]*self.duration) )
                        temp_ts = TimeSeries(self.noise_examples[int_idx][rand_idx : rand_idx + int(self.config["data"]["sampling_frequency"]*self.duration)], sample_rate = self.config["data"]["sampling_frequency"], t0 = self.start_time)
                        ifo.strain_data.set_from_gwpy_timeseries(temp_ts)
            else:
                ifos.set_strain_data_from_power_spectral_densities(
                    sampling_frequency=self.config["data"]["sampling_frequency"], duration=self.duration,start_time=self.start_time)

        self.frequency_domain_strain = []
        for i in range(len(self.config["data"]["detectors"])):
            self.frequency_domain_strain.append(ifos[i].strain_data.frequency_domain_strain)

        # inject signal
        ifos.inject_signal(waveform_generator=self.waveform_generator,
                           parameters=self.injection_parameters,raise_error=False)
        
        print('... Injected signal')
        whitened_signal_td_all = []
        whitened_h_td_all = [] 
        unwhitened_h_td_all = []
        # iterate over ifos
        whiten_data = True
        for i in range(len(self.config["data"]['detectors'])):
            # get frequency domain noise-free signal at detector
            signal_fd = ifos[i].get_detector_response(self.waveform_polarisations, self.injection_parameters) 
            
            # get frequency domain signal + noise at detector
            h_fd = ifos[i].strain_data.frequency_domain_strain

            # whitening
            if whiten_data:
                # whiten frequency domain noise-free signal (and reshape/flatten)
                whitened_signal_fd = signal_fd/ifos[i].amplitude_spectral_density_array
                #whitened_signal_fd = whitened_signal_fd.reshape(whitened_signal_fd.shape[0])    
                
                # inverse FFT noise-free signal back to time domain and normalise
                whitened_signal_td = np.sqrt(2.0*self.Nt)*np.fft.irfft(whitened_signal_fd)
                
                # whiten noisy frequency domain signal
                whitened_h_fd = h_fd/ifos[i].amplitude_spectral_density_array
                
                # inverse FFT noisy signal back to time domain and normalise
                whitened_h_td = np.sqrt(2.0*self.Nt)*np.fft.irfft(whitened_h_fd)
                unwhitened_h_td = np.sqrt(2.0*self.Nt)*np.fft.irfft(h_fd)
                
                whitened_h_td_all.append([whitened_h_td])
                whitened_signal_td_all.append([whitened_signal_td])            
                unwhitened_h_td_all.append([unwhitened_h_td])
            else:
                whitened_h_td_all.append([h_fd])
                whitened_signal_td_all.append([signal_fd])
                    
        print('... Whitened signals')
        self.whitened_signals_td = np.squeeze(np.array(whitened_signal_td_all),axis=1)
        self.whitened_data_td = np.squeeze(np.array(whitened_h_td_all),axis=1)
        self.unwhitened_data_td = np.squeeze(np.array(unwhitened_h_td_all),axis=1)
        self.ifos = ifos
        #self.waveform_generator = waveform_generator
        self.snrs = [self.ifos[j].meta_data['optimal_SNR'] for j in range(len(self.config["data"]['detectors']))]


    def run_pe(self, sampler = "dynesty", start_ind = 0):
        """
        Run traditional PE on the saved waveform
        Current options are dynesty and nessai
        """
        label = "bilby_out_{}".format(start_ind)
        try:
            bilby.core.utils.setup_logger(outdir=self.save_dir, label=label)
        except Exception as e:
            print(e)
            pass

        # initialise likelihood
        phase_marginalization=self.config["testing"]["phase_marginalisation"]
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=self.waveform_generator, phase_marginalization=phase_marginalization,priors=self.config["priors"])

        # look for dynesty sampler option
        if sampler == "dynesty":

            run_startt = time.time()
            # Run sampler dynesty 1 sampler

            result = bilby.run_sampler(
                likelihood=likelihood, priors=self.config["priors"], sampler='dynesty', npoints=1000, nact=50, npool=8, dlogz=0.1, injection_parameters=self.injection_parameters, outdir=os.path.join(self.save_dir,sampler), label=label, save='hdf5', plot=True)
            run_endt = time.time()

            # save test sample waveform
            self.dynesty = result

            # loop over randomised params and save samples
            #for p in inf_pars:
            self.dynesty_runtime = run_endt - run_startt


        elif sampler == "nessai":

            run_startt = time.time()
            # Run sampler dynesty 1 sampler

            result = bilby.run_sampler(
                likelihood=likelihood, priors=self.config["priors"], sampler='nessai', npoints=1000, dlogz=0.1, injection_parameters=self.injection_parameters, outdir=os.path.join(self.save_dir,sampler), label=label, save='hdf5', plot=True,flow_class='gwflowproposal')
            run_endt = time.time()

            # save test sample waveform
            self.nessai = result

            # loop over randomised params and save samples
            #for p in inf_pars:
            self.nessai_runtime = run_endt - run_startt

        else:
            print("Currently only comparing to dynesty or nessai please choose one of these")

    def generate_real_noise(self,):
        """
        Get a segment of real noise
        """
        # compute the number of time domain samples
        Nt = int(self.config["data"]["sampling_frequency"]*self.duration)

        # Get ifos bilby variable
        ifos = bilby.gw.detector.InterferometerList(self.config["data"]["detectors"])

        start_range, end_range = self.config["data"]["{}_time_range".format(self.run_type)]

        # select times within range that do not overlap by the duration
        file_length_sec = 4096
        num_load_files = 1#int(0.1*num_segments/file_length_sec)

        num_f_seg = int((end_range - start_range)/file_length_sec)
        if num_f_seg == 0:
            num_f_seg = 1

        if num_load_files > num_f_seg:
            num_load_files = num_f_seg

        file_time = np.random.choice(num_f_seg, size=(num_load_files))*file_length_sec + start_range

        load_files   = []
        start_times  = []
        durations    = []
        sample_rates = []
        #for fle in range(num_load_files):
        fle = 0
        st1 = time.time()
        for ifo_idx,ifo in enumerate(ifos): # iterate over interferometers
            det_str = ifo.name
            gwf_url = find_urls("{}".format(det_str[0]), "{}_HOFT_C01".format(det_str), file_time[fle],file_time[fle] + self.duration)
            time_series = TimeSeries.read(gwf_url, "{}:{}".format(det_str,self.channel_name))
            time_series = time_series.resample(self.config["data"]["sampling_frequency"])
            load_files.append(time_series.value)
            start_times.append(time_series.t0.value)
            durations.append(time_series.duration.value)
            sample_rates.append(time_series.sample_rate.value)
            #ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.params["psd_files"][0])
            del time_series


        self.noise_examples = load_files
        self.noise_start_times = start_times
        self.noise_durations = durations
        self.noise_sample_rates = sample_rates
