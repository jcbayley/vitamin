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
        if self.config["data"]["use_real_det_noise"]:
            self.real_noise_seg = self.config["data"]["{}_real_noise_seg".format(run_type)] 

        self.channel_name = "DCS-CALIB_STRAIN_C01"
        
    def get_prior(self):
        self.prior = bilby.gw.prior.BBHPriorDict(self.config["data"]["prior_file"])
        
    def get_injection_parameters(self):
        self.injection_parameters = self.prior.sample()

    def clear_attributes(self):
        """ Remove attributes to regenerate
        """
        for key in ["injection_parameters", "waveform_polarisations", "whitened_signals_td", "ifos", "waveform_generator", "snrs"]:
            if hasattr(self, key):
                delattr(self, key)

    def generate_polarisations(self):
        """ Generates a whitened waveforms in Gaussian noise.
        """

        if sampling_frequency>4096:
            print('EXITING: bilby doesn\'t seem to generate noise above 2048Hz so lower the sampling frequency')
            exit(0)

        # compute the number of time domain samples
        Nt = int(self.config["data"]["sampling_frequency"]*self.config["data"]["duration"])
        
        # define the start time of the timeseries
        start_time = self.config["data"]["ref_geocent_time"]-self.config["data"]["duration"]/2.0
        
        # Fixed arguments passed into the source model 
        waveform_arguments = dict(waveform_approximant=self.waveform_approximant,
                                  reference_frequency=self.reference_frequency, 
                                  minimum_frequency=self.minimum_frequency,
                                  maximum_frequency=self.config["data"]["sampling_frequency"]/2.0)

        # Create the waveform_generator using a LAL BinaryBlackHole source function
        self.waveform_generator = bilby.gw.WaveformGenerator(
            duration=self.config["data"]["duration"], sampling_frequency=self.config["data"]["sampling_frequency"],
            frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
            parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
            waveform_arguments=waveform_arguments,
            start_time=start_time)
        
        # extract waveform from bilby
        self.waveform_generator.parameters = self.injection_parameters
        self.waveform_polarisations = self.waveform_generator.frequency_domain_strain()
        time_signal = self.waveform_generator.time_domain_strain()
        
    def get_detector_reponse(self,):

        ifos = bilby.gw.detector.InterferometerList(self.config["data"]['detectors'])
        
        # If user is specifying PSD files
        if len(psd_files) > 0:
            type_psd = psd_files[0].split('/')[-1].split('_')[-1].split('.')[0]
            for int_idx,ifo in enumerate(ifos):
                if self.type_psd == 'psd':
                    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=self.config["data"]["psd_files"][int_idx])
                elif self.type_psd == 'asd':
                    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.config["data"]["psd_files"][int_idx])
                else:
                    print('Could not determine whether psd or asd ...')
                    exit()
        if real_noise_seg:
            #load_files, start_times, durations, sample_rates = get_real_noise(params = params, channel_name = "DCS-CALIB_STRAIN_C01", real_noise_seg = real_noise_seg)
            if not hasattr(self, "noise_examples"):
                print("Please load some real data using get_real_noise")
                # need to test this peice of code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for int_idx,ifo in enumerate(ifos):
                rand_time = np.random.uniform(start_times[int_idx] + self.config["data"]["duration"], start_times[int_idx] + durations[int_idx] + self.config["data"]["duration"] )
                temp_ts = self.noise_examples[int_idx].crop(rand_time, rand_time + self.config["data"]["duration"])
                ifo.strain_data.set_from_gwpy_imeseries(temp_ts)

        else:
            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=sampling_frequency, duration=duration,
                start_time=start_time)

        # inject signal
        #ifos.inject_signal(waveform_generator=waveform_generator,
        #                   parameters=self.injection_parameters)



    
        print('... Injected signal')
        whitened_signal_td_all = []
        whitened_h_td_all = [] 
        # iterate over ifos
        whiten_data = True
        for i in range(len(self.config["data"]['detectors'])):
            # get frequency domain noise-free signal at detector
            signal_fd = ifos[i].get_detector_response(freq_signal, self.injection_parameters) 
            
            # get frequency domain signal + noise at detector
            h_fd = ifos[i].strain_data.frequency_domain_strain
            
            # whitening
            if whiten_data:
                # whiten frequency domain noise-free signal (and reshape/flatten)
                whitened_signal_fd = signal_fd/ifos[i].amplitude_spectral_density_array
                #whitened_signal_fd = whitened_signal_fd.reshape(whitened_signal_fd.shape[0])    
                
                # inverse FFT noise-free signal back to time domain and normalise
                whitened_signal_td = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_signal_fd)
                
                # whiten noisy frequency domain signal
                whitened_h_fd = h_fd/ifos[i].amplitude_spectral_density_array
                
                # inverse FFT noisy signal back to time domain and normalise
                whitened_h_td = np.sqrt(2.0*Nt)*np.fft.irfft(whitened_h_fd)
                
                whitened_h_td_all.append([whitened_h_td])
                whitened_signal_td_all.append([whitened_signal_td])            
            else:
                whitened_h_td_all.append([h_fd])
                whitened_signal_td_all.append([signal_fd])
            
        print('... Whitened signals')
        self.whitened_signals_td = np.squeeze(np.array(whitened_signal_td_all),axis=1)
        self.whitened_data_td = np.squeeze(np.array(whitened_h_td_all),axis=1)
        self.ifos = ifos
        self.waveform_generator = waveform_generator
        self.snrs = [self.ifos[j].meta_data['optimal_SNR'] for j in range(len(self.config["data"]['detectors']))]


    def run_pe(self):

        try:
            bilby.core.utils.setup_logger(outdir=self.save_dir, label=label)
        except Exception as e:
            print(e)
            pass

        
        phase_marginalization=True
        likelihood = bilby.gw.GravitationalWaveTransient(
            interferometers=self.ifos, waveform_generator=self.waveform_generator, phase_marginalization=phase_marginalization,
            priors=priors)

        # save test waveform information
        try:
            os.mkdir('%s' % (self.save_dir+'_waveforms'))
        except Exception as e:
            print(e)
            pass

        if params != None:
            hf = h5py.File('%s/data_%d.h5py' % (self.save_dir+'_waveforms',int(label.split('_')[-1])),'w')
            for k, v in params.items():
                try:
                    hf.create_dataset(k,data=v)
                except:
                    pass

            hf.create_dataset('x_data', data=np.array([temp]))
            hf.create_dataset('y_data_noisefree', data=test_samples_noisefree)
            hf.create_dataset('y_data_noisy', data=test_samples_noisy)
            hf.create_dataset('snrs', data=snr)
            hf.close()

        # look for dynesty sampler option
        print("samplers......", samplers)
        if sampler == "dynesty":

            run_startt = time.time()
            # Run sampler dynesty 1 sampler

            result = bilby.run_sampler(
                likelihood=likelihood, priors=self.priors, sampler='dynesty', npoints=1000, nact=50, npool=8, dlogz=0.1,
                injection_parameters=self.injection_parameters, outdir=self.save_dir+'_'+samplers[-1], label=label,
                save='hdf5', plot=True)
            run_endt = time.time()

            # save test sample waveform
            hf = h5py.File('%s_%s/%s.h5py' % (self.save_dir,sampler,label), 'w')
            hf.create_dataset('noisy_waveform', data=test_samples_noisy)
            hf.create_dataset('noisefree_waveform', data=test_samples_noisefree)

            # loop over randomised params and save samples
            #for p in inf_pars:
            for q,qi in result.posterior.items():
                #if p==q:
                name = q + '_post'
                print('saving PE samples for parameter {}'.format(q))
                hf.create_dataset(name, data=np.array(qi))
            hf.create_dataset('runtime', data=(run_endt - run_startt))
            hf.close()

            # return samples if not doing a condor run
            if condor_run == False:
                # Make a corner plot.
                result.plot_corner()
                # remove unecessary files
                png_files=glob.glob("%s_dynesty1/*.png*" % (self.save_dir))
                hdf5_files=glob.glob("%s_dynesty1/*.hdf5*" % (self.save_dir))
                pickle_files=glob.glob("%s_dynesty1/*.pickle*" % (self.save_dir))
                resume_files=glob.glob("%s_dynesty1/*.resume*" % (self.save_dir))
                pkl_files=glob.glob("%s_dynesty1/*.pkl*" % (self.save_dir))
                filelist = [png_files,pickle_files,resume_files,pkl_files]
                for file_type in filelist:
                    for file in file_type:
                        os.remove(file)
                print('finished running pe')
                return test_samples_noisy,test_samples_noisefree,np.array([temp]),snr
        else:
            print("Currently only comparing to dynesty")

    def get_real_noise(self,):

        # compute the number of time domain samples
        Nt = int(self.config["data"]["sample_rate"]*self.config["data"]["duration"])

        # Get ifos bilby variable
        ifos = bilby.gw.detector.InterferometerList(self.config["data"]["detectors"])

        if self.real_noise_seg is None:
            start_range, end_range = self.config["data"]["real_noise_time_range"]
        else:
            start_range, end_range = self.real_noise_seg
        # select times within range that do not overlap by the duration
        file_length_sec = 4096
        num_load_files = 1#int(0.1*num_segments/file_length_sec)

        num_f_seg = int((end_range - start_range)/file_length_sec)

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
            gwf_url = find_urls("{}".format(det_str[0]), "{}_HOFT_C01".format(det_str), file_time[fle],file_time[fle] + self.config["data"]["duration"])
            time_series = TimeSeries.read(gwf_url, "{}:{}".format(det_str,self.channel_name))
            time_series = time_series.resample(self.config["data"]["sample_rate"])
            load_files.append(time_series.value)
            start_times.append(time_series.t0.value)
            durations.append(time_series.duration.value)
            sample_rates.append(time_series.sample_rate.value)
            #ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.params["psd_files"][0])
            del time_series


        self.noise_examples = load_files
        self.noie_start_times = start_times
        self.noise_durations = durations
        self.noise_sample_rates = sample_rates
