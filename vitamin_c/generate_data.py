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


class GenerateSignals():

    def __init__(self, params, bounds, fixed_vals):
        
        self.params = params
        self.bounds = bounds
        self.fixed_vals = fixed_vals


    def gen_template(duration,
                     sampling_frequency,
                     pars,
                     ref_geocent_time, psd_files=[],
                     use_real_det_noise=False,
                     real_noise_seg =[None,None],
                     return_polarisations = False,
                 ):
        """ Generates a whitened waveforms in Gaussian noise.
        
        Parameters
        ----------
        duration: float
            duration of the signal in seconds
        sampling_frequency: float
            sampling frequency of the signal
        pars: dict
            values of source parameters for the waveform
        ref_geocent_time: float
            reference geocenter time of injected signal
        psd_files: list
            list of psd files to use for each detector (if other than default is wanted)
        use_real_det_noise: bool
        if True, use real ligo noise around ref_geocent_time
        real_noise_seg: list
        list containing the starting and ending times of the real noise 
        segment
        return_polarisations: bool
        if true return freq domain h+ and hx waveform, otherwise injects signal into ifo and returns whitened waveform
        
        Returns
        -------
        whitened noise-free signal: array_like
        whitened noisy signal: array_like
        injection_parameters: dict
        source parameter values of injected signal
        ifos: dict
        interferometer properties
        waveform_generator: bilby function
        function used by bilby to inject signal into noise 
        """

        if sampling_frequency>4096:
            print('EXITING: bilby doesn\'t seem to generate noise above 2048Hz so lower the sampling frequency')
            exit(0)

            # compute the number of time domain samples
            Nt = int(self.params["sampling_frequency"]*self.params["duration"])

            # define the start time of the timeseries
            start_time = self.params["ref_geocent_time"]-self.params["duration"]/2.0

            # fix parameters here
            injection_parameters = dict(
                mass_1=self.pars['mass_1'],mass_2=self.pars['mass_2'], a_1=self.pars['a_1'], a_2=self.pars['a_2'], tilt_1=self.pars['tilt_1'], tilt_2=self.pars['tilt_2'],
                phi_12=self.pars['phi_12'], phi_jl=self.pars['phi_jl'], luminosity_distance=self.pars['luminosity_distance'], theta_jn=self.pars['theta_jn'], psi=self.pars['psi'],
                phase=self.pars['phase'], geocent_time=self.pars['geocent_time'], ra=self.pars['ra'], dec=self.pars['dec'])

    # Fixed arguments passed into the source model 
    waveform_arguments = dict(waveform_approximant=self.waveform_approximant,
                              reference_frequency=self.reference_frequency, 
                              minimum_frequency=self.minimum_frequency,
                              maximum_frequency=self.params["sampling_frequency"]/2.0)

    # Create the waveform_generator using a LAL BinaryBlackHole source function
    waveform_generator = bilby.gw.WaveformGenerator(
        duration=self.params["duration"], sampling_frequency=self.["sampling_frequency"],
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
        start_time=start_time)

    # create waveform
    wfg = waveform_generator

    # extract waveform from bilby
    wfg.parameters = injection_parameters
    freq_signal = wfg.frequency_domain_strain()
    time_signal = wfg.time_domain_strain()

    # Set up interferometers. These default to their design
    # sensitivity
    if return_polarisations == False:
        ifos = bilby.gw.detector.InterferometerList(pars['det'])

        # If user is specifying PSD files
        if len(psd_files) > 0:
            type_psd = psd_files[0].split('/')[-1].split('_')[-1].split('.')[0]
            for int_idx,ifo in enumerate(ifos):
                if self.type_psd == 'psd':
                    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=self.psd_file)
                elif self.type_psd == 'asd':
                    ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(asd_file=self.psd_file)
                else:
                    print('Could not determine whether psd or asd ...')
                    exit()
        if real_noise_seg:
            load_files, start_times, durations, sample_rates = get_real_noise(params = params, channel_name = "DCS-CALIB_STRAIN_C01", real_noise_seg = real_noise_seg)
            # need to test this peice of code !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            for int_idx,ifo in enumerate(ifos):
                rand_time = np.random.uniform(start_times[int_idx] + self.params["duration"], start_times[int_idx] + durations[int_idx] + self.params["duration"] )
                temp_ts = load_files[int_idx].crop(rand_time, rand_time + self.params["duration"])
                ifo.strain_data.set_from_gwpy_timeseries(temp_ts)

                ifo.set_strain_data_from_gwpy_timeseries(ts_1)
                ifo.set_strain_data(load_files[int_idx])
        else:
            ifos.set_strain_data_from_power_spectral_densities(
                sampling_frequency=sampling_frequency, duration=duration,
                start_time=start_time)

        # inject signal
        ifos.inject_signal(waveform_generator=waveform_generator,
                           parameters=injection_parameters)

        print('... Injected signal')
        whitened_signal_td_all = []
        whitened_h_td_all = [] 
        # iterate over ifos
        whiten_data = True
        for i in range(len(pars['det'])):
            # get frequency domain noise-free signal at detector
            signal_fd = ifos[i].get_detector_response(freq_signal, injection_parameters) 
            
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
        self.injection_parameters = injection_parameters
        self.ifos = ifos
        self.waveform_generator = waveform_generator
    
    else:
        self.plus_cross_polarisations = np.array([freq_signal["plus"], freq_signal["cross"]])
        self.injection_parameters = injection_parameters
        self.waveform_generator = waveform_generator

