import h5py
import gwpy
import numpy as np
import bilby
import vitamin
from vitamin.gw import gw_parser
from pycbc import types
from pycbc import frame
import matplotlib.pyplot as plt
import argparse
from gwpy.timeseries import TimeSeries
import os

def load_waveform(fname, config_file, save_dir):

    with h5py.File(fname,"r") as f:
        print(f.keys())
        data = np.array(f["frequency_domain_strain"])
        inj_keys = f["injection_parameters_keys"]
        inj_vals = f["injection_parameters_values"]
        injection_parameters = {}
        save_injection_parameters = {}
        for k in range(len(inj_keys)):
            save_injection_parameters[inj_keys[k].decode()] = inj_vals[k]
            injection_parameters[inj_keys[k].decode()] = inj_vals[k]


    config = gw_parser.GWInputParser(config_file)

    duration = config["data"]["duration"] + 2
    start_time = config["data"]["ref_geocent_time"] - config["data"]["duration"]/2

    ifos = bilby.gw.detector.InterferometerList(config["data"]['detectors'])

    ifos_append = bilby.gw.detector.InterferometerList(config["data"]['detectors'])

    for int_idx,ifo in enumerate(ifos):
        ifo.set_strain_data_from_frequency_domain_strain(data[int_idx],
                                                         sampling_frequency=config["data"]["sampling_frequency"],
                                                         duration=duration,
                                                         start_time=start_time)

    #ifos_append.set_strain_data_from_power_spectral_densities(
    #    sampling_frequency=config["data"]["sampling_frequency"], duration=duration,start_time=start_time + config["data"]["duration"])



    waveform_arguments = dict(waveform_approximant=config["data"]["waveform_approximant"],
                              reference_frequency=config["data"]["reference_frequency"],
                              minimum_frequency=config["data"]["minimum_frequency"],
                              maximum_frequency=config["data"]["sampling_frequency"]/2.0)


    waveform_generator = bilby.gw.WaveformGenerator(
        duration=duration, sampling_frequency=config["data"]["sampling_frequency"],
        frequency_domain_source_model=bilby.gw.source.lal_binary_black_hole,
        parameter_conversion=bilby.gw.conversion.convert_to_lal_binary_black_hole_parameters,
        waveform_arguments=waveform_arguments,
        start_time=start_time)


    ifos.inject_signal(waveform_generator=waveform_generator,
                       parameters=injection_parameters,raise_error=False)


    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    """
    fig, ax = plt.subplots()
    ax.plot(ifos[0].strain_data.time_domain_strain)
    fig.savefig(os.path.join(save_dir, "test_plot.png"))

    fig, ax = plt.subplots()
    ax.plot(np.append(ifos[0].strain_data.time_domain_strain,ifos_append[0].strain_data.time_domain_strain))
    fig.savefig(os.path.join(save_dir, "test_plot_app.png"))


    """
    for int_idx,ifo in enumerate(ifos):
        #xzeros = np.zeros(config["data"]["sampling_frequency"]*2)
        #new_tdarr = np.append(ifos[int_idx].strain_data.time_domain_strain, zeros)
        new_tdarr = ifos[int_idx].strain_data.time_domain_strain
        fr = types.TimeSeries(new_tdarr, delta_t=1./ifos[int_idx].sampling_frequency)
        fr.start_time = start_time
        frame.write_frame(os.path.join(save_dir,f"{ifo.name}test.gwf"), f"{ifo.name}:TEST_VITAMIN", fr)
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--root-dir', type=str)
    parser.add_argument('--condition', type=str, default = None)
    args = parser.parse_args()

    fname = "/home/joseph.bayley/data/CBC/O4_2/data_1024Hz_4s_gaussnoise_polarisation_unim1m2_mr_test3/test/waveforms/test_data_3.h5py"
    config_file = "/home/joseph.bayley/public_html/CBC/vitamin_refactor_O4/chirp4s/vitamin_run1/config.ini"
    save_dir = "/home/joseph.bayley/public_html/CBC/lalinftest/"
    load_waveform(fname, config_file, save_dir)

