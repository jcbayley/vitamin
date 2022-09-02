import tensorflow as tf
from ..vitamin_model import CVAE
from . import load_data
import os
import gwpy
import argparse
from gwdatafind import find_urls
import bilby
import numpy as np

def get_model(config, checkpoint_dir=None):
    """load in a CVAE based on the configuration file and the model weights"""
    model = CVAE(config)

    model.compile(run_eagerly = None)

    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(config["output"]["output_directory"],"checkpoint","model")

    model.load_weights(checkpoint_dir)

    return model

def get_gw_data(config, channel_names, merger_time=None, start_time=None, end_time=None, filenames = None, psd_files = []):
    """load a section of GW data"""
    if merger_time:
        start_time = merger_time - (config["priors"]["geocent_time"].minimum - config["data"]["ref_geocent_time"]) - (config["priors"]["geocent_time"].maximum - config["priors"]["geocent_time"].minimum)/2 - config["data"]["duration"]/2
        end_time = start_time + config["data"]["duration"]
    if start_time is not None:
        start_time = start_time
    if end_time is not None:
        end_time = end_time

    if end_time - start_time != config["data"]["duration"]:
        raise Exception(f"Please input data of length {config['data']['duration']}, currently using times {start_time} and {end_time} with duration {end_time - start_time}i.e. define a merger time or a start_time and end_time of correct duration")

    if psd_files is None:
        psd_files = config["data"]["psd_files"]

    whitened_ts = []
    ifo_list = bilby.gw.detector.InterferometerList([])
    for ifo_idx,det in enumerate(config["data"]["detectors"]): # iterate over interferometers
        print(f"Loading data from {det} channel {channel_names[det]} between {start_time} and {end_time} with duration {end_time - start_time}")
        ifo = bilby.gw.detector.get_empty_interferometer(det)
        if filenames is None:
            gwf_url = find_urls("{}".format(det), channel_names[det], start_time,end_time)
        else:
            gwf_url = filenames[det]
        time_series = gwpy.timeseries.TimeSeries.read(gwf_url, channel_names[det], start=start_time, end=end_time)
        time_series = time_series.resample(config["data"]["sampling_frequency"])
        ifo.strain_data.set_from_gwpy_timeseries(time_series)

        ifo.power_spectral_density = bilby.gw.detector.PowerSpectralDensity(psd_file=psd_files[ifo_idx])

        h_fd = ifo.strain_data.frequency_domain_strain
        whitened_h_fd = h_fd/ifo.amplitude_spectral_density_array
        whitened_h_td = np.sqrt(2.0*config["data"]["sampling_frequency"]*config["data"]["duration"])*np.fft.irfft(whitened_h_fd)

        whitened_ts.append(whitened_h_td)
    

    normed_wh_ts = np.array(whitened_ts)/config["data"]["y_normscale"]

    if np.shape(normed_wh_ts) != (len(config["data"]["detectors"]), config["data"]["sampling_frequency"]*config["data"]["duration"]):
        raise Exception(f"Data not right shape, should be (ndet, duration*samples_rate) = {(len(config['data']['detectors']), config['data']['sampling_frequency']*config['data']['duration'])} but it is {np.shape(normed_wh_ts)}")

    return normed_wh_ts.T

def get_samples(config, data, n_samples= 10000, checkpoint_dir = None, get_latent_samples = False):

    model = get_model(config, checkpoint_dir = checkpoint_dir)
    
    if get_latent_samples:
        mu_r1, z_r1, mu_q, z_q, scale_r1, scale_q, logvar_q = self.model.gen_z_samples(tf.expand_dims(data,0), tf.expand_dims(data,0), nsamples=1000)
    
    samples = model.gen_samples(tf.expand_dims(data,0), nsamples=n_samples,).numpy()

    samples = load_data.unconvert_parameters(config, samples)

    return samples
    #with open(os.path.join(output_dir, "posterior_samples.pkl"), "wb") as f:
    #    pickle.dump(samples)

def run_all(config, data):
    data = get_gw_data(config, channel_names, merger_time=merger_time, start_time=start_time, end_time=end_time, filenames = filenames)

    samples = get_samples(config, data)

    return samples

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Input files and options')
    parser.add_argument('--ini-file', metavar='i', type=str, help='path to ini file containing model parameters')
    parser.add_argument('--out-dir', type=str, help='output directory')
    parser.add_argument('--channel_name', type=str, help='output directory')
    #parser.add_argument('--gpu', metavar='i', type=int, help='path to ini file', default = None)
    args = parser.parse_args()