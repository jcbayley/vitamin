import natsort
import os
import h5py
import numpy as np
import tensorflow as tf
import time
from lal import GreenwichMeanSiderealTime
from astropy.time import Time
from astropy import coordinates as coord

class DataLoader(tf.keras.utils.Sequence):

    def __init__(self, input_dir,  batch_size = 512, params=None, bounds=None, masks = None, fixed_vals = None, test_set = False, silent = True, chunk_batch = 40, val_set = False):
        
        self.params = params
        self.bounds = bounds
        self.masks = masks
        self.fixed_vals = fixed_vals
        self.input_dir = input_dir
        self.test_set = test_set
        self.val_set = val_set
        self.silent = silent
        self.shuffle = False
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
        
        self.num_epoch_load = 4
        self.epoch_iter = 0
        # will addthis to init files eventually
        self.params["noiseamp"] = 1


    def __len__(self):
        """ number of batches per epoch"""
        return(int(self.chunk_batch))
        #return int(np.floor(len(self.filenames)*self.params["tset_split"]/self.batch_size))

    def load_next_chunk(self):
        """
        Loads in one chunk of data where the size if set by chunk_size
        """
        if self.test_set:
            self.X, self.Y_noisefree, self.Y_noisy, self.snrs = self.load_waveforms(self.filenames, None)
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
            
            self.X, self.Y_noisefree, self.Y_noisy, self.snrs = self.load_waveforms(self.filenames[temp_filename_indices], temp_chunk_indices_split)

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
        else:

            start_index = index*self.batch_size #- (self.chunk_iter - 1)*self.chunk_size
            end_index = (index+1)*self.batch_size #- (self.chunk_iter - 1)*self.chunk_size
            X, Y_noisefree = self.X[start_index:end_index], self.Y_noisefree[start_index:end_index]
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

        data={'x_data': [], 'y_data_noisefree': [], 'y_data_noisy': [], 'rand_pars': [], 'snrs': []}

        #idx = np.sort(np.random.choice(self.params["tset_split"],self.params["batch_size"],replace=False))
        
        for i,filename in enumerate(filenames):
            try:
                h5py_file = h5py.File(os.path.join(self.input_dir,filename), 'r')
                # dont like the below code, will rewrite at some point
                if self.test_set:
                    data['x_data'].append(h5py_file['x_data'])
                    data['y_data_noisefree'].append([h5py_file['y_data_noisefree']])
                    data['rand_pars'] = h5py_file['rand_pars']
                    data['snrs'].append(h5py_file['snrs'])
                else:
                    data['x_data'].append(h5py_file['x_data'][indices[i]])
                    data['y_data_noisefree'].append(h5py_file['y_data_noisefree'][indices[i]])
                    data['rand_pars'] = [i for i in h5py_file['rand_pars']]
                    data['snrs'].append(h5py_file['snrs'][indices[i]])
                if not self.silent:
                    print('...... Loaded file ' + os.path.join(self.input_dir,filename))
            except OSError:
                print('Could not load requested file')
                continue

        # concatentation all the x data (parameters) from each of the files
        data['x_data'] = np.concatenate(np.array(data['x_data']), axis=0).squeeze()
        # concatenate, then transpose the dimensions for keras, from (num_templates, num_dets, num_samples) to (num_templates, num_samples, num_dets)
        data['y_data_noisefree'] = np.transpose(np.concatenate(np.array(data['y_data_noisefree']), axis=0),[0,2,1])
        data['snrs'] = np.concatenate(np.array(data['snrs']), axis=0)

        # convert the parameters from right ascencsion to hour angle
        data['x_data'] = convert_ra_to_hour_angle(data['x_data'], self.params, self.params['rand_pars'])

        decoded_rand_pars, par_idx = self.get_infer_pars(data)

        for i,k in enumerate(decoded_rand_pars):
            par_min = k + '_min'
            par_max = k + '_max'

            # Ensure that psi is between 0 and pi
            if par_min == 'psi_min':
                data['x_data'][:,i] = np.remainder(data['x_data'][:,i], np.pi)

            # normalize each parameter by its bounds
            data['x_data'][:,i] = (data['x_data'][:,i] - self.bounds[par_min]) / (self.bounds[par_max] - self.bounds[par_min])

        if not self.silent:
            print('...... {} will be inferred'.format(infparlist))

        # cast data to floats
        data["x_data"] = tf.cast(data['x_data'][:,par_idx],dtype=tf.float32)
        data["y_data_noisefree"] = tf.cast(data['y_data_noisefree'],dtype=tf.float32)
        data["y_data_noisy"] = tf.cast(data['y_data_noisy'],dtype=tf.float32)

        # randomise phase, time and distance
        
        data["x_data"], time_correction = self.randomise_time(data["x_data"])
        data["x_data"], phase_correction = self.randomise_phase(data["x_data"], data["y_data_noisefree"])
        data["x_data"], distance_correction = self.randomise_distance(data["x_data"], data["y_data_noisefree"])
        
        # apply phase, time and distance corrections
        y_temp_fft = tf.signal.rfft(tf.transpose(data["y_data_noisefree"],[0,2,1]))*phase_correction*time_correction
        data["y_data_noisefree"] = tf.transpose(tf.signal.irfft(y_temp_fft),[0,2,1])*distance_correction
        del y_temp_fft
        
        # add noise to the noisefree waveforms and normalise and normalise
        y_normscale = tf.cast(self.params['y_normscale'], dtype=tf.float32)
        data["y_data_noisefree"] = (data["y_data_noisefree"] + self.params["noiseamp"]*tf.random.normal(shape=tf.shape(data["y_data_noisefree"]), mean=0.0, stddev=1.0, dtype=tf.float32))/y_normscale
        
        return data['x_data'], data['y_data_noisefree'], data['y_data_noisy'],data['snrs']


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


    def randomise_phase(self, x, y):
        """ randomises phase of input parameter x"""
        # get old phase and define new phase
        old_phase = self.bounds['phase_min'] + tf.boolean_mask(x,self.masks["phase_mask"],axis=1)*(self.bounds['phase_max'] - self.bounds['phase_min'])
        new_x = tf.random.uniform(shape=tf.shape(old_phase), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
        new_phase = self.bounds['phase_min'] + new_x*(self.bounds['phase_max'] - self.bounds['phase_min'])
        # defice 
        x = tf.gather(tf.concat([tf.reshape(tf.boolean_mask(x,self.masks["not_phase_mask"],axis=1),[-1,tf.shape(x)[1]-1]), tf.reshape(new_x,[-1,1])],axis=1),tf.constant(self.masks["idx_phase_mask"]),axis=1)

        phase_correction = -1.0*tf.complex(tf.cos(new_phase-old_phase),tf.sin(new_phase-old_phase))
        phase_correction = tf.tile(tf.expand_dims(phase_correction,axis=1),(1,self.num_dets,tf.shape(y)[1]/2 + 1))
        return x, phase_correction

    def randomise_time(self, x):

        old_geocent = self.bounds['geocent_time_min'] + tf.boolean_mask(x,self.masks["geocent_mask"],axis=1)*(self.bounds['geocent_time_max'] - self.bounds['geocent_time_min'])
        new_x = tf.random.uniform(shape=tf.shape(old_geocent), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
        
        new_geocent = self.bounds['geocent_time_min'] + new_x*(self.bounds['geocent_time_max'] - self.bounds['geocent_time_min'])

        x = tf.gather(tf.concat([tf.reshape(tf.boolean_mask(x,self.masks["not_geocent_mask"],axis=1),[-1,tf.shape(x)[1]-1]), tf.reshape(new_x,[-1,1])],axis=1),tf.constant(self.masks["idx_geocent_mask"]),axis=1)

        fvec = tf.range(self.params['ndata']/2 + 1, dtype = tf.dtypes.float32)/self.params['duration']

        time_correction = -2.0*np.pi*fvec*(new_geocent-old_geocent)
        time_correction = tf.complex(tf.cos(time_correction),tf.sin(time_correction))
        time_correction = tf.tile(tf.expand_dims(time_correction,axis=1),(1,self.num_dets,1))
        
        return x, time_correction
        
    def randomise_distance(self,x, y):

        old_d = self.bounds['luminosity_distance_min'] + tf.boolean_mask(x,self.masks["dist_mask"],axis=1)*(self.bounds['luminosity_distance_max'] - self.bounds['luminosity_distance_min'])
        new_x = tf.random.uniform(shape=tf.shape(old_d), minval=0.0, maxval=1.0, dtype=tf.dtypes.float32)
        new_d = self.bounds['luminosity_distance_min'] + new_x*(self.bounds['luminosity_distance_max'] - self.bounds['luminosity_distance_min'])
        x = tf.gather(tf.concat([tf.reshape(tf.boolean_mask(x,self.masks["not_dist_mask"],axis=1),[-1,tf.shape(x)[1]-1]), tf.reshape(new_x,[-1,1])],axis=1),tf.constant(self.masks["idx_dist_mask"]),axis=1)
        dist_scale = tf.tile(tf.expand_dims(old_d/new_d,axis=1),(1,tf.shape(y)[1],1))

        return x, dist_scale







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
        dataLocations = '%s_%s' % (params['pe_dir'],sampler+'1')
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

            dataLocations_inner = '%s_%s' % (params['pe_dir'],samp_idx_inner+'1')
            filename_inner = '%s/%s_%d.h5py' % (dataLocations_inner,params['bilby_results_label'],i)
            # If file does not exist, skip to next file
            inner_file_existance = os.path.isfile(filename_inner)                
            if inner_file_existance == False:
                break

        if inner_file_existance == False:
            i+=1
            print('File does not exist for one of the samplers')
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
            # Convert samples to hour angle if doing pp plot
            if p == 'ra_post' and pp_plot:
                data_temp[p] = convert_ra_to_hour_angle(data_temp[p], params, None, single=True)
            data_temp[p] = (data_temp[p] - bounds[par_min]) / (bounds[par_max] - bounds[par_min])
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
