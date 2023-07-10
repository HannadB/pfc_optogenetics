import numpy as np

import fklab.io.neuralynx as nlx
import fklab.signals.multirate
#import fklab.signals.ica as ica
#import fklab.signals.multitaper as mt
import fklab.signals.core
import fklab.signals.filter
import fklab.signals.ripple

#import matplotlib.pyplot as plt
import fklab.plot

from glob import glob
import h5py
import utilities

#%% Parameters

animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] 

#animals = ['B14','B18'] #higher fs - kernel dies

stimtype ='None' #do the 'None' sessions only - no artefacts
SESSION=0 #only go for the initial learning sessions

newfs = 1000 #downsample everything to 1000
Nchans=32
channelselec = 2 #for example 4 = take 1 in every 4 channels
indextype = 'Onmaze' #this is just for where to save the envelopes 

for animal in animals:
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)

    date = params['Alldates'][SESSION]
    folder = params['topfolder']+animal+'/'+date+'/'
    Nruns = len(glob(folder + 'epochs/run*'))
    save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'

    # load detections and stimulations
    events = nlx.NlxOpen(folder+'Events.nev').data[:]
    #stim_t = events.time[events.eventstring==b'stimulation']
    detect_t = events.time[events.eventstring==b'detection']

    #pick the right channel names
    #channels = np.arange(1,Nchans,channelselec) # start from one to match the ncs filenames
    #channames_ripples = params['channames_ripples'][SESSION] #also be sure to include the ripple channels
    #channels = np.unique(np.hstack((channels,channames_ripples)))
    channels = params['channames_ripples'][SESSION]
        
    #% load data
    loaded_channels = [] # this will contain the channels that were actually loaded
    data = [] # this will contain the decimated signals

    for chan in channels:

        f = nlx.NlxOpen(folder+'CSC'+str(chan)+'.ncs')
        fs = f.header['SamplingFrequency']
        decimate_factors = [int(fs/newfs)] #for most animals the decimate factor is 4
        if decimate_factors[0]==32: #in the case of B14 and B18, the decimate factor is 32 instead of 4 (sampling rate is 32kHz instead of 4kHz)
            decimate_factors = [2,4,4] #split the factors so we don't decimate all at once
            
        f.clip = False

        if len(f.data[:].signal)==0:
            print(f"Skipped channel {chan}")
            continue

        signal = f.data[:].signal
        time = f.data[:].time
        
        for decimate_factor in decimate_factors:
            time = time[::decimate_factor]
            signal = fklab.signals.multirate.decimate(signal, decimate_factor, axis=0)
            
        data.append(signal)

        loaded_channels.append(chan)      

        print('channel '+str(chan)+' done')

    data = np.column_stack(data)

    # filter the data and compute the envelope
    ripple_data = fklab.signals.filter.apply_filter(data, [140,220], fs=newfs, axis=0)
    ripple_envelope = np.column_stack(
        [fklab.signals.ripple.ripple_envelope(ripple_data[:,k], isfiltered=True) for k in range(ripple_data.shape[1])])
    
    reconstructed_ripple_envelope = ripple_envelope.copy() #since no artefact is removed here, the 'reconstructed envelope' is just the same as the original
    
    # save the reconstructed envelope
    with h5py.File(save_folder+'reconstructed_envelopes.hdf5', 'w') as h5f: 
        h5f.create_dataset('ripple_envelope', data=reconstructed_ripple_envelope)
        h5f.create_dataset('original_ripple_envelope', data=ripple_envelope)
        h5f.create_dataset('time', data=time)
        h5f.create_dataset('channels', data=np.vstack(loaded_channels))
        h5f.create_dataset('fs', data=[newfs])
        
    print(animal+' done')