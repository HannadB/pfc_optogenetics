#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import yaml
import numpy as np
import scipy.signal as scipysignal
import fklab.signals.filter.filter as fklabfilter
import fklab.signals.ripple as ripple
import fklab.segments 
import fklab.geometry.shapes # to recognize YAML representations of shapes
#import gc
import fklab.signals.core as signalscore
import pandas as pd
import fklab.io.neuralynx as nlx
from math import sin, cos


def get_parameters(animal):

    with open("config.yaml", "r") as file:
        config = yaml.unsafe_load(file)

    return config[animal]

def point_pos(x0, y0, d, rad):
    """ get the distance to x and y
    given the starting point x0 and y0, the distance d and the radians rad """
    x1 = x0 + d*cos(rad)
    y1 = y0 + d*sin(rad)
    
    dx = x1-x0
    dy = y1-y0
    return dx,dy 

def artefact_removal(HC_epoch_data,HC_epoch_time,stimpertrial,fs,**kwargs):
    
    """ inputs:
            stimpertrial - the times of the stimulations for each trial (learning block)
            artefactwindow (optional) - window around the stimulation to be removed ( default = [0.005*fs, 0.020*fs] )
            duration (optional) - duration of the LED pulse (and until the off-artifact; default = 0.200 s)
        outputs:
            HC_epoch_data - a list with length = number of 15-minute epochs, each list item is the HPC data with the artifacts removed
            HC_epoch_time - a list of the same size as HC_epoch_data with the time in NlX time
            """

    #use the default artefact window and duration until the 'off-artifact' if no values were given
    artefactwindow= kwargs['artefactwindow'] if 'artefactwindow' in kwargs.keys() else [-0.002, 0.005] 
    duration = kwargs['duration'] if 'duration ' in kwargs.keys() else 0.200 #remove another artifact 200 ms after the first stim artifact)
    
    allstim = np.hstack(( np.hstack(stimpertrial), np.hstack(stimpertrial)+duration )) if duration>0 else np.hstack(stimpertrial)
                                        
    for TRIAL in range(len(HC_epoch_data)):
        HC_epoch_data[TRIAL] = signalscore.remove_artefacts(HC_epoch_data[TRIAL], allstim, time=HC_epoch_time[TRIAL], axis=-1, window=artefactwindow, interp='linear')
        
    return HC_epoch_data, HC_epoch_time

def artefact_removal_new(HC_epoch_data,HC_epoch_time,fs,threshold_type='raw',amplitude_threshold=50,frequency_threshold=500,):
    
    """ 
        function to detect and remove artefacts based on high frequency threshold crossings
        
        inputs:
            HC_epoch_data - a list with length N channels, each list is the full data of the session (either on-maze or ITI)
            HC_epoch_time - a time vector with length is equal to each of the HC_epoch_time list items
            fs - sampling frequency
            
            frequency_threshold (optional) -default=500 #filter above this frequency band to get the artefacts
            amplitude_threshold (optional) -default=50 #threshold above this amplitude to determine the threshold crossings
            -- should we add another way to set the threshold? based on the median of the signal or something?
            
        outputs:
            HC_epoch_data - the list with data per channel where the artefacts have been removed and interpolated
            HC_epoch_time - unchanged
            """
    
    for CH in range(len(HC_epoch_data)): #do this separately for each channel
    
        filtered = fklabfilter.apply_filter(HC_epoch_data[CH],[frequency_threshold,1000],fs=fs) #filter in high frequency range to find the artefacts
        envelope = fklabfilter.compute_envelope(filtered, fs=fs, isfiltered=True) #take the absolute values as the artefact can either be a negative or a positive deviation
        
        if threshold_type=='raw':
            amplitude_threshold = amplitude_threshold
        else:
            amplitude_threshold = fklab.signals.core.compute_threshold(HC_epoch_time,envelope,threshold=amplitude_threshold,kind=threshold_type)
        
        #get the segments of threshold crossings
        #tempsegment = fklab.segments.Segment.fromlogical(abs(filtered)>amplitude_threshold) #,x=HC_epoch_time)
        #tempsegment = tempsegment.join(gap=5) #0.002)
        #for SEG in range(len(tempsegment)):
        #    HC_epoch_data[CH][int(tempsegment[SEG].start):int(tempsegment[SEG].stop+1)] = np.nan
        
        #don't segment, but just remove all separate threshold crossings
        HC_epoch_data[CH][np.flatnonzero(envelope>amplitude_threshold)]=np.nan
        
        #interpolate all nans
        HC_epoch_data[CH] = np.asarray( pd.Series(HC_epoch_data[CH]).interpolate())
        
    return HC_epoch_data, HC_epoch_time
   
def artefact_subtraction(HC_epoch_data,HC_epoch_time,stimpertrial,fs,artefact,window):

    stimpertrial = np.hstack(stimpertrial)
    for S in range(len(np.hstack(stimpertrial))):
        stim=stimpertrial[S]
        
        findstim = np.argmin(abs(HC_epoch_time-stim))
        
        if int(findstim+window[0]*fs)<len(HC_epoch_time) and int(findstim+window[1]*fs)<len(HC_epoch_time): #don't take into account artefacts that happen all the way at the end of the signal
            
            findstimwindow = slice( int(findstim+window[0]*fs), int(findstim+window[1]*fs) )
            
            for CH in range(len(HC_epoch_data)):
                HC_epoch_data[CH][findstimwindow] -= artefact[S]
        
    return HC_epoch_data, HC_epoch_time

def extract_data(data,trials):
    
    """ inputs:
            data - One HPC channel opened with nlx.NlxOpen()
            trials - segments of the start and end of each 15-minute epoch
        outputs:
            HC_epoch_data - a list with length = number of 15-minute epochs, each list item is the HPC data with the artifacts removed
            HC_epoch_time - a list of the same size as HC_epoch_data with the time in NlX time
            """
            
    HC_epoch_data=[]
    
    for CH in range(len(data)):
        HC_epoch_data.append( np.hstack( [ data[CH].readdata(start=trials.start[TRIAL],stop=trials.stop[TRIAL])[1] for TRIAL in range(len(trials)) ] ) )
    
    HC_epoch_time = np.hstack( [data[0].readdata(start=trials.start[TRIAL],stop=trials.stop[TRIAL])[0] for TRIAL in range(len(trials)) ])
        
    #interpolate any NaNs that may be present
    for CH in range(len(data)):
        HC_epoch_data[CH] = np.asarray( pd.Series(HC_epoch_data[CH]).interpolate())
            
    return(HC_epoch_data, HC_epoch_time)

def downsample(data,trials,**kwargs):
        
    """ in theory there should be a good way to determine how often to downsample the old fs to
    get to the new fs. But in practice the only downsampling that needs to be done is from 32000 to 4000"""
    #oldfs= kwargs['oldfs'] if 'oldfs' in kwargs.keys() else 32000
    #newfs = kwargs['newfs'] if 'newfs' in kwargs.keys() else 4000 
    
    HC_epoch_downsampled = [[] for N in range(len(data))]
    HC_time_downsampled = []
    
    for TRIAL in range(len(trials)):
        for CH in range(len(data)):
        
            HC_epoch_data =  data[CH].readdata(start=trials.start[TRIAL],stop=trials.stop[TRIAL])[1]
            HC_epoch_data = np.asarray( pd.Series(HC_epoch_data).interpolate() )
            
            #downsample by 4 (32000/4=8) and then by 2 (8000/2=4000)
            temp_downsampled = scipysignal.decimate(HC_epoch_data,4)
            temp_downsampled = scipysignal.decimate(temp_downsampled,2)
            HC_epoch_downsampled[CH].append( temp_downsampled )
            
        HC_epoch_time =  data[0].readdata(start=trials.start[TRIAL],stop=trials.stop[TRIAL])[0]
        HC_time_downsampled.append( HC_epoch_time[::8] )

    #interpolate any NaNs that may be present
    for CH in range(len(data)):
        HC_epoch_downsampled[CH] = np.hstack(HC_epoch_downsampled[CH])
    HC_time_downsampled = np.hstack(HC_time_downsampled)
        
    return (HC_epoch_downsampled, HC_time_downsampled)

def extract_artefacts( data,time,stimpertrial,fs,window,envelope=False ):
    
    """code to extract the artefacts around a stimulation window"""
    
    if envelope==True:
        filtered = [ fklabfilter.apply_filter( data[CH], [160,225],fs=fs) for CH in range(len(data)) ]
        envelope = [ fklabfilter.compute_envelope(filtered[CH], fs=fs, isfiltered=True) for CH in range(len(data)) ]
        avgsignal = np.mean(envelope, axis=0)
    else:
        avgsignal = np.mean(data,axis=0) #in the case of cortex data, this is just one channel, so nothing changes
        
    artefacts=[]
    times=[] #=basically the same as stimpertrial, except for the ones that happen at the borders
    
    stimpertrial = np.hstack(stimpertrial)
    for stim in np.hstack(stimpertrial):
        findstim = np.argmin(abs(time-stim))
        
        if int(findstim+window[0]*fs)<len(avgsignal) and int(findstim+window[1]*fs)<len(avgsignal): #don't take into account artefacts that happen all the way at the end of the signal
            
            findstimwindow = slice( int(findstim+window[0]*fs), int(findstim+window[1]*fs) )
            artefact = avgsignal[findstimwindow]
            baseline = np.mean( np.hstack(( artefact[0:50],artefact[-50:] )) )
            
            artefacts.append( artefact - baseline )
            times.append( stim )
    #artefact = np.mean(artefacts,axis=0)
    
    return artefacts, times  

def findslice(HC_epoch_time,folder,trials,string='stimulation',fs=4000,window=[-1,1]):
    
    slices = []
    window = np.hstack(window)*fs
    
    events = nlx.NlxOpen(folder+'/Events.nev').data[:]
    if string=='stimulation':
        stimulation = events.time[events.eventstring==b'stimulation']
    elif string=='detection':
        stimulation = events.time[events.eventstring==b'detection']
        
    stimulation = np.delete( stimulation, np.flatnonzero(np.diff(stimulation)<0.2)+1 ) #somehow, all stimulations are being counted double - delete everything that is less than 200 ms apart
    stimpertrial= [ stimulation[ np.flatnonzero( trials[TRIAL].contains(stimulation)[0] )] for TRIAL in range(len(trials)) ]

    allstim = np.hstack(stimpertrial)
    for stim in allstim:
        findstim = np.argmin(abs(stim-HC_epoch_time))
            
        if int(findstim+window[0])>0 and int(findstim+window[1])<len(HC_epoch_time):
            slices.append( [int(findstim+window[0]),int(findstim+window[1])] )
            
    return slices

def remove_large_artefacts(HC_epoch_data,HC_epoch_time,fs,**kwargs):
    
    high_threshold = kwargs['high_threshold'] if 'high_threshold' in kwargs.keys() else 100 #3000
    
    artefacts=[]
    for TRIAL in range(len(HC_epoch_data)):
        
        filtered = fklabfilter.apply_filter(HC_epoch_data[TRIAL],[300,1000],fs=fs)
        nans_trial = fklab.segments.Segment.fromlogical( abs(filtered) > high_threshold )
        nans_trial = nans_trial.ijoin(5)
        
        #nans_trial = fklab.signals.core.detect_mountains(
        #    abs(HC_epoch_data[TRIAL]), np.arange(len(HC_epoch_data[TRIAL])), low=abs(np.nanmedian(HC_epoch_data[TRIAL])*100), high=high_threshold, 
        #    allowable_gap = 0.02, minimum_duration=0.02) 
        
        #maximum 1 second at a time (if a signal is above the threshold for much longer that that, it is likely not an artefact)
        if len(nans_trial)>0:
            print('Run '+str(TRIAL)+': '+str(round(np.sum(nans_trial.duration)/fs,2))+' s removed')
            #save the timestamps of the removed artefacts
            artefacts.append( fklab.segments.Segment(HC_epoch_time[TRIAL][np.vstack(nans_trial).astype(int)]) )
            
        #delete the artefacts
        for S in range(len(nans_trial)):
            nansegment = np.arange( int(nans_trial[S].start),int(nans_trial[S].stop)+1 )
            HC_epoch_data[TRIAL][nansegment] = np.nan

        #and then interpolate the NaNs
        HC_epoch_data[TRIAL] = np.asarray( pd.Series(HC_epoch_data[TRIAL]).interpolate())  

    return(HC_epoch_data, HC_epoch_time, artefacts)

def ripple_detection_onethreshold(HC_epoch_time,HC_epoch_data,trials,fs,duration_threshold,threshold,**kwargs):

    maxamp=100 #set a maximum amplitude for the ripples (above this is probably noise)

    (ripple_peak_time, ripple_peak_amp), ripple_segments, (threshlow,threshigh) = ripple.detect_ripples(HC_epoch_time,HC_epoch_data,isenvelope=False,
                                                                isfiltered=False, segments=None,threshold=threshold,  #[0.5,7]
                                                                threshold_kind='median mad', allowable_gap=0.02, minimum_duration=duration_threshold[0])
    
    """ get only one max amplitude per ripple, so that the ripple amp length matches the ripple_segment length """
    selection = [ np.argmax( ripple_peak_amp[ ripple_segments[RIPPLE].contains(ripple_peak_time)[0] ] ) + ripple_segments[RIPPLE].contains(ripple_peak_time)[2][0,0] for RIPPLE in range(len(ripple_segments)) ]
    ripple_peak_time = ripple_peak_time[selection]
    ripple_peak_amp = ripple_peak_amp[selection]
    
    #remove too long and too high amp ripples (=noise)
    keepindextime = np.flatnonzero( ripple_segments.duration<duration_threshold[1] )
    keepindexamp = np.flatnonzero( ripple_peak_amp<= maxamp )
    
    keepindex = np.intersect1d( np.hstack(keepindexamp),keepindextime )
    ripple_segments = ripple_segments[keepindex]
    
    #detect ripples in the cortex as well and discard any overlapping ones
    if len(kwargs)>0:
        if 'cortex_time' in kwargs.keys() and 'cortex_data' in kwargs.keys():
            CTX_epoch_time = kwargs['cortex_time']
            CTX_epoch_data = kwargs['cortex_data']
            cortex_threshold = kwargs['cortex_threshold']
            
            #filtered = [fklabfilter.apply_filter(CTX_epoch_data[CH],[160,225],fs=fs) for CH in range(len(CTX_epoch_data)) ]
                
            (time, amp), ripple_segments_cortex, (_,_) = ripple.detect_ripples(CTX_epoch_time,CTX_epoch_data,isenvelope=False,
                                                                        isfiltered=True, segments=trials,threshold=cortex_threshold,
                                                                        threshold_kind='median mad', allowable_gap=0.02, minimum_duration=duration_threshold, filter_options={}, smooth_options={}) 
            
            removeripples = [ np.flatnonzero(ripple_segments_cortex[N].contains(ripple_segments.center)[0]) for N in range(len(ripple_segments_cortex)) ]
            if len(removeripples)>0:
                keepripples = np.delete(np.arange(len(ripple_segments)), np.hstack(removeripples) )
                ripple_segments = ripple_segments[keepripples]

    return ripple_segments

def ripple_detection(HC_epoch_time,HC_epoch_data,trials,fs,duration_threshold,threshold):

    ripple_segments=[]
    
    for TRIAL in range(len(trials)):
        
        startrow=np.argmin(abs(HC_epoch_time-trials[TRIAL].start))
        endrow=np.argmin(abs(HC_epoch_time-trials[TRIAL].stop))
        
        temptime = HC_epoch_time[startrow:endrow]
        tempdata=[ HC_epoch_data[CH][startrow:endrow] for CH in range(len(HC_epoch_data)) ]
        
        (time, amp), temp_ripple_segments, (_,_) = ripple.detect_ripples(temptime,tempdata,isenvelope=False,
                                                                    isfiltered=False, segments=None,threshold=threshold,  #[0.5,7]
                                                                    threshold_kind='median mad', allowable_gap=0.02, minimum_duration=duration_threshold, filter_options={}, smooth_options={}) 
        ripple_segments.append(temp_ripple_segments)
        
    ripple_segments = fklab.segments.Segment( np.vstack(ripple_segments) )

    
    return ripple_segments

def contrast_frequency_bands(
    signal, fs, target = [160,225], contrast = [[100,140],[250,400]],
    weights = None, kind='power', transition_width="10%", smooth=0.01
):
    
    if not isinstance(signal, (list, tuple)):
        signal = [signal]
    
    signal = [y[:,None] if y.ndim==1 else y for y in signal]
    
    # filter signal
    y = [
        np.dstack(
            [
                fklabfilter.apply_filter(
                    x, band, fs=fs, axis=0,
                    transition_width=transition_width
                )
                for band in [target] + contrast
            ]
        )
        for x in signal
    ]
        
    if kind=='power':
        y = [k**2 for k in y]
    elif kind=='power x frequency':
        fcenter = np.array([(a+b)/2 for a,b in [target] + contrast])[None,None,:]
        y = [(k**2) * fcenter for k in y]
    elif kind=='envelope':
        y = [np.abs(scipysignal.hilbert(k, axis=0)) for k in y]
    else:
        raise ValueError("Unknown value for kind argument.")
    
    if weights is None:
        weights = np.ones(len(contrast))/len(contrast)
    
    y = [k[:,:,0] - np.average(k[:,:,1:], axis=2, weights=weights) for k in y]
    
    if not smooth is None:
        y = [fklab.signals.smooth.smooth1d(k, axis=0, delta=1.0 / fs, bandwidth=smooth) for k in y]
    
    contrast_signal = np.mean(np.concatenate(y, axis=1), axis=1)
    
    return contrast_signal
        
def ripple_detection_contrast(HC_epoch_time,HC_epoch_data,trials,fs,duration_threshold,threshold):

        contrast_signal = [ contrast_frequency_bands(HC_epoch_data[TRIAL],fs=fs,kind='power') for TRIAL in range(len(trials))]
        
        contrast_threshold = fklab.signals.core.compute_threshold(
        np.concatenate(HC_epoch_time),np.concatenate(contrast_signal),threshold=threshold,kind='median mad')
        
        ripple_segments = [
        fklab.signals.core.detect_mountains(
            contrast_signal[TRIAL], HC_epoch_time[TRIAL], low=contrast_threshold[0], high=contrast_threshold[1], 
            allowable_gap = 0.02, minimum_duration=duration_threshold) for TRIAL in range(len(trials))]
        
        ripple_segments = np.vstack(ripple_segments)  
        
        return ripple_segments

def get_areas(maze,xypos,**kwargs):  
    """
    inputs:
        - maze: the maze.yaml loaded from amaze
        - xypos: the x and y posisionts loaded from localize
        - size (optional): the sizes of the platforms. If not specified, the default
        values of 40 is used.'Amaze' can also be passed here to use the sizes of the amaze shapes.
    outputs:
        - Areas: a vector the same length as xypos, with the maze area corresponding to each tracked position
    """
        
    Mazeareas = ['P1','P2','P3','Center']
    centers = [maze["maze"]["shapes"][A]["shape"].center for A in Mazeareas]

    #if the size is of the platforms is not given, use the default size of 40
    if 'size' not in kwargs.keys():
        sizes = [[40] for A in Mazeareas] #use a fixed size so that it's the same for all sessions
    elif kwargs['size'] == 'Amaze': #if specified, use the sizes from amaze
        sizes = [np.min(maze["maze"]["shapes"][A]["shape"].size) for A in Mazeareas]
    else: #otherwise, size needs to be a number
        sizes = [[kwargs.values()] for A in Mazeareas] 
        
    densitymap = 50
    linranges = [[ARM] * densitymap for ARM in range(1,4)]
    linranges = np.hstack(linranges)
    xyranges = np.zeros((densitymap*3,2))
    for ARM in range(3):
        PlatformCenter = maze['maze']['shapes']['P'+str(ARM+1)]['shape'].center
        Centercenter = maze['maze']['shapes']['Center']['shape'].center
        #get N equally spaced points between the platform and the center to map the tracked positions onto
        xyranges[ARM*densitymap:(ARM+1)*densitymap,0] = np.linspace(PlatformCenter[0],Centercenter[0],num=densitymap,endpoint=False) # the x range between platform and center
        xyranges[ARM*densitymap:(ARM+1)*densitymap,1] = np.linspace(PlatformCenter[1],Centercenter[1],num=densitymap,endpoint=False) # the y range between platform and center

    Areas = [[np.nan] for N in range(len(xypos))]
    for POS in range(len(xypos)):
        if ~np.isnan(xypos[POS,0]):
            distance = [ np.sqrt( (xypos[POS,0]-centers[A][0])**2 +  (xypos[POS,1]-centers[A][1])**2 ) for A in range(len(Mazeareas)) ]
            foundarea=np.flatnonzero(np.vstack(distance)<np.vstack(sizes))
            if len(foundarea)==1: #position is found in one of the platforms (P1,P2,P3,P4 or center)
                Areas[POS]=Mazeareas[int(foundarea)]
            else: #closest position to one of the arms
                #find the closest point to the tracked position in the xyranges variable
                closestpoint = np.argmin( np.sqrt((xypos[POS,0]-xyranges[:,0])**2 + (xypos[POS,1]-xyranges[:,1])**2) )
                armnr=linranges[closestpoint]
                Areas[POS]='A'+str(armnr)
                
    Areas = np.vstack(Areas) 
    
    return Areas

def get_linearpos(maze,xypos,collapse=False, armlength=50):
    
    #get the different nodes from the yaml file
    P1=maze["maze"]["shapes"]["P1"]["shape"].center
    P2=maze["maze"]["shapes"]["P2"]["shape"].center
    P3=maze["maze"]["shapes"]["P3"]["shape"].center
    C=maze["maze"]["shapes"]["Center"]["shape"].center
    
    #construct maze
    nodes = np.vstack(( P1,C,P2,P3 ))
    maze = fklab.geometry.shapes.graph(
        [fklab.geometry.shapes.polyline( [P1,C], forced_length=armlength),
         fklab.geometry.shapes.polyline( [P2,C], forced_length=armlength),
         fklab.geometry.shapes.polyline( [P3,C], forced_length=armlength)],
        nodes = nodes)

    linear_pos = maze.point2path(xypos)[0]
    
    if collapse==True: #treat all arms as equal (every position just indicates the distance between platform and center, no difference which arm it is)
        linear_pos[np.flatnonzero( linear_pos >= armlength )] -= armlength
        linear_pos[np.flatnonzero( linear_pos >= armlength )] -= armlength
        
    return linear_pos