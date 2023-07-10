#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 28 20:36:19 2022

@author: hanna
"""

# import libraries:
import numpy as np
import matplotlib.pyplot as plt
import fklab.io.neuralynx as nlx
#from pathlib import Path
import fklab.geometry.shapes
import fklab.segments 
import h5py
import utilities
import fklab.signals.filter.filter as fklabfilter
#import pickle
import pandas as pd
#import matplotlib.lines as mlines
import fklab.signals.multitaper as mt
#from glob import glob
import fklab.utilities.yaml as yaml 

#%% parameters

for TRIAL in range(5,6):
    
    animal = 'B21'
    SESSION=6
    #TRIAL=5
    indextype='ITI' #'Onmaze' or 'ITI'
    
    fs = 4000 #sampling rate of the saved data
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)
    date = params['Alldates'][SESSION]
    folder = params['topfolder']+animal+'/'+date+'/'
    save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+params['Alldates'][SESSION]+'/'+indextype+'/'
    
    #load trial times
    with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
        trials = fklab.segments.Segment( h5f['trials'][:] )
    
    #extract the data
    channames = params['channames_ripples'][SESSION]
    HC_channel = nlx.NlxOpen(folder+'CSC'+str(channames[0])+'.ncs')
    HC_epoch_data = [ HC_channel.readdata(start=trials.start[TRIAL],stop=trials.stop[TRIAL])[1] for TRIAL in range(len(trials)) ] 
    HC_epoch_time = [ HC_channel.readdata(start=trials.start[TRIAL],stop=trials.stop[TRIAL])[0] for TRIAL in range(len(trials)) ]     
    
    #interpolate any NaNs that may be present
    HC_epoch_data[TRIAL] = np.asarray( pd.Series(HC_epoch_data[TRIAL]).interpolate())
            
    with open(folder+'/epochs/maze.yaml') as file:
        maze = yaml.unsafe_load(file)
    
    pixel2cm = maze['maze']['shapes']['track']['shape'].pathlength / 290 #290 cm = real total path length
    
    runfolder = folder+'/epochs/run'+str(TRIAL+1) if indextype=='Onmaze' else folder+'/epochs/rest'+str(TRIAL+1)
    with h5py.File(runfolder+'/position.hdf5', "r") as behavior:
        behavior_time=behavior['time'][:]
        xypos = behavior['position'][:]
        behavior_velocity = abs(behavior['velocity'][:]) / pixel2cm
    
    areas = utilities.get_areas(maze,xypos,size='Amaze') 
    findplatform = np.hstack(['P' in A for A in np.hstack(areas)])
    platform_segments=fklab.segments.Segment.fromlogical(findplatform,behavior_time)
    
    # load detections and stimulations
    events = nlx.NlxOpen(folder+'Events.nev').data[:]
    detect_t = events.time[events.eventstring==b'detection']
    stim_t = events.time[events.eventstring==b'stimulation']
    
    #seperate the detections per learning or rest block
    stimpertrial = [ stim_t[ np.flatnonzero( trials[TRIAL].contains(stim_t)[0] )] for TRIAL in range(len(trials)) ]
    detecpertrial = [ detect_t[ np.flatnonzero( trials[TRIAL].contains(detect_t)[0] )] for TRIAL in range(len(trials)) ]
    
    #artefact removal
    s=params['Stimulation'][SESSION]
    if s=='Delayed' or s=='On-time':
        HC_epoch_data, HC_epoch_time = utilities.artefact_removal(HC_epoch_data,HC_epoch_time,stimpertrial,fs)
    
    # Signal transformations
    filtered = fklabfilter.apply_filter( HC_epoch_data[TRIAL],[160,225],fs=fs)
    
    S_HC,t,f,Serr,options=mt.mtspectrogram(data=HC_epoch_data[TRIAL], fs=fs, window_size=0.2, window_overlap=0.5) 
    S_HC = 10.0 * np.log10(S_HC) #transform to dB
    
    #% 
    #xlimsfind = np.hstack((np.vstack(np.arange(195,295,5)),np.vstack(np.arange(205,305,5))))
    xlimsfind = np.vstack(( np.arange(500,600,10), np.arange(510,610,10) )).T #[5,15]
    xlimsfind=[[170,180]]
    for PLOT in range(len(xlimsfind)):
        
        fig, ax = plt.subplots(3,1,figsize=(6.24,2.145), dpi=400)
        plt.suptitle(animal+' session: '+str(SESSION)+', run: '+str(TRIAL)+' '+indextype+' stim: '+s)
        
        ax[0].plot(behavior_time,behavior_velocity,color='k',lw=0.5)
        if indextype=='Onmaze':
            for P in range(len(platform_segments)):
                ax[0].axvspan(platform_segments.start[P],platform_segments.stop[P],color='green',alpha=0.2)
        ax[0].set_ylim(-10,100)
        
        ax[1].plot(HC_epoch_time[TRIAL],filtered,color='k',lw=0.3)
        for S in stimpertrial[TRIAL]:
            ax[1].axvspan( S, S+0.2, color='orange', alpha=0.3 )
            
        for D in detecpertrial[TRIAL]:
            ax[1].plot( D, 90,color='red', marker='o',markersize=1 )
        ax[1].set_ylim(-100,100)
        
        im2 = ax[2].imshow(S_HC.T, extent=[t[0][0],t[-1][-1],f[0],f[-1]], aspect='auto',origin='lower',cmap='viridis',vmin=-10,vmax=5)
        ax[2].set_ylim([100,250])
        #plt.colorbar(im2, ax=ax[2])
        #
        """    """ 
        ax[0].set_xlim(np.array(xlimsfind[PLOT]) + behavior_time[0])
        ax[1].set_xlim(np.array(xlimsfind[PLOT]) + HC_epoch_time[TRIAL][0])
        ax[2].set_xlim( np.array(xlimsfind[PLOT]) - 0.35 ) 
        
        ax[0].set_xticklabels([])
        ax[1].set_xticklabels([])
        #ax[2].set_xticklabels([])
        
        ax[0].set_yticks([0,25,50,75])
        ax[1].set_yticks([-50,0,50])
        ax[2].set_yticks([100,150,200,250])
        
        plt.show()



#%% plot again to show the colorbar

plt.imshow(S_HC.T, extent=[t[0,0],t[-1,1],f[0],f[-1]], aspect='auto',origin='lower',cmap='viridis',vmin=-10,vmax=5)
plt.ylim([100,250])
plt.xlim(np.array([195,203]))
plt.colorbar(ticks=[-10,-5,0,5])

