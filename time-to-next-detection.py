#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 23 09:49:36 2022

@author: hanna
"""

import numpy as np

import fklab.io.neuralynx as nlx
import fklab.segments 
import matplotlib.pyplot as plt
from glob import glob
import h5py
import utilities

plt.style.use(['seaborn-talk'])
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

#%% Loop through sessions

""" -> Do SWR-triggered inhibitions lead to a change in SWR occurrences?

Look at the time from one detection to the next, and compare between 
the first 15 minutes and the last 15 minutes of a session
for no-stim, on-time and delayed
    
"""
animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] 
stimtypes =['None','On-time','Delayed']
indextype = 'Onmaze'
division = ['firstblock','lastblock']
divisionindx = [0,-1]

histrange = [0,2] #range of the histograms in seconds
histbinnum = 20 #number of bins in the histogram

#initialize variables
for s in stimtypes:
    for d in division:
        vars()[s+'_'+d+'_histogram']=[]
    
for animal in animals:
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)
    
    for s in stimtypes:
        
        findsession = np.flatnonzero(np.hstack(params['Stimulation'])==s)
        if len(findsession)>0:
            SESSION = findsession[0]
            
            date = params['Alldates'][SESSION]
            folder = params['topfolder']+animal+'/'+date+'/'
            Nruns = len(glob(folder + 'epochs/run*'))
                
            save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
        
            #load trial times
            with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
                trials = fklab.segments.Segment( h5f['trials'][:] )
                
            # load detections and stimulations
            events = nlx.NlxOpen(folder+'Events.nev').data[:]
            detect_t = events.time[events.eventstring==b'detection']
            online_rippertrial = [ detect_t[ np.flatnonzero( trials[TRIAL].contains(detect_t)[0] )] for TRIAL in range(len(trials)) ]
    
            for D in range(len(division)): #first block versus last block
            
                session_histdata = np.zeros(histbinnum)
                
                tempdetections = online_rippertrial[divisionindx[D]]
                trialduration = trials[divisionindx[D]].duration
                for detection in tempdetections:
                    histdata,histbins = np.histogram(tempdetections-detection, bins=histbinnum, range=histrange)
                    session_histdata += histdata
                    
                vars()[s+'_'+division[D]+'_histogram'].append( session_histdata /len(tempdetections) )
                
#%% plot the histograms

colors= 'grey','darkorange','k'

fig,ax=plt.subplots(2,3,dpi=400)
plt.tight_layout()

for D in range(len(division)):

    for S in range(len(stimtypes)):
        
        histdata = np.vstack( vars()[stimtypes[S]+'_'+division[D]+'_histogram'] )
        tempmean = np.mean(histdata,axis=0) #[2:]
        tempsem = (np.std(histdata,axis=0)/np.sqrt(histdata.shape[0])) #[2:]
        
        ax[D,S].bar(histbins[0:-1],tempmean,yerr=tempsem,width=np.diff(histbins)[0],color='grey',edgecolor='k')
        ax[D,S].set_xticks(histbins[0:-1])
        ax[D,S].axvline(0,color=colors[S],linestyle='--')
        ax[D,S].set_ylim(0,0.2)
        ax[D,S].set_title(stimtypes[S]+', '+division[D])