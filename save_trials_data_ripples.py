#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue May  3 19:36:40 2022

@author: hanna
"""

# import libraries:
import numpy as np
from glob import glob
import fklab.io.neuralynx as nlx
import fklab.segments 
import fklab.utilities.yaml as yaml #import tools to parse YAML
import utilities
from pathlib import Path
import pickle
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.lines as mlines

#%% make a folder structure separate for results of the analyses

indextypes = ['Onmaze','ITI']
animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23']

for animal in animals:
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):
        
        for indextype in indextypes:
            save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+params['Alldates'][SESSION]+'/'+indextype+'/'
            # create destination folder for saving
            if not Path(save_folder).exists():
                Path(save_folder).mkdir(parents=True)
                
#%% save trial times

#redo:  - B9 ITI SESSION 5 and Onmaze session 5 - animal chewed through cable - neural data not usable
#B21 session 3 last rest period too noisy to use

printmessages = True
indextypes = ['Onmaze','ITI']
animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21']

for animal in animals:
    
    params = utilities.get_parameters(animal)

    for SESSION in range(len(params['Alldates'])):
        
        date = params['Alldates'][SESSION]
        folder = params['topfolder']+animal+'/'+date+'/'
        Nruns = len(glob(folder + 'epochs/run*'))
        
        for indextype in indextypes:
            save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'

            if indextype=='Onmaze':
                #For on maze, the right times are already indicated in the info.yaml file
                with open(folder+'info.yaml','r') as f:
                    info = yaml.safe_load(f)
                trials = [ info['epochs'][RUN]['time'] for RUN in range(Nruns) ]
                trials=np.vstack(trials)  
                
            elif indextype=='ITI':
                
                events = nlx.NlxOpen(folder+'Events.nev')
                eventstrings = events.data.eventstring[:]
                trials=[]
                
                if printmessages:
                    print(animal+', '+params['Alldates'][SESSION]+':')
                    
                if animal =='B21' and SESSION==3 and indextype=='ITI':
                    Nruns = 6 #B21 session 3 last rest period too noisy to use
                    
                for RUN in range(Nruns): 
                    startindex=np.argwhere(['start rest '+str(RUN+1) in str(eventstrings[I]) for I in range(len(eventstrings)) ])    
                    endindex = np.argwhere(['end rest '+str(RUN+1) in str(eventstrings[I]) for I in range(len(eventstrings)) ])
                    
                    if len(startindex)==0 and len(endindex)>0: #if there was no start index, take the end of the preceding run
                        startindex=np.argwhere(['end run '+str(RUN+1) in str(eventstrings[I]) for I in range(len(eventstrings)) ])
                    if len(endindex)==0 and len(startindex)>0: #if there was no endindex, take the start of the next run
                         endindex = np.argwhere(['start run '+str(RUN+2) in str(eventstrings[I]) for I in range(len(eventstrings)) ])
                    
                    if len(startindex)>0 or len(endindex)>0: #there has to be at least a start index or an endindex
                        #add a solution for when the length is more than one (then pick the one closer to start+15 or end-15)
                        if len(startindex)>1:
                            durations = [ (events.data.time[int(endindex)] - events.data.time[int(startindex[I])])/60 for I in range(len(startindex)) ]
                            rightduration = np.flatnonzero( [durations[I]>10 and durations[I]<20 for I in range(len(durations))] )
                            if len(rightduration)>1:
                                rightduration = int(np.argwhere( abs(15-np.hstack(durations))==min(abs(15-np.hstack(durations))) ) )
                                
                        else:
                            rightduration=0
                             
                        starttime = events.data.time[int(startindex[rightduration])] if len(startindex)>0 else events.data.time[int(endindex)] - 15*60
                        
                        if len(endindex)>1:
                            durations = [ (events.data.time[int(endindex[I])] - events.data.time[int(startindex)])/60 for I in range(len(endindex)) ]
                            rightduration = np.flatnonzero( [durations[I]>10 and durations[I]<20 for I in range(len(durations))] )[0]
                        else:
                            rightduration=0
                        
                        endtime = events.data.time[int(endindex[rightduration])] if len(endindex)>0 else events.data.time[int(startindex)] + 15*60
                        if endtime < starttime: #the end tim cannot be before the start time
                            endtime = starttime + 15*60
                            
                        if endtime > events.data.time[int(events.data.time.shape[0]-1)]: #the end time cannot be larger than the total duration of the recording
                            endtime = events.data.time[int(events.data.time.shape[0]-1)] 
                        
                        if (int(endtime)-int(starttime))/60 > 1:
                            trials.append( [int(starttime),int(endtime)] )
                            if printmessages:
                                print('Rest '+str(RUN+1)+': '+str((int(endtime)-int(starttime))/60)+' minutes') 
                        
            trials = np.vstack(trials)
             
            #one exception for B11 rest: run 3 session 3 didn't record properly and there are no stimulations
            if animal =='B11' and SESSION==3 and indextype=='ITI':
                indextrials = np.array([0, 1, 2, 4, 5, 6])
                trials = trials[indextrials,:]
            #exception B5: first trial on the maze the cable was twisted and the signal got too noisy and unusable
            if animal =='B5' and SESSION==3 and indextype=='Onmaze':
                trials = trials[1:,:]
                
            #save the trial times in the right folder
            with h5py.File(save_folder+'trials.hdf5', 'w') as h5f: 
                h5f.create_dataset('trials', data=trials)
                
#%% save sequences of c, l and r

indextype = 'Onmaze' #indextype is only Onmaze here
plotexamples = False #whether or not to plot some example trials per animal
SAVE = True #whether or not to save the trial sequences

animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21'] #
seqname = 'L','C','R'

for animal in animals:
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)
    plotcount = 0
    
    for SESSION in range(len(params['Alldates'])):
        
        date = params['Alldates'][SESSION]
        folder = params['topfolder']+animal+'/'+date+'/'
        save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'

        #load trials neuralynx time
        Nruns = len(glob(folder + 'epochs/run*'))

        sequences=[[] for R in range(Nruns)]
        
        with open(folder+'/epochs/maze.yaml') as file:
            maze = yaml.load(file)

        for RUN in range(Nruns):

            runfolder = glob(folder+'/epochs/*'+str(RUN+1))[0]
            with h5py.File(runfolder+'/position.hdf5', "r") as behavior:
                behavior_time=behavior['time'][:]
                xypos = behavior['position'][:]
                
            areas = utilities.get_areas(maze,xypos,size='Amaze') 
            
            centersegments = fklab.segments.Segment.fromlogical( np.hstack(areas)=='Center',behavior_time ).start #for the center we only want the start times
            platformsegments = [ fklab.segments.Segment.fromlogical( np.hstack(areas)=='P'+str(P),behavior_time ).start for P in range(1,4) ]
            seq = params['Sequences'][SESSION]
            trialtimes=[]
            for TRIAL in range(len(centersegments)):

                    findnextplatform =[np.nan,np.nan,np.nan]
                    for P in range(3):
                        if len( np.flatnonzero(platformsegments[P] > centersegments[TRIAL]))>0:
                            findnextplatform[P] = platformsegments[P][ np.flatnonzero(platformsegments[P] > centersegments[TRIAL])[0] ]
                    
                    if TRIAL==0 or np.nanmin(findnextplatform) > trialtimes[-1]:
                        trialtimes.append( np.nanmin(findnextplatform) )
                        
                        nextplatform = np.nanargmin(findnextplatform) + 1 #+1 to index the platforms from 1 instead of from 0
                        sequences[RUN].append( seqname[int(np.flatnonzero(seq==nextplatform))] )
                        
                        #if there is a repeat (e.g. C to C), check if the animal really went all the way to the center and back again (and it's not a jump)
                        if TRIAL>1 and sequences[RUN][-1] == sequences[RUN][-2]:
                            lasttrial = fklab.segments.Segment( np.hstack(( trialtimes[-2], trialtimes[-1])) )
                            areaslasttrial = np.unique(areas[np.flatnonzero(lasttrial.contains(behavior_time)[0])])
                            if len(np.flatnonzero(['A' in A[0] for A in areaslasttrial]))<1: #did he visit the arm as well? (otherwise it must be a jump)
                                del sequences[RUN][-1]
                                del trialtimes[-1]
                                
            if plotexamples and plotcount<10:
                Mazeareas = ['P1','Center','P2','Center','P3']
                centers = np.vstack([maze["maze"]["shapes"][A]["shape"].center for A in Mazeareas])
                for TRIAL in range(50): #len(trialtimes)-1):
                    plt.figure(dpi=400)
                    plt.title('Trial '+str(TRIAL)+', '+sequences[RUN][TRIAL]+' to '+sequences[RUN][TRIAL+1])
                    startrow = int(np.flatnonzero(behavior_time==trialtimes[TRIAL]))
                    endrow =  int(np.flatnonzero(behavior_time==trialtimes[TRIAL+1]))
                    plt.plot(xypos[startrow:endrow,0],xypos[startrow:endrow,1],marker=None,color='k',alpha=1) 
                    for PLOT in range(len(centers)-1):
                        plt.plot(np.linspace(centers[PLOT,0],centers[PLOT+1,0],num=10),np.linspace(centers[PLOT,1],centers[PLOT+1,1],num=10),'r')
                    plt.show()
        
        if SAVE:
            #save the sequences to a csv file
            maxlength = np.max([ len(sequences[R]) for R in range(Nruns)])
            allsequences=[[None] * maxlength for R in range(Nruns)]
            for R in range(Nruns):
                allsequences[R][0:len(np.hstack(sequences[R]))] = np.hstack(sequences[R])
                
            columnames = [['Run'+str(R)] for R in range(1,Nruns+1)]
            df = pd.DataFrame(np.vstack(allsequences).T,columns=np.hstack(columnames))
            
            df.to_csv(save_folder+'sequences.csv')
    print(animal+" done")

#%% save data
""" loop through all animals and
for sessions without stimulation: extract the data per run (15 minute epoch)
for sessions with stimulation: extract the data per run, and remove stimulation artefacts
for sessions with a higher sampling rate, downsample to 4000
for all sessions: remove large artifacts and interpolate any NaNs

output: HC_epoch_data and HC_epoch_time , to be saved in the /Analysis/ folder
"""

downsample = True
artefactremoval =False
largeartefactremoval=False

animals = ['B23'] #['B5','B9','B10','B11','B12','B17','B20','B21','B23']
#animals = ['B14','B18'] #higher fs - kernel dies
indextypes = ['Onmaze'] #['Onmaze','ITI'] #'Onmaze' or 'ITI'
region = 'CTX' #'CTX' or 'HC' - extract data from cortex and/or hippocampus

for indextype in indextypes:

    print('extracting data '+indextype)
    
    for animal in animals:
    
        params = utilities.get_parameters(animal)
        
        for SESSION in range(1): #len(params['Alldates'])):
            
            date = params['Alldates'][SESSION]
            folder = params['topfolder']+animal+'/'+date+'/'
            save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
            
            #load trial times
            with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
                trials = fklab.segments.Segment( h5f['trials'][:] )
            
            #open the data
            channames = params['channames_ripples'][SESSION] if region=="HC" else params['channames_cortex'][SESSION] 
            HC_channel = [ nlx.NlxOpen(folder+'CSC'+str(CH)+'.ncs') for CH in channames ]
            
            #extract the data and already interpolate any NaNs
            HC_epoch_data,HC_epoch_time = utilities.extract_data(HC_channel,trials)
            
            #downsample if necessary
            fs = HC_channel[0].header['SamplingFrequency']
            if downsample and fs == 32000:
                HC_epoch_data,HC_epoch_time = utilities.downsample(HC_epoch_data,HC_epoch_time)
                fs=4000 #the new sampling rate would be 4000
                
            #remove stimulation artefacts
            if artefactremoval and (params['Stimulation'][SESSION]=='On-time' or params['Stimulation'][SESSION]=='Delayed'): #there are no stimulation artifacts in 'None' sessions
    
                """Load online detections"""
                events = nlx.NlxOpen(folder+'/Events.nev')
                eventstrings = events.data.eventstring[:]
                stimulation = events.data.time[ [eventstrings[I]==b'stimulation' for I in range(len(eventstrings))] ]
    
                HC_epoch_data,HC_epoch_time = utilities.artefact_removal(HC_epoch_data,HC_epoch_time,trials,stimulation,fs=fs)
            
            #remove any large artefacts found in the data (not due to the stimulation but to movement or something else)
            #problem with the artefact removal: when a signal has a high amplitude throughout, the signal is seen as an artefact for very large periods at a time
            #it would be better to look at sudden increases/decreases in signal
            if largeartefactremoval and region=='HC':
                HC_epoch_data,HC_epoch_time,artefacts = utilities.remove_large_artefacts(HC_epoch_data,HC_epoch_time,fs=fs)
                      
                #save an hdf5 file with the times of the removed artefacts per run  
                if len(artefacts)>0:
                    artefacts = np.vstack(artefacts)
                with h5py.File(save_folder+'large_artefacts.hdf5', 'w') as h5f: 
                    h5f.create_dataset('artefacts', data=artefacts)
                
            #save data   
            trial_data = open(save_folder+region+"_epoch_data.pkl", "wb")
            pickle.dump(HC_epoch_data, trial_data)
            trial_data.close()
       
            trial_data = open(save_folder+region+"_epoch_time.pkl", "wb")
            pickle.dump(HC_epoch_time, trial_data)
            trial_data.close()
    
            print(animal+" "+str(SESSION)+" done")

     
#%% save ripples

"""
Then loop through the animals and sessions again and do the offline ripple detections of 
the cleaned signal
"""

downsample = True
artefactremoval = False
largeartefactremoval=False

animals = ['B5','B9','B10','B11','B12','B17','B20','B21','B23']
#animals = ['B14','B18'] #higher fs - kernel dies
indextypes = ['Onmaze','ITI'] #'Onmaze' or 'ITI'

duration_threshold=[0.03,1] #also give a maxiumum duration (a "ripple" of more than 2 seconds is noise)

for indextype in indextypes:
    print('Detecting ripples '+indextype)
    
    for animal in animals:
        
        params = utilities.get_parameters(animal)
        fs = 4000 if downsample else params['fs'] #if no downsampling was done, take the original sampling rate

        for SESSION in range(1): #len(params['Alldates'])):
    
            date = params['Alldates'][SESSION]
            folder = params['topfolder']+animal+'/'+date+'/'
            save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
    
            #load trial times
            with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
                trials = fklab.segments.Segment( h5f['trials'][:] )
                
            #load the saved data (artefacts already removed)
            with open(save_folder+"HC_epoch_data.pkl", "rb") as pckl:
                HC_epoch_data=pickle.load(pckl)
            with open(save_folder+"HC_epoch_time.pkl", "rb") as pckl:
                HC_epoch_time=pickle.load(pckl)
    
            threshold = [0.5,params['Threshold'][SESSION]] #lower threshold is always constant for now, upper is different per animal (7 or 8)

            ripple_segments = utilities.ripple_detection_onethreshold(HC_epoch_time,HC_epoch_data[0],trials,fs,
                                                                      duration_threshold,threshold=threshold)
            
            #print the ripple rate
            ripplerate = len(ripple_segments) / np.sum( trials.duration )
            print(animal+' '+str(SESSION)+' ripple rate: '+str(round(ripplerate,2))+' ripples/s')
            
            #save ripples    
            with h5py.File(save_folder+'ripple_segments.hdf5', 'w') as h5f: 
                h5f.create_dataset('ripple_segments', data=ripple_segments) 
                
#%% for each session, get the true positive and false discovery rate
    
animals = ['B5','B9','B10','B11','B12','B17','B20','B21'] #'B14','B18',
indextype='Onmaze' #Onmaze or ITI

#choose which things to plot
plotpersession=False #one plot with the different runs for each session
plotperanimal=True #one plot per animal
plottotal=True #one plot with all the session averages for all animals
plotexamples=False #examples of detections

#the parameters for plotting the examples
import fklab.signals.filter.filter as fklabfilter
plotpos=0
plotneg=0
window = [-0.5, 0.5]
filteringranges = [[10,250],
                   [160,225],
                   [300,1000]]

#for plotting the animal averages and total averages
stim = ['None','Delayed','On-time']
colors = ['k','purple','red']
for s in stim:
    for animal in animals:
        vars()['TPR_'+s+'_'+animal]=[]
        vars()['FDR_'+s+'_'+animal]=[]
    
for animal in animals:
    
    params = utilities.get_parameters(animal)
    fs = 4000 if downsample else params['fs'] #if no downsampling was done, take the original sampling rate
    
    for SESSION in range(len(params['Alldates'])):
        
            truepos=[]
            falsdisc=[]
            
            date = params['Alldates'][SESSION]
            folder = params['topfolder']+animal+'/'+date+'/'
            save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
    
            #load trial times
            with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
                trials = fklab.segments.Segment( h5f['trials'][:] )

            #load the saved data (artefacts already removed)
            with open(save_folder+"HC_epoch_data.pkl", "rb") as pckl:
                HC_epoch_data=pickle.load(pckl)
            with open(save_folder+"HC_epoch_time.pkl", "rb") as pckl:
                HC_epoch_time=pickle.load(pckl)
                        
            """ load the online detected ripples """
            events = nlx.NlxOpen(folder+'/Events.nev')
            eventstrings = events.data.eventstring[:]
            detection = events.data.time[ [eventstrings[I]==b'detection' for I in range(len(eventstrings))] ]
            stimpertrial= [ detection[ np.flatnonzero( trials[TRIAL].contains(detection)[0] )] for TRIAL in range(len(trials)) ]
            #don't count the online detections that happened during the large artefacts (cause that data is not included in the offline detection)
            if largeartefactremoval and Path(save_folder+'large_artefacts.hdf5').exists():
                with h5py.File(save_folder+'large_artefacts.hdf5', 'r') as h5f: 
                    artefacts = fklab.segments.Segment( h5f['artefacts'][:] )
                for TRIAL in range(len(stimpertrial)):
                    keepstims = np.flatnonzero([ artefacts.contains(STIM)[0]==False for STIM in stimpertrial[TRIAL] ])
                    stimpertrial[TRIAL]=stimpertrial[TRIAL][keepstims]
                    
            """ load the offline detected ripples"""
            #load offline detected ripples    
            with h5py.File(save_folder+'ripple_segments.hdf5', 'r') as h5f: 
                ripple_segments = fklab.segments.Segment( h5f['ripple_segments'][:] )
            offline_rippertrial = [ ripple_segments[ np.flatnonzero( trials[TRIAL].contains(ripple_segments.start)[0] )] for TRIAL in range(len(trials)) ]

            """ get the true positive and false discovery scores for this session """
            for TRIAL in range(len(trials)):
                truep = 0
                falsed = 0
                #false discovery: the fraction of online detections that did not correspond to an offline defined SWRs
                for RIPPLE in stimpertrial[TRIAL]:
                    if len(offline_rippertrial[TRIAL])>0:
                        closestoffline = np.argmin(abs(offline_rippertrial[TRIAL].center - RIPPLE))
                        difference = offline_rippertrial[TRIAL][closestoffline].center - RIPPLE
                        
                        if abs(difference) > offline_rippertrial[TRIAL][closestoffline].duration + 0.05: #50 ms difference is likely still the same ripple
                            falsed+=1 

                            """ plot the first 10 false discoveries """
                            if plotexamples and plotneg<10:
                                plotneg+=1
                                startrow = np.argmin(abs(HC_epoch_time[TRIAL]-(RIPPLE+window[0])))
                                endrow = np.argmin(abs(HC_epoch_time[TRIAL]-(RIPPLE+window[1])))
                    
                                fig,ax=plt.subplots(3,1)
                                plt.suptitle(animal+" session: "+str(SESSION)+", ripple: " + str(plotneg)+" false discovery")
                                
                                for CHAN in range(len(filteringranges)):
                                    filtered = fklabfilter.apply_filter( HC_epoch_data[TRIAL][startrow:endrow],filteringranges[CHAN],fs=fs)
                                    plottime = np.linspace(window[0],window[1],num=len(filtered)) #HC_epoch_time[TRIAL][startrow:endrow] 
                                    
                                    ax[CHAN].plot(plottime , filtered, "k",lw=0.5)
                                    ax[CHAN].axvline(0,color='r',linestyle='--') #plot the online detection in red
                                    plotdifference = offline_rippertrial[TRIAL][closestoffline].start - RIPPLE
                                    if abs(plotdifference)<window[1]:
                                        ax[CHAN].axvline(difference,color='b',linestyle='--') #plot the offline detection in blue
                                    ax[CHAN].set_xlabel("Time (s)")
                                    ax[CHAN].axis('off')
                                offline = mlines.Line2D([], [], color='b', marker=None, ls='--', label='Offline')
                                online = mlines.Line2D([], [], color='r', marker=None, ls='--', label='Online')
                                plt.legend(handles=[offline,online],bbox_to_anchor=(1, 0.5))
                                plt.show()                           
                            
                #true positive: fraction of offline SWRs that were correctly detected online
                for RIPPLE in range(len(offline_rippertrial[TRIAL])):
                    if len(stimpertrial[TRIAL])>0:
                        closestoffline = np.argmin(abs(stimpertrial[TRIAL] - offline_rippertrial[TRIAL][RIPPLE].center))
                        difference = stimpertrial[TRIAL][closestoffline] - offline_rippertrial[TRIAL][RIPPLE].center
                        if abs(difference) < offline_rippertrial[TRIAL][RIPPLE].duration + 0.05: #50 ms difference is likely still the same ripple
                            truep+=1 
                            
                            """ plot the first 10 true positives """
                            if plotexamples and plotpos<10:
                                plotpos+=1
                                startrow = np.argmin(abs(HC_epoch_time[TRIAL]-(offline_rippertrial[TRIAL][RIPPLE].start+window[0])))
                                endrow = np.argmin(abs(HC_epoch_time[TRIAL]-(offline_rippertrial[TRIAL][RIPPLE].start+window[1])))
                    
                                fig,ax=plt.subplots(3,1)
                                plt.suptitle(animal+" session: "+str(SESSION)+", ripple: " + str(plotpos)+" true positive")
                                
                                for CHAN in range(len(filteringranges)):
                                    filtered = fklabfilter.apply_filter( HC_epoch_data[TRIAL][startrow:endrow],filteringranges[CHAN],fs=fs)
                                    plottime = np.linspace(window[0],window[1],num=len(filtered)) #HC_epoch_time[TRIAL][startrow:endrow] 
                                    
                                    ax[CHAN].plot(plottime , filtered, "k",lw=0.5)
                                    ax[CHAN].axvline(0,color='b',linestyle='--')  #plot the offline detection in blue
                                    plotdifference = stimpertrial[TRIAL][closestoffline] - offline_rippertrial[TRIAL][RIPPLE].start
                                    ax[CHAN].axvline(plotdifference,color='r',linestyle='--') #plot the online detection in red
                                    ax[CHAN].set_xlabel("Time (s)")
                                    ax[CHAN].axis('off')
                                offline = mlines.Line2D([], [], color='b', marker=None, ls='--', label='Offline')
                                online = mlines.Line2D([], [], color='r', marker=None, ls='--', label='Online')
                                plt.legend(handles=[offline,online],bbox_to_anchor=(1, 0.5))
                                plt.show()
                                        
                falsdisc.append( np.divide( falsed , len(stimpertrial[TRIAL]) ) )
                truepos.append( np.divide( truep , len(offline_rippertrial[TRIAL]) ) )
                
            if plotpersession:         
                plt.figure(dpi=400)
                plt.title(animal+' session '+str(SESSION)+', stim: '+params['Stimulation'][SESSION])
                for RUN in range(len(truepos)):
                    plt.plot(falsdisc[RUN],truepos[RUN],marker='o', label='Run '+str(RUN))
                plt.legend()
                plt.xlabel('False discovery rate (FDR)')
                plt.ylabel('True positive rate (TPR)')
                plt.ylim(0,1)
                plt.xlim(0,1)
                plt.show()
                print('session '+str(SESSION)+', FDR: '+str(round(np.nanmean(falsdisc),2))+', TPR: '+str(round(np.nanmean(truepos),2)))
            
            #append this session to the averages per animal
            s = params['Stimulation'][SESSION]
            vars()['TPR_'+s+'_'+animal].append( np.nanmean(truepos)  )
            vars()['FDR_'+s+'_'+animal].append( np.nanmean(falsdisc) )
            
    #plot the session means per animal
    if plotperanimal:
        plt.figure(dpi=400)
        plt.title(animal)
        for S in range(len(stim)):
            plt.scatter(vars()['FDR_'+stim[S]+'_'+animal],vars()['TPR_'+stim[S]+'_'+animal],marker='o', color=colors[S])
        plt.xlabel('False discovery rate (FDR)')
        plt.ylabel('True positive rate (TPR)')
        plt.ylim(0,1)
        plt.xlim(0,1)
        plt.show()        
        
if plottotal:
    plt.figure(dpi=400)
    plt.title('All sessions '+indextype)
    for S in range(len(stim)):
        for animal in animals:
            plt.scatter(vars()['FDR_'+stim[S]+'_'+animal],vars()['TPR_'+stim[S]+'_'+animal],marker='o', color=colors[S])
    plt.xlabel('False discovery rate (FDR)')
    plt.ylabel('True positive rate (TPR)')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.show()        
