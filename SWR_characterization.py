#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 22 14:01:41 2022

@author: hanna
"""

import numpy as np

import fklab.io.neuralynx as nlx
import fklab.segments 
import matplotlib.pyplot as plt
from glob import glob
import h5py
import utilities
import pickle
from scipy.stats import median_abs_deviation
import pandas as pd
import fklab.core as fklabcore
import scipy.stats as scistats

plt.style.use(['seaborn-talk'])
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

#%% Charactarization of detected SWRs

""" Only look at the first session from all animals (initial learning, no stim)
And get the following variables to compare between on-maze and ITI:
    - the average envelopes (+average time of online detection)
    - Average time to online detection (only for true positives)
    - Average duration of the offline ripples
    
"""
animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] 

stimtype ='None' #do the 'None' sessions only - no artefacts
SESSION=0 #only go for the initial learning sessions
envelope_window = [-0.1,0.3]

#initialize variables
indextypes = ['Onmaze','ITI'] 
variables = ['Envelope','Rippledur','Timetodetect']
for indextype in indextypes:
    for tempvars in variables:
        vars()[tempvars+'_'+indextype]=[]
        
        
for animal in animals:
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)

    date = params['Alldates'][SESSION]
    folder = params['topfolder']+animal+'/'+date+'/'
    Nruns = len(glob(folder + 'epochs/run*'))
    
    # Load the ripple envelope (full length of data)
    awake_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/Onmaze/'
    with h5py.File(awake_folder+'reconstructed_envelopes.hdf5', 'r') as h5f: 
        ripple_envelope = h5f['original_ripple_envelope'][:]
        envelope_time = h5f['time'][:]
        envelope_fs = h5f['fs'][:]
        
    for indextype in indextypes:
        
        session_timetodetect=[]
        session_envelopes=[]
        
        save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
    
        #load trial times
        with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
            trials = fklab.segments.Segment( h5f['trials'][:] )
            
        # load detections and stimulations
        events = nlx.NlxOpen(folder+'Events.nev').data[:]
        detect_t = events.time[events.eventstring==b'detection']

        #load ripples
        with h5py.File(save_folder+'ripple_segments.hdf5', 'r') as h5f: 
            ripple_segments = fklab.segments.Segment( h5f['ripple_segments'][:] )

        #Seperate the detections per learning or rest block
        online_rippertrial = [ detect_t[ np.flatnonzero( trials[TRIAL].contains(detect_t)[0] )] for TRIAL in range(len(trials)) ]
        offline_rippertrial = [ ripple_segments[ np.flatnonzero( trials[TRIAL].contains(ripple_segments.start)[0] )] for TRIAL in range(len(trials)) ]

        for TRIAL in range(len(trials)): #for each learning block                
            for R in range(len(offline_rippertrial[TRIAL])): #for each ripple in this block
                #get the ripple envelope for this offline detection
                findripple = np.argmin( abs(envelope_time-offline_rippertrial[TRIAL][R].start) )
                startrow = findripple+ envelope_window[0]*envelope_fs
                endrow = findripple+ envelope_window[1]*envelope_fs
                session_envelopes.append( ripple_envelope[int(startrow):int(endrow)])
                #if its a true postive, get the time until online detection
                if len(online_rippertrial[TRIAL])>0:
                    closestonline = np.argmin( abs(online_rippertrial[TRIAL] - offline_rippertrial[TRIAL][R].start) )
                    difference = online_rippertrial[TRIAL][closestonline] - offline_rippertrial[TRIAL][R].start

                    if difference > 0 and difference < offline_rippertrial[TRIAL][R].duration: 
                        session_timetodetect.append( difference )
        
        #print(animal+' '+indextype+', max duration: '+str(round(np.max(ripple_segments.duration),2)))
        #print('average time to detect: '+str(round(np.nanmean(session_timetodetect),3))+' s')
        """ Append the session averages """
        vars()['Rippledur_'+indextype].append( np.median(ripple_segments.duration) )                        
        vars()['Timetodetect_'+indextype].append( np.nanmedian(session_timetodetect) ) 
        vars()['Envelope_'+indextype].append( np.mean(np.hstack(session_envelopes),axis=1) )

#%% plot ripple durations and time to detections

plotvars = ['Timetodetect','Rippledur']
printvars = ['time to detection','SWR duration']
ylabels = ['Time to detection (ms)','SWR duration (ms)']

fig,ax = plt.subplots(1,2, dpi=400)
plt.tight_layout()
ylims = (0,0.11)

for P in range(len(plotvars)):
    
    ax[P].set_title(ylabels[P])
    ax[P].set_xticks([0,1])
    ax[P].set_xticklabels( indextypes )

    for I in range(len(indextypes)):
        
        tempvars = np.vstack( vars()[plotvars[P]+'_'+indextypes[I]] )
        
        offset = np.linspace(-0.3,0.3,num=len(tempvars))
        for A in range(len(tempvars)): #for each animal
            ax[P].plot( I+offset[A], tempvars[A], 'ko',markersize=5 )
        
        tempvars =tempvars[np.flatnonzero(~np.isnan(tempvars))] #remove NaNs
        tempmean = np.mean(tempvars)
        tempsem = np.std(tempvars)/np.sqrt(len(tempvars))
        ax[P].bar(I,tempmean,yerr=tempsem,color='k',alpha=0.2,edgecolor='k',linewidth=3)
        ax[P].set_ylim(ylims)
        print(printvars[P]+', '+indextypes[I]+': M='+str(round(tempmean*1000,2))+', SEM='+str(round(tempsem*1000,2)) )
        
        
#%% plot envelopes

fig,ax = plt.subplots(2,1,dpi=400)
plt.tight_layout()

for I in range(len(indextypes)):
    ax[I].set_title(indextypes[I])
    
    tempvars = np.vstack( vars()['Envelope_'+indextypes[I]] )
    xtemp = np.linspace(envelope_window[0],envelope_window[1],num=tempvars.shape[1])
    for A in range(tempvars.shape[0]-1): #plot a separate line for each animal
        ax[I].plot(xtemp,tempvars[A,:],'k',alpha=0.2)
    #also plot the averages    
    ax[I].plot(xtemp,np.mean(tempvars,axis=0),color='k',alpha=0.5)
    ax[I].fill_between(xtemp,np.mean(tempvars,axis=0),color='k',alpha=0.5)

    ax[I].axvline(0,color='blue',linestyle='--')
    #indicate the average time to online detection
    onlinedetect = np.nanmean( vars()['Timetodetect_'+indextypes[I]] )
    ax[I].axvline(onlinedetect,color='red',linestyle='--')
    ax[I].set_ylim(0,45)

#%% plot MR and NMR (ITI and on-maze in one figure)

animals = ['B5','B9','B10','B11','B12','B17','B20','B21','B23'] #
#animals = ['B14','B18'] #higher fs - kernel dies

indextypes = ['Onmaze','ITI'] #Onmaze or ITI

#choose which things to plot
plotpersession=False #one plot with the different runs for each session
plottotal=True #one plot with all the session averages for all animals
plotexamples = True
window = [-0.10, 0.3]

import fklab.signals.filter.filter as fklabfilter
filteringranges = [10,225]
                   
#for plotting the animal averages and total averages
stimtypes = ['None'] 
colors=['black','white'] #the fill of the circles for onmaze and ITI
for indextype in indextypes:
    for animal in animals:
        vars()['TPR_'+indextype+'_'+animal]=[]
        vars()['FDR_'+indextype+'_'+animal]=[]
        
        vars()['plotcount_MR_'+indextype+'_'+animal]=0
        vars()['plotcount_NMR_'+indextype+'_'+animal]=0
#Loop through all animals
for indextype in indextypes:
    
    for animal in animals:
        
        # Get the parameters per animal
        params = utilities.get_parameters(animal) 
        SESSION = 0
    
        date = params['Alldates'][SESSION]
        folder = params['topfolder']+animal+'/'+date+'/'
        Nruns = len(glob(folder + 'epochs/run*'))
        save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
           
        #load trial times
        with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
            trials = fklab.segments.Segment( h5f['trials'][:] )

        #load ripples
        with h5py.File(save_folder+'ripple_segments.hdf5', 'r') as h5f: 
            ripple_segments = fklab.segments.Segment( h5f['ripple_segments'][:] )

        #load events
        events = nlx.NlxOpen(folder+'/Events.nev').data[:]
        detec_t = events.time[events.eventstring==b'detection']
        
        #get the offline and online detected ripples per learning block
        offline_rippertrial = [ ripple_segments[ np.flatnonzero( trials[TRIAL].contains(ripple_segments.start)[0] )] for TRIAL in range(len(trials)) ]
        online_rippertrial  = [ detec_t[ np.flatnonzero( trials[TRIAL].contains(detec_t)[0] )] for TRIAL in range(len(trials)) ]
        
        if plotexamples:
            #load the saved data (artefacts already removed)
            with open(save_folder+"HC_epoch_data.pkl", "rb") as pckl:
                HC_epoch_data=pickle.load(pckl)
            with open(save_folder+"HC_epoch_time.pkl", "rb") as pckl:
                HC_epoch_time=pickle.load(pckl)
                
        """ get the true positive and false discovery scores for this session """
        truepos=[]
        falsdisc=[]
        for TRIAL in range(len(trials)): #for each learning block
            truep = 0
            falsed = 0
            #false discovery: the fraction of online detections that did not correspond to an offline defined SWRs
            for ON in range(len(online_rippertrial[TRIAL])):
                if len(offline_rippertrial[TRIAL])>0:
                    closestoffline = np.argmin(abs(offline_rippertrial[TRIAL].start - online_rippertrial[TRIAL][ON]))
                    difference = offline_rippertrial[TRIAL][closestoffline].start - online_rippertrial[TRIAL][ON]
                    
                    if abs(difference) > offline_rippertrial[TRIAL][closestoffline].duration + 0.05: #50 ms difference is likely still the same ripple
                        falsed+=1         
                        
                        """ plot the first 5 false discoveries """
                        if plotexamples and  vars()['plotcount_NMR_'+indextype+'_'+animal]<5:
                            vars()['plotcount_NMR_'+indextype+'_'+animal]+=1
                            
                            RIPPLE = online_rippertrial[TRIAL][ON]
                            startrow = np.argmin(abs(HC_epoch_time-(RIPPLE+window[0])))
                            endrow = np.argmin(abs(HC_epoch_time-(RIPPLE+window[1])))
                
                            plt.figure(dpi=400)
                            plt.title(animal+" "+indextype+", ripple: " + str(ON)+" false discovery")
                            
                            plotdata = fklabfilter.apply_filter( HC_epoch_data[0][startrow:endrow],filteringranges,fs=params['fs'])
                            plottime = np.linspace(window[0],window[1],num=len(plotdata)) #HC_epoch_time[TRIAL][startrow:endrow] 
                            plt.plot(plottime ,plotdata, "k",lw=0.5)
                            
                            #plot online and offline detections
                            plt.axvline(0,color='r',linestyle='--') #plot the online detection in red
                            plotdifference = offline_rippertrial[TRIAL][closestoffline].start - RIPPLE
                            if abs(plotdifference)<window[1]:
                                plt.axvspan(difference,difference+offline_rippertrial[TRIAL][closestoffline].duration,color='b',alpha=0.3) #plot the offline detection in blue
                            plt.xlabel("Time (s)")
                            plt.ylim(-300,300)

                            plt.show()         
                        
            #true positive: fraction of offline SWRs that were correctly detected online
            for OFF in range(len(offline_rippertrial[TRIAL])):
                if len(online_rippertrial[TRIAL])>0:
                    closestonline = np.argmin( abs(online_rippertrial[TRIAL] - offline_rippertrial[TRIAL][OFF].start) )
                    difference = online_rippertrial[TRIAL][closestonline] - offline_rippertrial[TRIAL][OFF].start
                    if difference > 0 and difference < offline_rippertrial[TRIAL][OFF].duration: 
                        truep+=1
                        
                        """ plot the first 5 true positives """
                        if plotexamples and  vars()['plotcount_MR_'+indextype+'_'+animal]<5:
                            vars()['plotcount_MR_'+indextype+'_'+animal]+=1
                            
                            RIPPLE = offline_rippertrial[TRIAL][OFF].start
                            startrow = np.argmin(abs(HC_epoch_time-(RIPPLE+window[0])))
                            endrow = np.argmin(abs(HC_epoch_time-(RIPPLE+window[1])))
                
                            plt.figure(dpi=400)
                            plt.title(animal+" "+indextype+", ripple: " + str(ON)+" true positive")
                            
                            plotdata = fklabfilter.apply_filter( HC_epoch_data[0][startrow:endrow],filteringranges,fs=params['fs'])
                            plottime = np.linspace(window[0],window[1],num=len(plotdata)) #HC_epoch_time[TRIAL][startrow:endrow] 
                            plt.plot(plottime ,plotdata, "k",lw=0.5)
                            
                            #plot online and offline detections
                            plt.axvspan(0,offline_rippertrial[TRIAL][closestoffline].duration,color='b',alpha=0.3)
                            plt.axvline(difference,color='r',linestyle='--') #plot the online detection in red
                            plt.xlabel("Time (s)")
                            plt.ylim(-300,300)
                            plt.show()    

            falsdisc.append( np.divide( falsed , len(online_rippertrial[TRIAL]) ) )
            truepos.append( np.divide( truep , len(offline_rippertrial[TRIAL]) ) )
        
        if plotpersession:    
            plt.figure(dpi=400)
            #plt.title(animal+' session '+str(SESSION)+', channel nr: '+str(params['channames_ripples'][SESSION][CH]))
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
        vars()['TPR_'+indextype+'_'+animal].append( np.nanmean(truepos)  )
        vars()['FDR_'+indextype+'_'+animal].append( np.nanmean(falsdisc) )

if plottotal:
    plt.figure(dpi=400)
    plt.title('All sessions')
    for I in range(len(indextypes)):
        for animal in animals:
            plt.scatter(vars()['FDR_'+indextypes[I]+'_'+animal],vars()['TPR_'+indextypes[I]+'_'+animal],marker='o',edgecolor='k',facecolor=colors[I],lw=2)
    plt.xlabel('False discovery rate (FDR)')
    plt.ylabel('True positive rate (TPR)')
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.axhline(0.5,color='grey',linestyle='--')
    plt.axvline(0.5,color='grey',linestyle='--')
    plt.show()    

#% print stats

comparevars = ['TPR','FDR']
namevars = ['match rate','non-match rate']
indextypename = 'on-maze' if indextype=='Onmaze' else 'rest'

for indextype in indextypes:
    for V in range(len(comparevars)):
        
        tempvars = []
        for animal in animals:
            tempvars.append( vars()[comparevars[V]+'_'+indextype+'_'+animal] )

        tempmean = str( np.round( np.mean(tempvars), 2) )
        tempsem = str( np.round( np.std(tempvars)/np.sqrt(len(tempvars)), 2) )
        print(indextype+' '+namevars[V]+': M='+tempmean+', SEM='+tempsem+';')
        



#%% Get the detection rates table

animals= ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] #

indextypes = ['Onmaze','ITI'] 
stim = ['None','On-time','Delayed']
lockouttime = 'without' #excluding or including the 200 ms lockout

df = pd.DataFrame(animals, columns=['animal'])
for indextype in indextypes:
    for s in stim:
        vars()[indextype+'_'+s+'_vector']= [np.nan] * len(animals)

#Loop through all animals   
for A in range(len(animals)):
    animal=animals[A]
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal) 
    
    for SESSION in range(len(params['Alldates'])):
        
        s = params['Stimulation'][SESSION]
        
        if s=='Delayed' or s=='On-time' or SESSION==0:
                        
            for indextype in indextypes:
                
                date = params['Alldates'][SESSION]
                folder = params['topfolder']+animal+'/'+date+'/'
                Nruns = len(glob(folder + 'epochs/run*'))
                save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
                   
                #load trial times
                with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
                    trials = fklab.segments.Segment( h5f['trials'][:] )

                #load online detections events
                events = nlx.NlxOpen(folder+'/Events.nev').data[:]
                detec_t = events.time[events.eventstring==b'detection']
                
                #remove all detections within 200 ms 
                if lockouttime=='without':
                    detec_segments = fklab.segments.Segment( np.vstack((detec_t,detec_t+0.2)).T )
                    detec_t = detec_segments.join(gap=0).start
                
                N_rippertrial  = [ trials[TRIAL].contains(detec_t)[1] for TRIAL in range(len(trials)) ]
                totalrate = np.sum(N_rippertrial) / np.sum(trials.duration)
                vars()[indextype+'_'+s+'_vector'][A] = totalrate
                    
df = pd.DataFrame(animals, columns=['animal'])
for indextype in indextypes:
    for s in stim:
        df[indextype+'_'+s] = vars()[indextype+'_'+s+'_vector']
        
df.to_csv('/media/hanna/B42ED3C52ED37F34/Figures/Optogenetics/Detectionrates.csv')

#%% Get the detection rates over time

animals= ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] #

lockouttimes = ['without','with'] #'with' or 'without'; run twice

indextypes = ['Onmaze','ITI'] 
stim = ['On-time','Delayed']

totalblocks = 16
blocksOnmaze = np.arange(0,16,2) #[0,1,2,4,6,8,10,12,14,16]
blocksITI =  np.arange(1,17,2) #[3,5,7,9,11,13,15,17]

for lockouttime in lockouttimes:
    
    #initialize variables
    for animal in animals:
        for s in stim:
            vars()['rate_'+s+'_'+animal+lockouttime] = np.hstack( [np.nan] * totalblocks ) #append 10 values per animal - first minute, first 5 minutes, first block, rest *4, learning*4 interleaved
    
    for s in stim:
        vars()['rate_'+s] = [] 
    
    #Loop through all animals   
    for A in range(len(animals)):
        animal=animals[A]
        
        # Get the parameters per animal
        params = utilities.get_parameters(animal) 
        
        for SESSION in range(len(params['Alldates'])):
            
            s = params['Stimulation'][SESSION]
            
            if s=='Delayed' or s=='On-time':
                
                sessionvalues = np.hstack( [np.nan] *totalblocks )
                
                for indextype in indextypes:
                    
                    date = params['Alldates'][SESSION]
                    folder = params['topfolder']+animal+'/'+date+'/'
                    Nruns = len(glob(folder + 'epochs/run*'))
                    save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
                       
                    #load trial times
                    with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
                        trials = fklab.segments.Segment( h5f['trials'][:] )
    
                    #load online detections events
                    events = nlx.NlxOpen(folder+'/Events.nev').data[:]
                    detec_t = events.time[events.eventstring==b'detection']
                    
                    #remove all detections within 200 ms 
                    if lockouttime=='without':
                        detec_segments = fklab.segments.Segment( np.vstack((detec_t,detec_t+0.2)).T )
                        detec_t = detec_segments.join(gap=0).start
                    
                    N_rippertrial  = [ trials[TRIAL].contains(detec_t)[1] for TRIAL in range(len(trials)) ]
                    totalrate = np.sum(N_rippertrial) / np.sum(trials.duration)
                    #vars()[indextype+'_'+s+'_vector'][A] = totalrate
                                    
                    #segment the first learning block into the first 1 minute, the next 4 minutes and the next 10 minutes
                    """
                    if indextype=='Onmaze':
                        
                        #before segmenting, first get the rates of the first and last trial block
                        ratesfirst = trials[0].contains(detec_t)[1] / trials[0].duration
                        rateslast = trials[-1].contains(detec_t)[1] / trials[-1].duration
    
                        firstsections = np.vstack(( [ trials.start[0] , trials.start[0]+60],
                                                    [ trials.start[0]+60 , trials.start[0]+(5*60)],
                                                    [ trials.start[0]+(5*60) , trials.start[0]+(10*60)] ))
                        trials = fklab.segments.Segment( np.vstack(( firstsections,np.vstack(trials)[1:,:] )) )
                    """
                    
                    N_rippertrial  = [ trials[TRIAL].contains(detec_t)[1] for TRIAL in range(len(trials)) ]
                    ratespertrial = np.hstack(N_rippertrial) / trials.duration
                    
                    index = vars()['blocks'+indextype]
                    if len(trials)<len(index):
                        print(animal+' session '+str(SESSION)+' '+indextype+' trials missing')
                        index = index[0:len(trials)]
                    
                    sessionvalues[index] = ratespertrial
                    
                vars()['rate_'+s+'_'+animal+lockouttime][:] = sessionvalues
                vars()['rate_'+s].append( sessionvalues )

            
#%%  Plot the rates over time, on-time and delayed

xvar= np.arange(totalblocks) 
colors=['Darkorange','black']    

plt.figure(dpi=300, figsize=(6,5))
for S in range(len(stim)):
    
    tempvar = np.vstack(vars()['rate_'+stim[S]])

    tempmean = np.nanmean(tempvar,axis=0)
    tempsem = np.nanstd(tempvar,axis=0)  / np.sqrt(tempvar.shape[0])
    plt.errorbar(np.arange(totalblocks),tempmean,yerr=tempsem, fmt='o',color=colors[S],alpha=0.9, markersize=8,elinewidth=2,label="{} inhibition".format(stim[S].lower()))
    
        
    #plot the fitted lines for on-maze and ITI separately
    for indextype in indextypes:
        blocks = vars()['blocks'+indextype]
        xtemp = xvar[blocks]
        ytemp = tempmean[blocks] 
            
        res=scistats.linregress(xtemp,ytemp) 
        linestyle = '-' if indextype=='Onmaze' else '--'
        plt.plot(xtemp, res.intercept + res.slope*xtemp, color=colors[S],alpha=0.9,linestyle=linestyle, lw=1)
        
    #plt.legend()
    plt.xlabel('trial block')
    plt.xticks(np.arange(totalblocks),['1',' ','2',' ','3',' ','4',' ','5',' ','6',' ','7',' ','8',' '])
    plt.ylabel('Detection rate (Hz)')
    
for rest in np.arange(0.5,totalblocks,2):
    plt.axvspan(rest,rest+1,color='k',alpha=0.1)
    
plt.ylim(0,1.5)
    
#%% plot the difference between ontime and delayed, with lockouttime and without
    
colors=['black','blue'] #with, without lockout time
lockouttimes = ['with','without'] 


plt.figure(dpi=300, figsize=(6,5))

for L in range(len(lockouttimes)):
    
    lockouttime = lockouttimes[L]
    
    #for the missing sessions (N=3) impute the median of the other animals
    allmedian_ontime=[]
    allmedian_delayed=[]
    for animal in animals:
        allmedian_ontime.append( vars()['rate_On-time_'+animal+lockouttime] )
        allmedian_delayed.append( vars()['rate_Delayed_'+animal+lockouttime] )
    allmedian_ontime  = np.nanmedian(allmedian_ontime, axis=0)
    allmedian_delayed  = np.nanmedian(allmedian_delayed, axis=0)
    
    tempvar = []
    for animal in animals:
        
        if np.isnan(vars()['rate_On-time_'+animal+lockouttime]).all():
            vars()['rate_On-time_'+animal+lockouttime] = allmedian_ontime
        if np.isnan(vars()['rate_Delayed_'+animal+lockouttime]).all():
            vars()['rate_Delayed_'+animal+lockouttime] = allmedian_delayed
        tempvar.append( vars()['rate_On-time_'+animal+lockouttime] - vars()['rate_Delayed_'+animal+lockouttime] )
    
    tempvar = np.vstack(tempvar)
    
    tempmean = np.nanmean(tempvar,axis=0)
    tempsem = np.nanstd(tempvar,axis=0)  / np.sqrt(tempvar.shape[0])
    plt.errorbar(np.arange(totalblocks),tempmean,yerr=tempsem, fmt='o',color=colors[L],alpha=0.9, markersize=8,elinewidth=2,label=lockouttime)
    
    #plot the fitted lines for on-maze and ITI separately
    for indextype in indextypes:
        blocks = vars()['blocks'+indextype]
        xtemp = xvar[blocks]
        ytemp = tempmean[blocks] 
            
        res=scistats.linregress(xtemp,ytemp) 
        linestyle = '-' if indextype=='Onmaze' else '--'
        plt.plot(xtemp, res.intercept + res.slope*xtemp, color=colors[L],alpha=0.9,linestyle=linestyle, lw=1)
        
plt.xlabel('trial block')
plt.xticks(np.arange(totalblocks),['1',' ','2',' ','3',' ','4',' ','5',' ','6',' ','7',' ','8',' '])
plt.ylabel('Difference detection rate on-time vs delayed')
plt.axhline(0,color='grey',linestyle='--')   
for rest in np.arange(0.5,totalblocks,2):
    plt.axvspan(rest,rest+1,color='k',alpha=0.1)
            
plt.show()               
      
#%% plot bargraphs of the regression for each animal
    
colors=['black','blue'] #with, without lockout time
lockouttimes = ['with','without'] 

xvar= np.arange(totalblocks) #np.hstack(( 1/15,5/15,np.arange(1,17) ))

plt.figure(dpi=300, figsize=(1.5,5))

for L in range(len(lockouttimes)):
    
    lockouttime = lockouttimes[L]
    
    #for the missing sessions (N=3) impute the median of the other animals
    allmedian_ontime=[]
    allmedian_delayed=[]
    for animal in animals:
        allmedian_ontime.append( vars()['rate_On-time_'+animal+lockouttime] )
        allmedian_delayed.append( vars()['rate_Delayed_'+animal+lockouttime] )
    allmedian_ontime  = np.nanmedian(allmedian_ontime, axis=0)
    allmedian_delayed  = np.nanmedian(allmedian_delayed, axis=0)
    
    regressions = []
    for animal in animals:
        
        if np.isnan(vars()['rate_On-time_'+animal+lockouttime]).all():
            vars()['rate_On-time_'+animal+lockouttime] = allmedian_ontime
        if np.isnan(vars()['rate_Delayed_'+animal+lockouttime]).all():
            vars()['rate_Delayed_'+animal+lockouttime] = allmedian_delayed
            
        tempvar = vars()['rate_On-time_'+animal+lockouttime] - vars()['rate_Delayed_'+animal+lockouttime]
        #remove nans and only take the blocks on the maze (exclude ITIs):
        xtemp = xvar[np.intersect1d(blocksOnmaze,np.flatnonzero(~np.isnan(tempvar)))]
        ytemp = tempvar[np.intersect1d(blocksOnmaze,np.flatnonzero(~np.isnan(tempvar)))]                            
        
        slope,_,r,pval,_=scistats.linregress(xtemp,ytemp) 
        
        regressions.append(r)

    plt.bar(L,np.mean(regressions),yerr=np.std(regressions)/np.sqrt(len(regressions)),color=colors[L],alpha=0.5, edgecolor=colors[L])
        
    #plot the markers and lines per animal 
    scatter = np.linspace(-0.02,0.02,num=len(animals))
    for A in range(len(animals)):
        plt.plot(L+scatter[A], regressions[A],color=colors[L],mec='white',mew=1,marker='o',markersize=5 )

    #do a wilcoxon signed-rank test to see if the r2s are larger than 0
    t,pval = scistats.wilcoxon(regressions)

    print(lockouttime+' lock-out period:')
    print('regressions: M='+str(round(np.mean(regressions),2))+', SEM='+str(round(np.std(regressions)/np.sqrt(len(regressions)),2))+';')
    print('Wilcoxon signed-rank test: t('+str(len(regressions)-1)+')='+str(round(t,2))+', p='+str(round(pval,3))[1:])
    print(' ')
 
plt.xticks([0,1],['all detections','excluding lock-out'])
plt.ylim(-0.3,1.1)
plt.ylabel('R-value')       
plt.show()               

#%% Get the detection rates for the days following the on-time and delayed sessions

#look only at the first two runs after the session, because that is the minimun number used

animals= ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] #

indextypes = ['Onmaze','ITI'] 
stim = ['On-time','Delayed']

totalblocks = 4
blocksOnmaze = [0,2]
blocksITI =  [1,3]
trialindex = [0,1] #only take the first two runs and rests

for s in stim:
    vars()['rate_'+s] = [] 

#Loop through all animals   
for A in range(len(animals)):
    animal=animals[A]
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal) 
    
    for stimsession in range(len(params['Alldates'])):
        
        s = params['Stimulation'][stimsession]
        SESSION = stimsession+1 #use the day after the stim session
        
        if (s=='Delayed' or s=='On-time') and SESSION<len(params['Alldates']):
            
            sessionvalues = np.hstack( [np.nan] *totalblocks )
            
            for indextype in indextypes:
                
                date = params['Alldates'][SESSION]
                folder = params['topfolder']+animal+'/'+date+'/'
                Nruns = len(glob(folder + 'epochs/run*'))
                save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
                   
                #load trial times
                with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
                    trials = fklab.segments.Segment( h5f['trials'][:] )

                #load online detections events
                events = nlx.NlxOpen(folder+'/Events.nev').data[:]
                detec_t = events.time[events.eventstring==b'detection']

                N_rippertrial  = [ trials[TRIAL].contains(detec_t)[1] for TRIAL in trialindex ]
                ratespertrial = np.hstack(N_rippertrial) / trials.duration[trialindex]

                index = vars()['blocks'+indextype]
                sessionvalues[index] = ratespertrial

            vars()['rate_'+s].append( sessionvalues )


xvar= np.arange(totalblocks) 
colors=['Darkorange','black']    

plt.figure(dpi=300, figsize=(1.5,5))
offsets = [-0.05,0.05]
for S in range(len(stim)):
    
    tempvar = np.vstack(vars()['rate_'+stim[S]])

    tempmean = np.nanmean(tempvar,axis=0)
    tempsem = np.nanstd(tempvar,axis=0)  / np.sqrt(tempvar.shape[0])
    plt.errorbar(np.arange(totalblocks)+offsets[S],tempmean,yerr=tempsem, fmt='o',color=colors[S],alpha=0.9, markersize=8,elinewidth=2,label="{} inhibition".format(stim[S].lower()))

    #plt.legend()
    #plt.xlabel('trial block')
    plt.xticks(np.arange(totalblocks),['1',' ','2',' '])
    plt.ylabel('Detection rate (Hz)')
    
for rest in np.arange(0.5,totalblocks,2):
    plt.axvspan(rest,rest+1,color='k',alpha=0.1)
    
plt.ylim(0,1.5)

#%%

"""
Get the online and offline thresholds
"""

offline=[]
online=[]

animals= ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] #
indextypes = 'Onmaze'
remove_artefacts = False

for animal in animals:
    
    params = utilities.get_parameters(animal)
    fs = params['fs'] 

    for SESSION in range(1): #len(params['Alldates'])):

        date = params['Alldates'][SESSION]
        folder = params['topfolder']+animal+'/'+date+'/'
        save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'

        #load trial times
        with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
            trials = fklab.segments.Segment( h5f['trials'][:] )
        
        #extract the data
        channels = params['channames_ripples'][SESSION]
        HC_epoch_data = []           
        for F in range(len(channels)):

            file = folder+'CSC'+str(channels[F])+'.ncs'
            
            FP = nlx.NlxOpen(file)
            data = FP.readdata(start=trials.start[0],stop=trials.stop[0]) #get the first trial only to reduce memory load
            time = data.time
            
            #remove artefacts
            if remove_artefacts:
                # load Neuralynx events
                events = nlx.NlxOpen(folder+'Events.nev').data[:]
                detect_t = events.time[events.eventstring==b'stimulation']
            
                remove_triggers = np.hstack(( detect_t, detect_t+0.2 )) #remove the on and offset of the LEDs
                signal = fklabcore.remove_artefacts(data.signal, remove_triggers, time=time, axis=-1, window=0.005, interp='linear')
            else:
                signal = data.signal   
            #interpolate any nans
            signal = np.asarray( pd.Series(signal).interpolate())
            
            HC_epoch_data.append(signal)
            
        
        offline_threshold = 7 #lower threshold is always constant for now, upper is different per animal (7 or 8)
        offline.append( fklab.signals.core.compute_threshold(
        time,HC_epoch_data[0],threshold=offline_threshold,kind='median mad') )
         
        online_threshold = 12
        online.append( fklab.signals.core.compute_threshold(
        time,HC_epoch_data[0],threshold=online_threshold,kind='median') )
#%%

plt.figure(dpi=400)
scatter = np.linspace(-0.3,0.3,num=len(offline))
plt.bar(0,np.mean(offline),yerr=np.std(offline),fill=False)
plt.plot(0+scatter,np.hstack(offline),'ko')
plt.bar(1,np.mean(online),yerr=np.std(online),fill=False)
plt.plot(1+scatter,np.hstack(online),'ko')

plt.xticks([0,1],['Median mad threshold 7','Median threshold 12'])
