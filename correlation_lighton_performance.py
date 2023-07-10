#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  9 16:45:04 2022

@author: hanna
"""

import numpy as np

import fklab.io.neuralynx as nlx
import fklab.signals.multirate
import fklab.signals.core
import fklab.signals.filter
import fklab.signals.ripple
import matplotlib.pyplot as plt
import fklab.plot
from glob import glob
import h5py
import utilities
from fklab.behavior.task_analysis import tasks, detect_task_patterns
import pandas as pd
from scipy.stats import linregress
import scipy.stats as scistats

plt.style.use(['seaborn-talk'])
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
#import ternary

#%% First get the behavioral performance and store in a dict

animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] 
stim = ['None','On-time','Delayed']
indextype='Onmaze'

printmessages = False

variables=['Ntrials','correct','alternation','correct_inbound','correct_outbound','repetition','circular']
    
# create a dictionary with all information
allinfo = {}
for animal in animals:
    allinfo[animal] = {}
    for s in stim:
        allinfo[animal][s] = {} 
           
for animal in animals:
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)

    for SESSION in range(len(params['Alldates'])):
        
        if printmessages:
            print(animal+' '+params['Alldates'][SESSION]+':')
        s = params['Stimulation'][SESSION]
        learning = params['Allsessiontypes'][SESSION]
        
        allinfo[animal][s][learning] = {} 

        date = params['Alldates'][SESSION]
        save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'

        alltrials = pd.read_csv(save_folder+'sequences.csv')
        runs = np.hstack(alltrials.keys()[1:])
        Nruns = 8 if 'Rest' not in params["Allsessiontypes"][SESSION] else  len(runs) #non-rest sessions should always have 8 runs for comparison, rest days have as many as they have
        
        #pre-allocate the variables
        for var in variables:
            vars()[var] = np.hstack([np.nan] * Nruns)
        
        for RUN in range(len(runs)):
            trials = alltrials['Run'+str(RUN+1)][np.flatnonzero(~pd.isnull(alltrials['Run'+str(RUN+1)]))].values
            ntrials = len(trials)-1 #the first trial doesn't count for anything because it cannot be correct/incorrect, etc
            
            task = tasks['continuous-alternation-3arm-maze']                          
            outcomes = dict(reference='C',choice1='L',choice2='R')
            outcome = detect_task_patterns( trials, task, outcomes )
            
            vars()['Ntrials'][RUN]=         len(trials)-1
            vars()['correct'][RUN]=         np.divide( (len(outcome['inbound_success']) + len(outcome['outbound_success'])) , len(trials)-1 )
            vars()['correct_inbound'][RUN]= np.divide( len(outcome['inbound_success']) , len(outcome['inbound_trials']) )
            vars()['correct_outbound'][RUN]=np.divide( len(outcome['outbound_success']) , len(outcome['outbound_trials']) )
            vars()['alternation'][RUN]=     np.divide( len(outcome['alternation_trials']) , (len(trials)-3) )
            vars()['repetition'][RUN]=      np.divide( len(outcome['back_and_forth_trials']) , (len(trials)-2) )
            vars()['circular'][RUN]=        np.divide( len(outcome['circular_trials']) , (len(trials)-3) )

            if vars()['alternation'][RUN]>1:
                print(animal, SESSION, RUN, vars()['alternation'][RUN])
            
            if printmessages:
                print('Run '+str(RUN+1)+': '+str(ntrials)+' visits '+str(len(outcome['inbound_success']) + len(outcome['outbound_success']))+' correct '+str(int(vars()['correct'][RUN]*100))+'%')
                
        #at the end of the session, append the percentages to the right variables
        allinfo[animal][s][learning]['Ntrials'] = vars()['Ntrials']
        allinfo[animal][s][learning]['correct'] = vars()['correct']
        allinfo[animal][s][learning]['alternation']= vars()['alternation']
        allinfo[animal][s][learning]['correct_inbound'] = vars()['correct_inbound']
        allinfo[animal][s][learning]['correct_outbound']= vars()['correct_outbound']
        allinfo[animal][s][learning]['repetition']=  vars()['repetition']
        allinfo[animal][s][learning]['circular']=  vars()['circular']


#%% Then plot the correlation between the fraction of time light on vs alternation performance

indextypes = ['Onmaze','ITI']

#for plotting the animal averages and total averages
stimtypes = ['Delayed','On-time']
colors =['black','darkorange']

for indextype in indextypes:
    #initialize variables
    for s in stimtypes:
        vars()[s+'_lighton']=[]
        vars()[s+'_alternation'] = []
    
    # load saved ripples and check rate over runs
    for A in range(len(animals)):
        animal=animals[A]
        
        # Get the parameters per animal
        params = utilities.get_parameters(animal)
        
        for SESSION in range(len(params['Alldates'])):
                
            s = params['Stimulation'][SESSION]
            if (s=='On-time' or s=='Delayed'):
            
                date = params['Alldates'][SESSION]
                folder = params['topfolder']+animal+'/'+date+'/'
                Nruns = len(glob(folder + 'epochs/run*'))
                save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
                   
                #load trial times
                with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
                    trials = fklab.segments.Segment( h5f['trials'][:] )
    
                #load online stimulations
                events = nlx.NlxOpen(folder+'/Events.nev').data[:]
                stimulation = events.time[events.eventstring==b'stimulation']
                addstimpertrial=[]
                for TRIAL in range(len(trials)):
                    stimsegment = np.vstack( stimulation[ np.flatnonzero( trials[TRIAL].contains(stimulation)[0] )] )
                    stimsegment = fklab.segments.Segment( np.hstack(( stimsegment, stimsegment+0.2)) ).join(gap=0)
    
                    addstimpertrial.append(  np.sum(stimsegment.duration) )# / trials[TRIAL].duration )
                
                total_lighton = np.sum( addstimpertrial ) / np.sum(trials.duration)
                learning = list(allinfo[animal][s].keys())[0]
                alternation_performance = np.nanmean( allinfo[animal][s][learning]['alternation'] )
                
                vars()[s+'_lighton'].append( total_lighton )
                vars()[s+'_alternation'].append( alternation_performance )
    
    # plot correlations
    plt.figure(dpi=400)
    
    for s in stimtypes:
        
        total_lighton=[]
        total_alternation=[]
        
        color='black' if s=='Delayed' else 'darkorange'
        lighton = np.vstack( vars()[s+'_lighton'] )
        alternation = np.vstack( vars()[s+'_alternation'] )
        
        for SESSION in range(len(lighton)):
            plt.plot(lighton[SESSION],alternation[SESSION],color=color,marker='o')
            
            #append to the totals to to a correlation analysis
            #if ~np.isnan(lighton[SESSION]) and ~np.isnan(alternation[SESSION]):
            total_lighton.append(lighton[SESSION][0])
            total_alternation.append(alternation[SESSION][0])
    
        #plot the correlation
        gradient, intercept, r_value, p_value, std_err = linregress(total_lighton,total_alternation)
        x1=np.linspace(np.min(total_lighton),np.max(total_lighton),100)
        y1=gradient*x1+intercept
        plt.plot(x1,y1,color=color,linestyle='--',label='r='+str(round(r_value,2))+', p='+str(round(p_value,3)))
        
    plt.legend()
    plt.title(indextype)
    plt.xlabel('Fraction of time light on')
    plt.ylabel('Alternation performance')      
    plt.xlim(0,0.3)
    plt.ylim(0,1)
    plt.show()
    
#%% Fraction light on On-time vs delay

indextypes = ['Onmaze','ITI']

#for plotting the animal averages and total averages
stimtypes = ['Delayed','On-time']
colors =['black','darkorange']

animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] 

for indextype in indextypes:
    #initialize variables
    for s in stimtypes:
        vars()[s+'_online']=np.zeros( len(animals) )
        vars()[s+'_online'][:] = np.nan
        #vars()[stim+'_offline']=[]
    
    # load saved ripples and check rate over runs
    for A in range(len(animals)):
        animal=animals[A]
        
        # Get the parameters per animal
        params = utilities.get_parameters(animal)
        
        for SESSION in range(len(params['Alldates'])):
            
            s = params['Stimulation'][SESSION]
            if (s=='On-time' or s=='Delayed'): #there are no stimulation artifacts in 'None' sessions
            
                date = params['Alldates'][SESSION]
                folder = params['topfolder']+animal+'/'+date+'/'
                Nruns = len(glob(folder + 'epochs/run*'))
                save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
                   
                #load trial times
                with h5py.File(save_folder+'trials.hdf5', 'r') as h5f: 
                    trials = fklab.segments.Segment( h5f['trials'][:] )

                #load online stimulations
                events = nlx.NlxOpen(folder+'/Events.nev').data[:]
                stimulation = events.time[events.eventstring==b'stimulation']
                addstimpertrial=[]
                for TRIAL in range(len(trials)):
                    stimsegment = np.vstack( stimulation[ np.flatnonzero( trials[TRIAL].contains(stimulation)[0] )] )
                    stimsegment = fklab.segments.Segment( np.hstack(( stimsegment, stimsegment+0.2)) ).join(gap=0)
                    
                    addstimpertrial.append(  np.sum(stimsegment.duration) )
                
                total_lighton = np.sum(addstimpertrial) / np.sum(trials.duration) * 100

                vars()[s+'_online'][A] = total_lighton
    
    #plot bargraphs of total light on fractions
    colors=['black','darkorange'] 
    plt.figure(dpi=300, figsize=(2.5,5))
    for S in range(len(stimtypes)):
        tempvars = vars()[stimtypes[S]+'_online']
        plt.bar(S,np.nanmean(tempvars),yerr=np.nanstd(tempvars)/np.sqrt(len(tempvars)),color=colors[S],alpha=0.5, edgecolor='k')
    #plot the dots & connecting lines per animal
    scatter = np.linspace(-0.02,0.02,num=len(animals))
    for A in range(len(animals)):
        animalvalues = [np.nan,np.nan]
        for S in range(len(stimtypes)):
            animalvalues[S] = vars()[stimtypes[S]+'_online'][A]
            plt.plot(S+scatter[A], animalvalues[S],color=colors[S],mec='white',mew=1,marker='o',markersize=5 )
        plt.plot(np.arange(S+1)+scatter[A],animalvalues,'k',lw=1,alpha=0.5, zorder=0)
    plt.xticks(np.arange(S+1), ['{}\ninhibition'.format(x.lower()) for x in stimtypes])
    plt.title(indextype)
    plt.ylim(0,50)
    plt.ylabel('Fraction light on')
    
    #print stats:
    indextypename = 'on-maze' if indextype=='Onmaze' else 'rest'
    print('Percentage of time light on '+indextypename)
    for S in range(len(stimtypes)):
        tempvars = vars()[stimtypes[S]+'_online'] 
        
        #fill in any NaNs by the median of all other sessions
        tempvars[np.flatnonzero(np.isnan(tempvars))] = np.nanmedian( tempvars )

        tempmean = str( np.round( np.mean(tempvars), 2) )
        tempsem = str( np.round( np.std(tempvars)/np.sqrt(len(tempvars)), 2) )
        print(stimtypes[S]+': M='+tempmean+', SEM='+tempsem+';')
        
        vars()['mean_'+stimtypes[S]] = tempvars

    t,pval = scistats.ttest_rel(vars()['mean_'+stimtypes[0]],vars()['mean_'+stimtypes[1]])      
    print('paired samples t-test: t('+str(len(vars()['mean_'+stimtypes[0]])-1)+')='+str(round(t,2))+', p='+str(round(pval,3))[1:])
