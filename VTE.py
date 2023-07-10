#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 27 10:20:10 2022

@author: hanna
"""

# import libraries:
import numpy as np
from glob import glob
import fklab.io.neuralynx as nlx
import fklab.segments 
import fklab.utilities.yaml as yaml #import tools to parse YAML
#from fklab.behavior.task_analysis import tasks, detect_task_patterns
import utilities
#from pathlib import Path
#import pickle
import h5py
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as scistats
#import matplotlib.lines as mlines
import fklab.signals.filter.filter as fklabfilter

plt.style.use(['seaborn-talk'])
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400

#%% VTE analyses

indextype = 'Onmaze' #indextype is only Onmaze here
plotexamples = False

animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] #
seqname = 'L','C','R'

trialtypes = ['incomplete','linger','change','headturn','direct']
stim = ['Delayed','On-time']

linger_speedthresh = 5 #cm/s threshold for determining lingering behavior (lower than this threshold)
linger_durationthresh=2 #2 minimum number of seconds of below the speed threshold to count as a 'linger trial'

#initialize variables
for animal in animals:
    for s in stim:
        for trialt in trialtypes:
            vars()[trialt+'_'+s+'_'+animal]=[] #one value per animal: percent of trial type
        
for animal in animals:
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):
        
        s = params['Stimulation'][SESSION]
        if s=='On-time' or s=='Delayed':
            
            plotcount = 0 #plot the first 10 trials for this animal
            
            #initialize variables
            for trialt in trialtypes:
                vars()['session_'+trialt+'_'+s+'_'+animal] = 0
            trialcount=0
                    
            date = params['Alldates'][SESSION]
            folder = params['topfolder']+animal+'/'+date+'/'
            save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
    
            events = nlx.NlxOpen(folder+'Events.nev').data[:]
            stim_t = events.time[events.eventstring==b'stimulation']
                
            #load trials neuralynx time
            Nruns = len(glob(folder + 'epochs/run*'))
    
            sequences=[[] for R in range(Nruns)]
            
            with open(folder+'/epochs/maze.yaml') as file:
                maze = yaml.unsafe_load(file)
            truepathlength = 80*3
            pixel2cm = maze['maze']['shapes']['track']['shape'].pathlength / truepathlength
            
                
            for RUN in range(Nruns):
    
                runfolder = glob(folder+'/epochs/*'+str(RUN+1))[0]
                with h5py.File(runfolder+'/position.hdf5', "r") as behavior:
                    behavior_time=behavior['time'][:]
                    xypos = behavior['position'][:]
                    behavior_velocity = abs(behavior['velocity'][:]) / pixel2cm
                    head_direction = behavior['head_direction'][:]
                    diode1 = behavior['diodes']['diode1'][:]
                    diode2 = behavior['diodes']['diode2'][:]
                    
                fps = 1/(behavior_time[1]-behavior_time[0])
                #interpolate velocity and x and y pos so that there are no NaNs
                xpos = np.vstack( pd.Series(xypos[:,0]).interpolate() )
                ypos = np.vstack( pd.Series(xypos[:,1]).interpolate() )
                xypos = np.hstack(( xpos, ypos ))
                behavior_velocity = np.hstack( pd.Series(behavior_velocity).interpolate() )
                head_direction = np.hstack( pd.Series(head_direction).interpolate() )
                
                areas = utilities.get_areas(maze,xypos) #,size='Amaze') 
                
                #segment the times spent at the platforms
                fullsegments = [ fklab.segments.Segment.fromlogical( np.hstack(areas)=='P'+str(P),behavior_time ) for P in range(1,4) ]
                orderedsegments = fklab.segments.Segment( np.vstack(fullsegments)[ np.argsort( np.vstack(fullsegments)[:,0] ) ] )
                centervisits = fklab.segments.Segment.fromlogical( np.hstack(areas)=='Center' , behavior_time ).start
                
                #!! remove doubles (there should always be a center visit in between platform visits)
                trialtimes = []
                sequences = []
                for TRIAL in range(len(centervisits)):
                    findnextplatform = np.flatnonzero( orderedsegments.start > centervisits[TRIAL] )
                    if len(findnextplatform)>0:
                        trialtimes.append( orderedsegments.start[findnextplatform[0]] )
                        
                        #get the platform name
                        findrow =np.argmin( abs(trialtimes[-1]-behavior_time) )
                        platformname = int(areas[findrow][0][-1]) #1, 2 or 3
                        sequences.append( seqname[ int(np.flatnonzero( np.hstack(params['Sequences'][SESSION]) == platformname )) ] )
                        
                trialtimes =fklab.segments.Segment( np.vstack(( trialtimes[0:-1],trialtimes[1:] )).T )    
                keeptrials = np.flatnonzero( trialtimes.duration>=1 )
                if len(keeptrials)==0:
                    continue
                
                trialtimes = trialtimes[ keeptrials ]
                sequences =  np.vstack(( sequences[0:-1],sequences[1:] )).T[keeptrials]

                for TRIAL in range(len(trialtimes)):
                   
                    #get the trace of this trial
                    startrow =np.argmin( abs(trialtimes[TRIAL].start-behavior_time) )
                    endrow =np.argmin( abs(trialtimes[TRIAL].stop-behavior_time) )
                    #segments in the center (not on the platforms):
                    armsegments = fklab.segments.Segment.fromlogical(np.vstack([ 'P' not in area for area in np.hstack(areas[startrow:endrow]) ])) + startrow

                    #did the animal linger?
                    lingerdur = 0
                    for SEG in range(len(armsegments)):
                        trialvelocity = behavior_velocity[int(armsegments[SEG].start):int(armsegments[SEG].stop)]
                        lingersegments = fklab.segments.Segment.fromlogical( trialvelocity < linger_speedthresh,behavior_time )
                        lingerdur += np.sum(lingersegments.duration)
                    linger = 1 if lingerdur>linger_durationthresh else 0 #count the trial as a linger trial if there were at least 2 seconds of a speed under 5 cm/s
                    
                    #did the animal return to the same platform?
                    incomplete = 1 if sequences[TRIAL,0]==sequences[TRIAL,1] else 0

                    #did the animal visit another arm?
                    countarms = np.flatnonzero([ 'A' in temparms for temparms in np.unique(areas[startrow:endrow]) ])
                    change = 1 if len(countarms)==3 else 0
                    
                    #calculate head direction score  
                    headdircount=0
                    for SEG in range(len(armsegments)): #all the time not on the platforms during this trial
                        if armsegments[SEG].duration>0:
                            headdirtrial = head_direction[int(armsegments.start[SEG]):int(armsegments.stop[SEG])]
                            timetrial = behavior_time[int(armsegments.start[SEG]):int(armsegments.stop[SEG])]
                            headdirchanges = fklab.segments.Segment.fromlogical( np.sign(np.diff(headdirtrial))==1,timetrial )
                            #remove short duration and short gaps
                            headdirchanges = headdirchanges[np.flatnonzero(headdirchanges.duration>1)]
                            headdirchanges = headdirchanges.join(gap=1)
                            
                            headdircount += len( headdirchanges )
                    #a trial is considered a 'head turn trial' if the difference in direction
                    #changes at least once for at least 1 second
                    headturn = 1 if headdircount>2 else 0
                    
                    direct = 0 if incomplete+linger+change+headturn>0 else 1
                    
                    #add this trial to the session counts
                    for trialt in trialtypes:
                        vars()['session_'+trialt+'_'+s+'_'+animal] += vars()[trialt]

                    """ plot examples """
                    if plotexamples and plotcount<10 and headturn==1 and incomplete+linger+change==0:
                        
                        plt.figure(dpi=400)
                        
                        plt.title(animal+' trial '+str(TRIAL)+', incomplete: '+str(incomplete)+
                                                              ', linger: '+str(linger)+
                                                              ', change: '+str(change)+
                                                              ', head turn: '+str(headturn)+
                                                              ', direct: '+str(direct))
                                  
                        plt.plot(xypos[startrow:endrow,0],xypos[startrow:endrow,1],'k',alpha=0.5)

                        #plot the head direction
                        downsampled_pos = np.arange(startrow,endrow,5)
                        for pos in downsampled_pos:
                            dx,dy = utilities.point_pos(xypos[pos,0], xypos[pos,0], d=10, rad=head_direction[pos])
                            plt.arrow(xypos[pos,0], xypos[pos,1], dx, dy, lw=1,width=1,head_width=7,head_length=5,facecolor='k')

                        plt.axis('off')
                        #plot a dotted line for the full maze to orient where the trace is taking place
                        centercenter = maze['maze']['shapes']['Center']['shape'].center
                        for P in range(1,4):
                            pcenter = maze['maze']['shapes']['P'+str(P)]['shape'].center
                            plt.plot( np.linspace(centercenter[0],pcenter[0],num=50),
                                     np.linspace(centercenter[1],pcenter[1],num=50),'k',alpha=0.2,linestyle='--')
                        plt.show()
                        plotcount +=1
                        
                #count the total number of trials for this session        
                trialcount+=len(trialtimes)
            #at the end of the session, append the percentage of each trialtype (divided by total number of trials)        
            for trialt in trialtypes:
                vars()[trialt+'_'+s+'_'+animal] = vars()['session_'+trialt+'_'+s+'_'+animal] / trialcount

    print(animal+" done")
            
#%% Plot bargraphs

colors=['black','darkorange']
xspacing = np.linspace(-0.1,0.1,num=len(animals))

for trialt in trialtypes:

    plt.figure(figsize=(2.5,5))
    plt.title(trialt)
    
    animalvalues = np.vstack( [[np.nan]*len(animals)]*2 ).T
    for S in range(len(stim)):
        s = stim[S]

        tempvar = []
        for A in range(len(animals)):
            animal = animals[A]
            if type(vars()[trialt+'_'+s+'_'+animal])==float:
                tempvar.append( vars()[trialt+'_'+s+'_'+animal] )
                animalvalues[A,S] = vars()[trialt+'_'+s+'_'+animal]
                
        tempvar=np.vstack(tempvar)

        tempmean = np.mean(tempvar)
        tempsem = np.std(tempvar)  / np.sqrt(len(tempvar))
        
        plt.bar(S,tempmean,yerr=tempsem,color=colors[S],edgecolor=colors[S],linewidth=2,alpha=0.5,fill=True)
    
    for A in range(len(animals)):

        for S in range(len(stim)):
            #plot the markers
            plt.plot(S+xspacing[A],animalvalues[A,S],alpha=0.5,marker='o',markerfacecolor=colors[S],markeredgecolor='white', zorder=0)
        #plot the lines between the markers   
        plt.plot(np.arange(S+1)+xspacing[A],animalvalues[A,:],'k',lw=1,alpha=0.5,marker=None) #'o',markerfacecolor=colors[S],markeredgecolor='white', zorder=0)
       
    plt.xticks([0,1], stim)
    plt.ylim(0,1.1)
    
    #do the stats
    
    #impute the median
    for S in range(len(stim)):
        animalvalues[np.flatnonzero(np.isnan(animalvalues[:,S])),S] = np.nanmedian(animalvalues[:,S])
    # T-test
    t,pval = scistats.ttest_rel( animalvalues[:,0],animalvalues[:,1] )
    
    means=[]
    sems=[]
    for S in range(len(stim)):
        tempvars = animalvalues[:,S]
        means.append( str(round(np.mean(tempvars),2)) )
        sems.append( str(round(np.std(tempvars)/np.sqrt(len(tempvars)),2)) )

    print(trialt+' trials:')
    print('on-time inhibition: M='+means[0]+', SEM='+sems[0]+';')
    print('delayed inhibition: M='+means[1]+', SEM='+sems[1]+';')
    print('paired samples t-test: t('+str(animalvalues.shape[0]-1)+')='+str(round(t,2))+', p='+str(round(pval,3))[1:])
    print(' ')
