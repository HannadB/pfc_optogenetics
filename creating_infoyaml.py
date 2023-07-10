

#import fklab.io.data as fkdata
#from pathlib import Path
#from fklab.io.data import import_epoch_info
import os, yaml
import fklab.io.neuralynx as nlx
import fklab.segments 
import h5py
import utilities

from glob import glob
import numpy as np

#%%
animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23']
indextype='ITI'
yamltemplate='/home/hanna/info.yaml'

#gettimes=[]

for animal in animals:
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):
    
        date = params['Alldates'][SESSION]
        folder = params['topfolder']+animal+'/'+date+'/'
        
        #make the preprocessing folder
        if len(glob(folder+'preprocess/'))==0: #if there is not yet a folder called preprocessing
            os.mkdir(folder+'preprocess/')
    
        #open the template info.yaml
        with open(yamltemplate,'r') as f:
             info=yaml.load(f)    

        events = nlx.NlxOpen(folder+'Events.nev')
        eventstrings = events.data.eventstring[:]
        
        #first add all the general information
        info['epochs'][0]['environment'] = '3ArmR'    
        info['subject']['id'] = animal
        info['subject']['birthday'] = 'null'
        info['procedures'][0]['date'] = 'null'
        info['procedures'][0]['implantID'] = '8HPC_opticPFC'
        info['implants']['kind'] = 'tetrodes'
        info['implants']['model'] = 'null'
        info['implants']['sensors'] = 'tetrodes'
        info['experiment']['date'] = params['Alldates'][SESSION]
        
        #then add the epoch times
        maxruns=8 #there are max 8 runs per session
        if animal == 'B9' and date == '2021-11-13_11-39-08':
            maxruns = 2 #there are actually 3 runs for this session, but the last one the wire broke and the signal is unusable
        
        epochdict = info['epochs'][0] #copy the epoch dict
        info['epochs']=[epochdict.copy() for RUN in range(maxruns)]
        #epochdict = [[info['epochs'][0]][0] for RUN in range(maxruns)]
        print(animal+', '+params['Alldates'][SESSION]+':')
        for RUN in range(maxruns): #max 8 runs
            startindex=np.argwhere(['start run '+str(RUN+1) in str(eventstrings[I]) for I in range(len(eventstrings)) ])    
            endindex = np.argwhere(['end run '+str(RUN+1) in str(eventstrings[I]) for I in range(len(eventstrings)) ])
            
            if animal=='B21' and date=='2022-07-14_09-47-09' and RUN==0:
                startindex=startindex[1]
                endindex=endindex[1]
                
            if len(startindex)==0 and len(endindex)>0: #if there was no start index, take the end of the previous rest index
                startindex=np.argwhere(['end rest '+str(RUN) in str(eventstrings[I]) for I in range(len(eventstrings)) ])
            if len(endindex)==0 and len(startindex)>0: #if there was no endindex, take the start of the rest index
                 endindex = np.argwhere(['start rest '+str(RUN+1) in str(eventstrings[I]) for I in range(len(eventstrings)) ])
                
            if len(startindex)>0 or len(endindex)>0: #there has to be at least a start index or an endindex
                 #add a solution for when the length is more than one (then pick the one closer to start+15 or end-15)
                 if len(startindex)>1:
                     durations = [ (events.data.time[int(endindex)] - events.data.time[int(startindex[I])])/60 for I in range(len(startindex)) ]
                     rightduration = np.flatnonzero( [durations[I]>10 and durations[I]<20 for I in range(len(durations))] )
                     if len(rightduration)>1:
                         rightduration = int(np.argwhere( abs(15-np.hstack(durations))==min(abs(15-np.hstack(durations))) ) )
                         
                 else:
                     rightduration=0
                 
                 #one exception that I don't know how to deal with (the epoch started and then I took the animal off the maze, put him back on, and wrote 'start run 5' instead of 'start run 6')    
                 if animal=='B3' and params['Alldates'][SESSION]=='2021-08-27_10-21-14' and RUN==5:
                     startindex = [13807]
                     
                 starttime = events.data.time[int(startindex[rightduration])] if len(startindex)>0 else events.data.time[int(endindex)] - 15*60
                 
                 if len(endindex)>1:
                     durations = [ (events.data.time[int(endindex[I])] - events.data.time[int(startindex)])/60 for I in range(len(endindex)) ]
                     rightduration = np.flatnonzero( [durations[I]>10 and durations[I]<20 for I in range(len(durations))] )
                 else:
                     rightduration=0
                 endtime = events.data.time[int(endindex[rightduration])] if len(endindex)>0 else events.data.time[int(startindex)] + 15*60
    
                 info['epochs'][RUN]['id'] = 'run'+str(RUN+1)
                 info['epochs'][RUN]['time'] = [int(starttime),int(endtime)]
                 print('Run '+str(RUN+1)+': '+str((int(endtime)-int(starttime))/60)+' minutes')

                 #gettimes.append( np.hstack((nfo['epochs'][RUN]['time'], info['epochs'][RUN]['time']-events.data.time[0] )) )
        with open(folder+'info.yaml','w') as f:
            yaml.dump(info, f)

#import pandas as pd
#df = pd.DataFrame(np.vstack(gettimes),columns=['start','end','start_s','end_s'])

#%% Add the ITI epochs to the info.yaml

animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23']
yamltemplate='/home/hanna/info.yaml'
indextypes = ['Onmaze','ITI']

#gettimes=[]

for animal in animals:
    
    params = utilities.get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):

        date = params['Alldates'][SESSION]
        folder = params['topfolder']+animal+'/'+date+'/'
        
        #load trial times onmaze and ITI
        save_folder_Onmaze =  params['topfolder']+"/Analysis/"+animal+'/'+params['Alldates'][SESSION]+'/Onmaze/'
        with h5py.File(save_folder_Onmaze+'trials.hdf5', 'r') as h5f: 
            trials_Onmaze = fklab.segments.Segment( h5f['trials'][:] )
        
        save_folder_ITI =  params['topfolder']+"/Analysis/"+animal+'/'+params['Alldates'][SESSION]+'/ITI/'
        with h5py.File(save_folder_ITI+'trials.hdf5', 'r') as h5f: 
            trials_ITI = fklab.segments.Segment( h5f['trials'][:] )
           
        #open the template info.yaml
        with open(yamltemplate,'r') as f:
             info=yaml.safe_load(f)    
        
        #first add all the general information
        info['epochs'][0]['environment'] = '3ArmR'    
        info['subject']['id'] = animal
        info['subject']['birthday'] = 'null'
        info['procedures'][0]['date'] = 'null'
        info['procedures'][0]['implantID'] = '8HPC_opticPFC'
        info['implants']['kind'] = 'tetrodes'
        info['implants']['model'] = 'null'
        info['implants']['sensors'] = 'tetrodes'
        info['experiment']['date'] = params['Alldates'][SESSION]
        
        Nruns = len(trials_Onmaze) + len(trials_ITI)
        #then add the epoch times
        epochdict = info['epochs'][0] #copy the epoch dict
        info['epochs']=[epochdict.copy() for RUN in range(Nruns)]
        
        for indextype in indextypes:
            trials = vars()['trials_'+indextype]
            
            for RUN in range(len(trials)):
                
                N = RUN if indextype=='Onmaze' else int(RUN+len(trials_Onmaze))
                
                info['epochs'][N]['id'] = 'run'+str(RUN+1) if indextype=='Onmaze' else 'rest'+str(RUN+1)
                info['epochs'][N]['time'] = [int(trials.start[RUN]),int(trials.stop[RUN])]

        with open(folder+'info.yaml','w') as f:
            yaml.dump(info, f)