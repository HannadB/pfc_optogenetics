
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from fklab.behavior.task_analysis import tasks, detect_task_patterns
import numpy as np
import pandas as pd
import utilities
import scipy.stats as scistats

#from mpl_toolkits.mplot3d import axes3d, Axes3D 
from itertools import combinations

plt.style.use(['seaborn-talk'])
import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 400
#import ternary

#%% Get the information from the csv files and store in a dict

animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23'] 
stim = ['None','On-time','Delayed']
indextype='Onmaze'

printmessages = False

skipincompletetrials = True #if true, remove all inclomplete trials (e.g. C- C)
#Our analyses assume that these are not there, plus the learning criterium is based
#on Hive, in which we take out all the doubles too 

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
            
            if skipincompletetrials:
                keepindex = [True] * len(trials) #by default keep all trials
                for C in range(len(trials)):
                    samechoice = np.flatnonzero(trials==trials[C]) - C #the index of all the trials with this choice, and the distance from the current trial
                    if len(np.flatnonzero(samechoice==-1))==1: #if there was a same choice at a distance of -1, this is a double  
                        keepindex[C]=False
                trials = trials[keepindex]
                        
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
        
#save behavioral information in a hdf5 file
#with h5py.File(params['topfolder']+"/Analysis/behavior_info.hdf5", 'w') as h5f: 
#    h5f.create_dataset('info', data=allinfo)

#%% plot learningcurves

stim = ['Delayed','On-time'] 
colors =['black','darkorange']

learningtypes=['Initial','Reversal1','Reversal2']
#plot = ['correct','correct_inbound','correct_outbound','Ntrials','repetition','circular','alternation']

plot = ['correct','alternation','repetition','circular']
for p in plot:
    
    title = 'Total number of trials' if p=='Ntrials' else p+' trials'
    ylabel = 'Number of trials' if p=='Ntrials' else 'fraction of trials' 
    
    plt.figure(dpi=400, figsize=(5,5))
    plt.title(title)
        
    for Sk, S in enumerate(range(len(stim))):
        tempvar = []
        runminus1 = [] #get the last run of the previous day too
        
        for animal in animals:
            for learning in learningtypes:
                if learning in allinfo[animal][stim[S]].keys():
                    tempvar.append( allinfo[animal][stim[S]][learning][p] )   
                    #find the last score of the previous day (run -1)
                    previousday = 'Initial_Restday' if learning == 'Reversal1' else 'Reversal1_Restday'
                    Lpreviousday = np.flatnonzero([previousday in K for K in allinfo[animal]['None'].keys()])[-1]
                    Lpreviousday = list(allinfo[animal]['None'].keys())[Lpreviousday]
                    #take the last run only:
                    #runminus1.append( allinfo[animal]['None'][Lpreviousday][p][-1] )
                    #or take the mean of the previous day:
                    runminus1.append( np.mean(allinfo[animal]['None'][Lpreviousday][p]) )
        tempvar=np.hstack(( np.vstack(runminus1), np.vstack(tempvar) ))
        
        tempmean = np.nanmean(tempvar,axis=0)
        tempsem = np.nanstd(tempvar,axis=0)  / np.sqrt(tempvar.shape[0])
        
        plt.errorbar(0 + (Sk-0.5)*0.2,tempmean[0],yerr=tempsem[0], fmt='o',color=colors[S],alpha=0.9, markersize=8,elinewidth=2)
        plt.errorbar(np.arange(8)+1,tempmean[1:],yerr=tempsem[1:], fmt='o-',color=colors[S],alpha=0.9, markersize=8,elinewidth=2,label="{} inhibition".format(stim[S].lower()))
        if p=='correct':
            plt.legend(loc='lower right')
        else:
            plt.legend()
        plt.xlabel('trial block')
        plt.xticks(np.arange(9),['pre','1','2','3','4','5','6','7','8'])
        plt.ylabel(ylabel)
        #plt.axvline(0.5,color='grey',lw=1)
        if p=='alternation':
            plt.axhline(1/6, color='lightgrey', lw=1, zorder=0)
        if p=='correct':
            plt.axhline(1/2, color='lightgrey', lw=1, zorder=0)
        if p != 'Ntrials':
            plt.ylim(0,1)
        #plt.savefig(title+'.png')
        
#%% Plot distribution pyramid stereotypical behvior

summary = {'Delayed': {}, 'On-time': {}}

for S in ['Delayed', 'On-time']:
    
    for v in ['circular', 'repetition', 'alternation']:
        summary[S][v] = np.ravel([
            [
                np.nanmean(allinfo[animal][S][learning][v])
                for learning in ['Reversal1', 'Reversal2']
                if learning in allinfo[animal][S]
            ]
            for animal in animals
            if S in allinfo[animal] and len(allinfo[animal][S])>0
        ])


def plot_ax():               #plot tetrahedral outline
    verts=[[0,0,0],
     [1,0,0],
     [0.5,np.sqrt(3)/2,0],
     [0.5,0.28867513, 0.81649658]]
    lines=combinations(verts,2)
    for x in lines:
        line=np.transpose(np.array(x))
        ax.plot3D(line[0],line[1],line[2],c='0',lw=0.5)

def label_points(labels = None, **kwargs):  #create labels of each vertices of the simplex
    a=(np.array([1,0,0,0])) # Barycentric coordinates of vertices (A or c1)
    b=(np.array([0,1,0,0])) # Barycentric coordinates of vertices (B or c2)
    c=(np.array([0,0,1,0])) # Barycentric coordinates of vertices (C or c3)
    d=(np.array([0,0,0,1])) # Barycentric coordinates of vertices (D or c3)
    if labels is None:
        labels=['a','b','c','d']
    cartesian_points=get_cartesian_array_from_barycentric([a,b,c,d])
    for point,label in zip(cartesian_points,labels):
        # if 'a' in label:
        #     ax.text(point[0],point[1]-0.075,point[2], label, size=16)
        # elif 'b' in label:
        #     ax.text(point[0]+0.02,point[1]-0.02,point[2], label, size=16)
        # else:
        ax.text(point[0],point[1],point[2], label, **kwargs)

def get_cartesian_array_from_barycentric(b):      #tranform from "barycentric" composition space to cartesian coordinates
    verts=[[0,0,0],
         [1,0,0],
         [0.5,np.sqrt(3)/2,0],
         [0.5,0.28867513, 0.81649658]]

    #create transformation array vis https://en.wikipedia.org/wiki/Barycentric_coordinate_system
    t = np.transpose(np.array(verts))        
    t_array=np.array([t.dot(x) for x in b]) #apply transform to all points

    return t_array

def plot_3d_tern(df,c='1',**kwargs): #use function "get_cartesian_array_from_barycentric" to plot the scatter points
#args are b=dataframe to plot and c=scatter point color
    bary_arr=df.values
    cartesian_points=get_cartesian_array_from_barycentric(bary_arr)
    return ax.scatter(cartesian_points[:,0],cartesian_points[:,1],cartesian_points[:,2],c=c,**kwargs)

#Create Dataset 1
np.random.seed(123)
c1=np.random.normal(8,2.5,20)
c2=np.random.normal(8,2.5,20)
c3=np.random.normal(8,2.5,20)
c4=[100-x for x in c1+c2+c3]   #make sur ecomponents sum to 100

#df unecessary but that is the format of my real data
S = 'Delayed'
df1=pd.DataFrame(
    data=[
        summary[S]['circular'],
        summary[S]['repetition'],
        1 - summary[S]['alternation'] - summary[S]['circular'] - summary[S]['repetition'],
        summary[S]['alternation'],
    ],
    index=['c1','c2','c3','c4']
).T
#df1=df1/100


#Create Dataset 2
np.random.seed(1234)
c1=np.random.normal(16,2.5,20)
c2=np.random.normal(16,2.5,20)
c3=np.random.normal(16,2.5,20)
c4=[100-x for x in c1+c2+c3]

S = 'On-time'
df2=pd.DataFrame(
    data=[
        summary[S]['circular'],
        summary[S]['repetition'],
        1 - summary[S]['alternation'] - summary[S]['circular'] - summary[S]['repetition'],
        summary[S]['alternation'],
    ],
    index=['c1','c2','c3','c4']
).T
#df2=df2/100


#Create Dataset 3
np.random.seed(12345)
c1=np.random.normal(25,2.5,20)
c2=np.random.normal(25,2.5,20)
c3=np.random.normal(25,2.5,20)
c4=[100-x for x in c1+c2+c3]

df3=pd.DataFrame(data=[c1,c2,c3,c4],index=['c1','c2','c3','c4']).T
df3=df3/100

fig = plt.figure(figsize=(12,8))
ax = fig.add_subplot(projection='3d')
#ax = Axes3D(fig) #Create a 3D plot in most recent version of matplot

plot_ax() #call function to draw tetrahedral outline

#label_points(['circular','repetitive','rotated\nalternation',''],ha='center') #label the vertices

h1 = plot_3d_tern(df1,c=df1['c4'],depthshade=False,vmin=-0.25, vmax=1, cmap='gray_r',lw=0.5, s=50) #call function to plot df1

h2 = plot_3d_tern(df2,c=df2['c4'],depthshade=False,vmin=-0.25, vmax=1, cmap='Oranges',lw=0.5, s=40) #...plot df2

#plot_3d_tern(df3,'g') #...

ax.set_proj_type('ortho')
ax.axis('off');
ax.view_init(90,90)
ax.set(
    xlim=(-0.1,1.1),
    ylim=(-0.1,1.1),
    zlim=(-0.1,1.1)
);

c1 = plt.colorbar(h2, label='alternation', shrink=0.5, pad=0.0, fraction=0.05)
c2 = plt.colorbar(h1, shrink=0.5, pad=0.0, fraction=0.05)

c1.ax.set(ylim=(0,1))
c2.ax.set(ylim=(0,1))
c2.set_ticklabels([])

#%% plot bargraphs
    
stim = ['Delayed','On-time'] #'None',
learningtypes=['Reversal1','Reversal2'] #'Initial',
colors=['black','darkorange'] #'k',
markers=['v','o'] #'o', #different markers for initial, reversal & reversal2
#plot = ['correct','correct_inbound','correct_outbound','Ntrials','repetition','circular','alternation']
plot = ['correct','alternation','repetition','circular']

division = ['first block','overall']

for p in plot:

    title = 'Total number of trials' if p == 'Ntrials' else p+' trials'
    ylabel = 'Number of trials' if p=='Ntrials' else 'Fraction of trials' 

    plt.figure(dpi=300, figsize=(2.5,5))
    plt.title(title)
    
    xticks=[[0,1],[3,4]]
    for D in range(len(division)):
        d = division[D]        
        
        for S in range(len(stim)):
            tempvar = []
            for animal in animals:
                for learning in learningtypes:
                    if learning in allinfo[animal][stim[S]].keys():
                        if d=='first block':
                            tempvar.append( allinfo[animal][stim[S]][learning][p][0] ) #first block 
                        elif d=='overall':
                            tempvar.append( np.nanmean( allinfo[animal][stim[S]][learning][p] ) ) #average all 8 runs together 
            tempvar=np.vstack(tempvar)
            
            #tempmedian = np.median(tempvar)
            tempmean = np.mean(tempvar)
            tempsem = np.std(tempvar)  / np.sqrt(len(tempvar))
            
            plt.bar(xticks[D][S],tempmean,yerr=tempsem,color=colors[S],alpha=0.5, edgecolor='k')
        
            
        xspacing = np.linspace(-0.1,0.1,num=len(animals))
        for ANIMAL in range(len(animals)):
            animal = animals[ANIMAL]
            animalvalues = [np.nan] * len(stim) #append three values per animal; one per stim type
            for S in range(len(stim)):
                L = [L for L in range(len(learningtypes)) if learningtypes[L] in allinfo[animal][stim[S]].keys()]
                if len(L)==1:
                    L=int(L[0])
                    if d=='first block':
                        animalvalues[S] = np.nanmean( allinfo[animal][stim[S]][learningtypes[L]][p][0] )#first block 
                    elif d=='overall':
                        animalvalues[S] = np.nanmean( allinfo[animal][stim[S]][learningtypes[L]][p] )
                    plt.plot(xticks[D][S]+xspacing[ANIMAL],animalvalues[S],color=colors[S],marker=markers[L],ls='none',markersize=7, markeredgecolor='w', markeredgewidth=1, alpha=0.75)
            """ plot the markers and the lines separately"""     
            plt.plot(xticks[D]+xspacing[ANIMAL],animalvalues,'k',lw=1,alpha=0.5, zorder=0)
           
        # #manually make a legend so that all the colors are the same - only the markers are important
        # #initial = mlines.Line2D([], [], color='k', marker='o', ls='', label='Initial')
        # reversal1 = mlines.Line2D([], [], color='k', marker='v', ls='', label='Reversal1')
        # reversal2 = mlines.Line2D([], [], color='k', marker='o', ls='', label='Reversal2')
        # plt.legend(handles=[reversal1,reversal2],bbox_to_anchor=(1, 0.5), #initial,
        #            fancybox=True, shadow=True, ncol=1)
        plt.xticks([np.mean(xticks[0]),np.mean(xticks[1])], division)
        plt.ylabel(ylabel)
        if p != 'Ntrials':
            plt.ylim(0,1)
        # #plt.savefig(title+'.png')
    
#%% Plot 1 learningcurve per animal over all learning sessions

animals =  ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21','B23']
stim = ['None','Delayed','On-time']

variable='alternation' #e.g. correct or alternation
#learningtypes=['Initial','Initial_Restday1','Initial_Restday2','Reversal1','Reversal1_Restday1','Reversal1_Restday2','Reversal2']

for animal in animals:
    Nrunsperday=[]
    Nsessiontypes=[]
    stimtypes=[]
    
    params = utilities.get_parameters(animal)
    learningtypes = params['Allsessiontypes'][:]
    
    prev_lt = 'Initial'
    last_trialblock = 0
    
    learningcurve = []
    trialblocks = []
    
    for learning in learningtypes:
        lt,*_ = learning.split('_')
        findS = [s for s in stim if learning in allinfo[animal][s].keys() ]
        if len(findS)==1:
            findS = findS[0]
            
            if lt != prev_lt:
                #print(animal, prev_lt, lt)
                learningcurve.append([np.nan])
                trialblocks.append([last_trialblock + 0.5])
            
            prev_lt = lt
            
            learningcurve.append( allinfo[animal][findS][learning][variable] )
            trialblocks.append( np.arange(len(learningcurve[-1])) + last_trialblock + 1 )
            
            last_trialblock = trialblocks[-1][-1]
            
            stimtypes.append( findS )
            Nrunsperday.append( len(learningcurve[-1]) )
    
    #plot the figure
    title = 'Learning curve over days animal '+animal
    ylabel = 'fraction correct trials' if variable=='correct' else 'Fraction correct alternation trials'
    
    plt.figure(dpi=300, figsize=(5,5))
    plt.title(title)
    #plt.plot(np.arange(len(np.hstack(learningcurve))), np.hstack(learningcurve),color='black',marker='o',markersize=8,label=variable)
    plt.plot(np.hstack(trialblocks), np.hstack(learningcurve),color='black',marker='o',markersize=5,label=variable)
    #plt.legend()
    plt.ylim(0,1.05)
    plt.xlabel('trial block')
    plt.ylabel(ylabel)
    if variable=='alternation':
        plt.axhline(1/6,color='grey',linestyle='-',lw=0.5)
    elif variable == 'correct':
        plt.axhline(0.5,color='grey',linestyle='-',lw=0.5)
    #indicate the different days with dotted lines, and the different stimulation protocols in colors
    cumruns = np.cumsum(Nrunsperday) + 1
    for LINE in range(len(cumruns)):
        plt.axvline(cumruns[LINE]-0.5, color='lightgrey',linestyle='-', lw=1, zorder=0)
        if stimtypes[LINE]=='On-time':
            plt.axvspan(cumruns[LINE-1]-0.5,cumruns[LINE]-0.5, color='darkorange',alpha=0.2, zorder=0)
        elif stimtypes[LINE]=='Delayed':
            plt.axvspan(cumruns[LINE-1]-0.5,cumruns[LINE]-0.5, color='black',alpha=0.2, zorder=0)
    #plt.savefig(title+'.png')
    
    #plot a green circle around the learning block (first block of >=80%)
    if variable == 'correct': #for alternation, there is no behavioral criterium
        trialbreaks = np.hstack((0, np.flatnonzero(np.isnan(np.hstack(learningcurve))) ))
        allearningtrials = np.flatnonzero(np.hstack(learningcurve)>=0.8) #all trial blocks of >=80%
        for trialbreak in trialbreaks:
            learnedtrial = allearningtrials[ np.flatnonzero(allearningtrials>trialbreak) ]
            if len(learnedtrial)>0:
                LT = learnedtrial[0]
                plt.plot(np.hstack(trialblocks)[LT], np.hstack(learningcurve)[LT],color='black',marker='o',markersize=5,markeredgecolor='green',markeredgewidth=3)
        
#%% Obtain for each reversal the variables (% correct etc) according to the previous rule

#animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21']
stim = ['None','On-time','Delayed'] #only applicable for on-time and delayed (for None there is no previous rule)

variables=['Ntrials','correct','alternation','correct_inbound','correct_outbound','repetition','circular']
skipincompletetrials = True 
    
# create a dictionary with all information
allinfoprevious = {}
for animal in animals:
    allinfoprevious[animal] = {}
    for s in stim:
        allinfoprevious[animal][s] = {} 
           
for animal in animals:
    
    # Get the parameters per animal
    params = utilities.get_parameters(animal)

    for SESSION in range(len(params['Alldates'])):
        
        s = params['Stimulation'][SESSION]
        
        if s in stim:
            learning = params['Allsessiontypes'][SESSION]
            
            allinfoprevious[animal][s][learning] = {} 
            
            folder = params['topfolder']+animal+'/'+params['Alldates'][SESSION]+'/'
            
            date = params['Alldates'][SESSION]
            save_folder =  params['topfolder']+"/Analysis/"+animal+'/'+date+'/'+indextype+'/'
            alltrials = pd.read_csv(save_folder+'sequences.csv')
            
            #alltrials = pd.read_csv(folder+'preprocess/sequences.csv')
            runs = np.hstack(alltrials.keys()[1:])
            Nruns = 8
            Nruns = 8 if 'Rest' not in params["Allsessiontypes"][SESSION] else len(runs) #non-rest sessions should always have 8 runs for comparison, rest days have as many as they have
            
            #pre-allocate the variables
            for var in variables:
                vars()[var] = np.hstack([np.nan] * Nruns)
            
            for RUN in range(len(runs)):
                trials = alltrials['Run'+str(RUN+1)][np.flatnonzero(~pd.isnull(alltrials['Run'+str(RUN+1)]))].values
                ntrials = len(trials)-1 #the first trial doesn't count for anything because it cannot be correct/incorrect, etc
                
                if skipincompletetrials:
                    keepindex = [True] * len(trials) #by default keep all trials
                    for C in range(len(trials)):
                        samechoice = np.flatnonzero(trials==trials[C]) - C #the index of all the trials with this choice, and the distance from the current trial
                        if len(np.flatnonzero(samechoice==-1))==1: #if there was a same choice at a distance of -1, this is a double  
                            keepindex[C]=False
                    trials = trials[keepindex]
                    
                """now change the C,L and R considering the previous rule"""
                """except for None, then consider the next rule"""
                trialsprev = trials.copy()
                seqname = 'L', 'C', 'R'
                for seq in range(len(seqname)):
                    findtrials = np.flatnonzero(trials==seqname[seq])
                    armnumber = np.hstack(params['Sequences'][SESSION])[seq] 
                    changeto = seqname[ int(np.flatnonzero(np.hstack(params['Sequences'][SESSION+1 if s=='None' else SESSION-1])==armnumber)) ]
                    trialsprev[findtrials]=changeto

                task = tasks['continuous-alternation-3arm-maze']                          
                outcomes = dict(reference='C',choice1='L',choice2='R')
                outcome = detect_task_patterns( trialsprev, task, outcomes )
                
                vars()['Ntrials'][RUN]=ntrials
                vars()['correct'][RUN]=(len(outcome['inbound_success']) + len(outcome['outbound_success'])) / ntrials if ntrials>0 else np.nan
                vars()['alternation'][RUN]=len(outcome['alternation_trials']) / ntrials if ntrials>0 else np.nan
                vars()['correct_inbound'][RUN]=len(outcome['inbound_success']) / len(outcome['inbound_trials']) if len(outcome['inbound_trials'])>0 else np.nan
                vars()['correct_outbound'][RUN]=len(outcome['outbound_success']) / len(outcome['outbound_trials']) if len(outcome['outbound_trials'])>0 else np.nan
                vars()['repetition'][RUN]=len(outcome['back_and_forth_trials']) / ntrials if ntrials>0 else np.nan
                vars()['circular'][RUN]=len(outcome['circular_trials']) / ntrials if ntrials>0 else np.nan
                
                
            #at the end of the session, append the percentages to the right variables
            allinfoprevious[animal][s][learning]['Ntrials'] = vars()['Ntrials']
            allinfoprevious[animal][s][learning]['correct'] = vars()['correct']
            allinfoprevious[animal][s][learning]['alternation']= vars()['alternation']
            allinfoprevious[animal][s][learning]['correct_inbound'] = vars()['correct_inbound']
            allinfoprevious[animal][s][learning]['correct_outbound']= vars()['correct_outbound']
            allinfoprevious[animal][s][learning]['repetition']=  vars()['repetition']
            allinfoprevious[animal][s][learning]['circular']=  vars()['circular']
            
#%% plot ratio old correct / new correct 

stim = ['Delayed','On-time']
colors=['black','darkorange']

learningtypes=['Reversal1','Reversal2']
plot = ['alternation'] #'correct',

for p in plot:
    plt.figure(dpi=300, figsize=(5,5))
    title = p+' trials'
    plt.title(title)
        
    for Sk, S in enumerate(range(len(stim))):
        tempvar = []
        runminus1 = [] #get the last run of the previous day too
        for animal in animals:
            for learning in learningtypes:
                if learning in allinfo[animal][stim[S]].keys():
                    newcorrect = allinfo[animal][stim[S]][learning][p] 
                    oldcorrect = allinfoprevious[animal][stim[S]][learning][p]
                    tempvar.append( newcorrect - oldcorrect )        
                    
                    #find the last score of the previous day (run -1)
                    previousday = 'Initial_Restday' if learning == 'Reversal1' else 'Reversal1_Restday'
                    Lpreviousday = np.flatnonzero([previousday in K for K in allinfo[animal]['None'].keys()])[-1]
                    Lpreviousday = list(allinfo[animal]['None'].keys())[Lpreviousday]
                    
                    #take the last run only:
                    #tempprev = -(allinfo[animal]['None'][Lpreviousday][p][-1] - allinfoprevious[animal]['None'][Lpreviousday][p][-1])    
                    #or take the mean of the previous day:
                    tempprev = np.mean( -(allinfo[animal]['None'][Lpreviousday][p] - allinfoprevious[animal]['None'][Lpreviousday][p]) )    
                        
                    runminus1.append( tempprev )

                    
        #tempvar=np.vstack(tempvar)
        #print(runminus1)
        tempvar=np.hstack(( np.vstack(runminus1), np.vstack(tempvar) ))
        
        tempmean = np.nanmean(tempvar,axis=0)
        tempsem = np.nanstd(tempvar,axis=0)  / np.sqrt(tempvar.shape[0])
        
        #plt.errorbar(np.arange(8)+1,tempmean,yerr=tempsem, fmt='o-',color=colors[S],alpha=0.9, markersize=10,elinewidth=1,label="{} inhibition".format(stim[S].lower()))
        
        plt.errorbar(0 + (Sk-0.5)*0.2,tempmean[0],yerr=tempsem[0], fmt='o',color=colors[S],alpha=0.9, markersize=8,elinewidth=2)
        plt.errorbar(np.arange(8)+1,tempmean[1:],yerr=tempsem[1:], fmt='o-',color=colors[S],alpha=0.9, markersize=8,elinewidth=2,label="{} inhibition".format(stim[S].lower()))
        
        
        #for RUN in range(8):
        #    plt.plot([RUN+1]*tempvar.shape[0],tempvar[:,RUN],color=colors[S],marker='o',ls='none',markersize=2)
        plt.legend(loc='lower right')
        plt.xlabel('trial block')
        plt.ylabel('Δ performance\n(current rule - previous rule)')
        plt.axhline(0,color='lightgray',ls='-',lw=1, zorder=0)
        plt.ylim(-1,1)
        plt.gca().set(xticks=[0, 1,2,3,4,5,6,7,8], xticklabels=['pre', 1,2,3,4,5,6,7,8],
                      yticks=[-1,-0.8,-0.6,-0.4,-0.2,0,.2,.4,.6,.8,1],
                      yticklabels=[1.,0.8,0.6,0.4,0.2,0.,.2,.4,.6,.8,1.])
        #plt.savefig(title+'.png')
        
#%% bargraphs correct old-correct new

stim = ['Delayed','On-time']
learningtypes=['Reversal1','Reversal2'] 
colors=['black','darkorange'] 
markers=['v','o']
plot = ['alternation'] #'correct',

division = ['first block','overall']

for p in plot:

    title =  p+' trials'
    ylabel = 'Correct new - correct old'
    
    plt.figure(dpi=400, figsize=(2.5,5))
    plt.title(title)
    
    xticks=[[0,1],[3,4]]
    for D in range(len(division)):
        d = division[D]
        
        for S in range(len(stim)):
            tempvar = []
            for animal in animals:
                for learning in learningtypes:
                    if learning in allinfo[animal][stim[S]].keys():
                        newcorrect = allinfo[animal][stim[S]][learning][p] 
                        oldcorrect = allinfoprevious[animal][stim[S]][learning][p]
                        if d =='first block':
                            tempvar.append( newcorrect[0] - oldcorrect[0] )  
                        elif d=='overall':
                            tempvar.append( np.nanmean(newcorrect) - np.nanmean(oldcorrect) )  
            tempvar=np.vstack(tempvar)
    
            tempmean = np.nanmean(tempvar)
            tempsem = np.nanstd(tempvar)  / np.sqrt(len(tempvar))
            
            plt.bar(xticks[D][S],tempmean,yerr=tempsem,color=colors[S],alpha=0.5, edgecolor='k')

        #plot the markers and lines per animal
        xspacing = np.linspace(-0.1,0.1,num=len(animals))
        for ANIMAL in range(len(animals)):
            animal = animals[ANIMAL]
            animalvalues=[np.nan,np.nan] #append two values per animal; one per stim type
            for S in range(len(stim)):
                L = [L for L in range(len(learningtypes)) if learningtypes[L] in allinfo[animal][stim[S]].keys()]
                if len(L)==1:
                    L=int(L[0])
                    newcorrect = allinfo[animal][stim[S]][learningtypes[L]][p] 
                    oldcorrect = allinfoprevious[animal][stim[S]][learningtypes[L]][p]
                    if d =='first block':
                        animalvalues[S] = newcorrect[0] - oldcorrect[0]
                    elif d=='overall':
                        animalvalues[S] = np.nanmean( newcorrect - oldcorrect )
                    
                    plt.plot(xticks[D][S]+xspacing[ANIMAL],animalvalues[S],color=colors[S],marker=markers[L],ls='none',markersize=7, markeredgecolor='w', markeredgewidth=1, alpha=0.75)
            """ plot the markers and the lines separately"""     
            tempxticks = xticks[0:2] if d=='first block' else xticks[2:]
            plt.plot(xticks[D]+xspacing[ANIMAL],animalvalues,'k',lw=0.5,alpha=0.5)
    
        plt.xticks([np.mean(xticks[0]),np.mean(xticks[1])], division)
        plt.ylabel(ylabel)
        plt.ylim(-1,1)
        # #plt.savefig(title+'.png')
        
#%% The average and range of trials to learning criterium (for fig 1)
stim = 'None'
maxtrials=22

learningtypes=['Initial','Initial_Restday1','Initial_Restday2','Initial_Restday3']
plot = ['correct','correct_inbound','correct_outbound']

fig,ax = plt.subplots(1,3,dpi=400)
plt.tight_layout()

for P in range(len(plot)):
    p = plot[P]

    ax[P].set_title(p+' trials')   
    
    vars()['learningtrials_'+p]=[]
    allcurves = np.zeros((len(animals),maxtrials))
    allcurves[:]=np.nan
    for A in range(len(animals)):
        animal = animals[A]
        tempvar = []
        for learning in learningtypes:
            if learning in allinfo[animal][stim].keys():
                tempvar.append( allinfo[animal][stim][learning][p] )  
                
        tempvar=np.hstack(tempvar)
        untiltrial = np.flatnonzero(tempvar>0.8)[0]+1 if len(np.flatnonzero(tempvar>0.8))>0 else len(tempvar)
        vars()['learningtrials_'+p].append(untiltrial)

        allcurves[A,0:len(tempvar)] = tempvar      
        ax[P].plot(np.arange(len(tempvar)),tempvar,alpha=0.4, markersize=3)
    
    ax[P].plot(np.arange(maxtrials),np.nanmean(allcurves,axis=0),color='k',alpha=1,markersize=3)
    ax[P].set_xlabel('15-minute epoch')
    ax[P].set_xticks(np.arange(0,maxtrials,2))
    ax[P].set_ylim(0,1.1)
    #ax[P].set_ylabel('Fraction of trials')
    ax[P].axvline(8,color='grey',linestyle='--')
    ax[P].axvline(16,color='grey',linestyle='--')
    ax[P].axhline(0.8,color='grey',linestyle='-')
    #print(p+' overall mean: '+str(round(np.nanmean(allcurves),2)))

learningtrials_correct = vars()['learningtrials_correct']
print('Average number of epochs to reach criterium: '+str(round(np.mean(learningtrials_correct),2))+', range: '+str(np.min(learningtrials_correct))+'-'+str(np.max(learningtrials_correct)))   
print(' ')

#%% print stats

#trial parameters
stim = ['Delayed','On-time'] #'None',
division = ['first block','overall']

plot = ['correct','Ntrials','alternation','repetition','circular']

for p in plot:
    
    for d in division:
        
        for S in range(len(stim)):
            vars()[stim[S]+'_'+p]=[np.nan] * len(animals)
            for A in range(len(animals)):
                animal=animals[A]
    
                #in the case of missing data, replace the missing values by the median of the rest of the animals
                if len(allinfo[animal][stim[S]])==0: #if there was no delayed or on-time                   
                    other_animals=[]
                    for tempanimal in animals:
                        if len(allinfo[tempanimal][stim[S]])>0:
                            templearning = list(allinfo[tempanimal][stim[S]].keys())[0]
                            if d=='overall':
                                other_animals.append( np.nanmean(allinfo[tempanimal][stim[S]][templearning][p]) ) #average over 8 runs
                            elif d=='first block':
                                other_animals.append( allinfo[tempanimal][stim[S]][templearning][p][0] ) #only take the first run
                    vars()[stim[S]+'_'+p][A] =  np.nanmedian( other_animals ) 
                else:
                    learning = list(allinfo[animal][stim[S]].keys())[0]
                    if d=='overall':
                        vars()[stim[S]+'_'+p][A] = np.nanmean( allinfo[animal][stim[S]][learning][p] ) #average all 8 runs together 
                    elif d=='first block':
                        vars()[stim[S]+'_'+p][A] = allinfo[animal][stim[S]][learning][p][0] #take only the first run
        # T-test
        t,pval = scistats.ttest_rel(vars()[stim[0]+'_'+p],vars()[stim[1]+'_'+p])
    
        means=[]
        sems=[]
        for s in stim:
            means.append( str(round(np.mean(vars()[s+'_'+p])*100,2)) )
            sems.append( str(round(np.std(vars()[s+'_'+p])/np.sqrt(len(vars()[s+'_'+p]))*100,2)) )
        
        title = 'Total number of trials' if p=='Ntrials' else 'percent '+p
        print(title+', '+d+':')
        print(stim[0].lower()+' inhibition: M='+means[0]+', SEM='+sems[0]+';')
        print(stim[1].lower()+' inhibition: M='+means[1]+', SEM='+sems[1]+';')
        print('paired samples t-test: t('+str(len(vars()[s+'_'+p])-1)+')='+str(round(t,2))+', p='+str(round(pval,3))[1:])
        print(' ')
    
# old rule vs new rule stats
stim = ['Delayed','On-time']
plot = ['alternation']

for p in plot:
    
    for d in division:
        
        for S in range(len(stim)):
            vars()[stim[S]+'_'+p]=[]
            for animal in animals:
    
                    #in the case of missing data, replace the missing values by the median of the rest of the animals
                    if len(allinfo[animal][stim[S]])==0: #if there was no delayed or on-time                   
                        other_animals_new=[]
                        other_animals_prev=[]
                        for tempanimal in animals:
                            if len(allinfo[tempanimal][stim[S]])>0:
                                templearning = list(allinfo[tempanimal][stim[S]].keys())[0]
                                if d=='overall':
                                    other_animals_new.append( np.nanmean( allinfo[tempanimal][stim[S]][templearning][p]) ) #average over 8 runs
                                    other_animals_prev.append( np.nanmean(allinfoprevious[tempanimal][stim[S]][templearning][p]) )                                    
                                elif d=='first block':    
                                    other_animals_new.append( allinfo[tempanimal][stim[S]][templearning][p][0]) #only take the first run
                                    other_animals_prev.append( allinfoprevious[tempanimal][stim[S]][templearning][p][0] )
                                    
                        vars()[stim[S]+'_'+p].append( np.nanmedian(np.hstack(other_animals_new) - np.hstack(other_animals_prev) ) ) 
                        
                    else: 
                        learning = list(allinfo[animal][stim[S]].keys())[0]
                        newcorrect = allinfo[animal][stim[S]][learning][p] 
                        oldcorrect = allinfoprevious[animal][stim[S]][learning][p]  
                        
                        if d=='overall':
                            vars()[stim[S]+'_'+p].append( np.nanmean(newcorrect - oldcorrect) )
                        elif d=='first block':
                            vars()[stim[S]+'_'+p].append( newcorrect[0] - oldcorrect[0] )
    
        # T-test
        t,pval = scistats.ttest_rel(vars()[stim[0]+'_'+p],vars()[stim[1]+'_'+p])
    
        means=[]
        sems=[]
        for s in stim:
            means.append( str(round(np.mean(vars()[s+'_'+p])*100,2)) )
            sems.append( str(round(np.std(vars()[s+'_'+p])/np.sqrt(len(vars()[s+'_'+p]))*100,2)) )
        
        print('rotated alternation trials, '+d+':')
        print(stim[0].lower()+': M='+means[0]+', SEM='+sems[0]+';')
        print(stim[1].lower()+': M='+means[1]+', SEM='+sems[1]+';')
        print('paired samples t-test: t('+str(len(vars()[s+'_'+p])-1)+')='+str(round(t,2))+', p='+str(round(pval,3))[1:])
        print(' ')
        