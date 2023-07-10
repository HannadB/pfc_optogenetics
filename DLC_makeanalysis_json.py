
from glob import glob
import json
from utilities import get_parameters
import fklab.utilities.yaml as yaml
from collections import OrderedDict
import cv2

"""running deeplabcut for multiple projects

Step 1. Open the deeplabcut gui, select one video and add the right markers to the config file (e.g. redLED, blueLED, tailbase)
In command window:
ssh -Y ratlab@nerfcluster-fs (password= ratlab)
module load conda-local
source activate dlc-gui
ipython
import deeplabcut
deeplabcut.launch_dlc()


Step 2. Copy the config file to the outputfolder specified below (create e.g. "/mnt/fk-fileserver/Project_PFC/analysis_json/")
Step 3. Run the "add to config" cell below to add all the videos to the config file
Step 4. Copy the config file back to the folder in the nerfcluster, open the DLC gui and extract frames
Step 5. Delete all the frames of the sleepbox 
Step 6. Label frames (with 20 frames per video, deleting the sleepbox ones, this leaves ~600 frames for the tetrode project and ~1000 frames for neuropixels, = about 2 hours of manual work)
Step 7. After labeling the frames, go to the tab "Create training dataset" in the DLC gui and create the training dataset
Wait for this message in the console: "The training dataset is successfully created. Use the function 'train_network' to start training. Happy training!"
Step 8. Manually create 3 json files:
    - params_DLC_train.json   (take the default from the nerfcluster-fs documentation but put at least 500.000 iterations)
    - params_DLC_evaluate.json (take the default from the nerfcluster-fs, change the config file path)
    - params_DLC_analysis.json (manually put one videofolder name in this file, and save it the outpufolder listed below)
Step 9. Train the network (in the nerfcluster: launch_slurm_4DLC.py --input_json_file /absolutepath/to/your/params_DLC_train.json --out_directory /absolutepath/to/your/output/directory/ --program_to_run training --use_dlc_slurm_partition)
Step 10. Evaluate the network (in the nerfcluster: launch_slurm_4DLC.py --input_json_file /absolutepath/to/your/params_DLC_evaluate.json --out_directory /absolutepath/to/your/output/directory/ --program_to_run evaluation --use_dlc_slurm_partition)

Step 11. Convert all videos. This is necessary because the VT1.mpg is often corrupted and doesn't result in the same number of frames as the DLC output, therefore, convert everything to mp4
    - Run cell "Create convert_videos.sh" below. This creates a bash script to convert all video files
    - Copy this bash script to the the nerfcluster ratlab folder
    - Then open a terminal:
    ssh ratlab@nerfcluster-fs (password ratlab)
    bash convert_videos.sh
    
Step 12. Check if all output videos indeed have the same number of frames. Run the "Check if the duration of the videos matches" cell below

Step 13. Analyze all videos.
    - Run the cell "Create analysis.json" below to create a separate json file for each video
    - Then open a terminal:
    ssh ratlab@nerfcluster-fs (password ratlab)
    nano launch_analysis.sh
    - copy this bash script: (add the right number of videos in the loop, and make sure the input and output paths are right and that --node 3 is added)

    for i in {0..55}
    do
       launch_slurm_4DLC.py --input_json_file /mnt/fk-fileserver/Project_PFC_tetrodes/analysis_json/params_DLC_analysis{i}.json --out_directory /mnt/fk-fileserver/Project_PFC_tetrodes/analysis_json/ --program_to_run analysis --use_dlc_slurm_partition --job_name DLC_analysis{i} --node 3
    done
    
    - Then Ctrl+x, Y, Enter (to close and say yes to save)
    - cat launch_analysis.sh   - to verify the script is correctly written
    - bash launch_analysis.sh  - to run it

    - After analysis, check again if the DLC labeled video matches the original video length
    
Step 14. Localize. Run the last cell below to add the datafile-line to info.yaml in order use the
DLC output in localize.

"""

project = 'optogenetics' #tetrodes or neuropixels

if project == 'neuropixels':
    animals = "Andy","Collin","Dave","Eric","Felix","Ignacio"
    outputfolder = "/mnt/fk-fileserver/Project_PFC/analysis_json/" #create one output folder where all the json files and such can be saved
    from utilities import concatenate_sessions
    
elif project == 'tetrodes':
    animals = 'Marnix','Bart','Roberto','Pete','George'
    outputfolder = "/mnt/fk-fileserver/Project_PFC_tetrodes/analysis_json/"
    from utilities import get_folder
    
elif project == 'optogenetics':
    animals = ['B5','B9','B10','B11','B12','B14','B17','B18','B20','B21']
    outputfolder = '/mnt/fk-fileserver/Project_PFC_optogenetics/Analysis/DLC/'

#%%  Add to config (step 3)
#Adding all the videos to the config.yaml

with open(outputfolder+"config.yaml", "r") as file:
    configyaml = yaml.unsafe_load(file)

cropvalues = configyaml['video_sets'][list(configyaml['video_sets'].keys())[0]]['crop']

for animal in animals:

    params = get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):
        if project == 'neuropixels':
            folders, folder = concatenate_sessions(SESSION, params)
        elif project == 'tetrodes':
            folder = get_folder(params['Alldates'][SESSION], params, animal)
            folders = [folder]
        elif project == 'optogenetics':
            folder = "/mnt/fk-fileserver/Project_PFC_optogenetics/"+animal+"/"+params['Alldates'][SESSION]+"/"
            folders = [folder]
            
        for folder in folders: #there are only multiple folders in the case of neuropixels concatenated sessions
            
            nlxfolder = folder+'neuralynx/' if project=='neuropixels' else folder
            
            if len(glob(nlxfolder+'VT1.mpg'))>0: #if there was a video file
                videoname = nlxfolder+'VT1.mpg'
                #add the video name and the crop values to the config file
                configyaml['video_sets'][videoname]=OrderedDict()
                configyaml['video_sets'][videoname]['crop']=cropvalues


with open(outputfolder+"config.yaml", 'w') as outfile:
   yaml.dump(configyaml, outfile)                
                
                     
#%% Create convert_videos.sh (step 11)
#making bash script that converts all videos to mp4

# Get all video names and make a params_analysis.json file for each video
commandline = ['ffmpeg -i','videoname','-c:v h264 -crf 18 -preset fast','outputname'] #the command line for converting mpg to mp4
alltext=['#!/bin/bash'] # the bash script should start with this

for animal in animals:

    params = get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):
        if project == 'neuropixels':
            folders, folder = concatenate_sessions(SESSION, params)
        elif project == 'tetrodes':
            folder = get_folder(params['Alldates'][SESSION], params, animal)
            folders = [folder]
        elif project == 'optogenetics':
            folder = "/mnt/fk-fileserver/Project_PFC_optogenetics/"+animal+"/"+params['Alldates'][SESSION]+"/"
            folders = [folder]
            
        for folder in folders: #there are only multiple folders in the case of neuropixels concatenated sessions
                        
            nlxfolder = folder+'neuralynx/' if project=='neuropixels' else folder
            
            if len(glob(nlxfolder+'VT1.mpg'))>0: #if there was a video file
                videoname = nlxfolder+'VT1.mpg'
                destfolder = nlxfolder
                
                commandline[1]=nlxfolder+'VT1.mpg' #add the right video name
                commandline[3]=nlxfolder+'VT1_DLC.mp4' #add the right output name
                alltext.append( ' '.join(commandline) )

with open(outputfolder+'convert_videos.sh', 'w') as f:
    for element in alltext:    
        f.write(element + "\n")
                        
#%% Check if the durations of the mpg and mp4 videos match (step 12)
#in some cases, DLC finishes and it seems like all is fine, but there is a part of the video
#not analyzed, and this causes problems with misalignment.
# Therefore, check all videos to see if the DLC output video is the same length as the original video

misalignment=[]
for animal in animals:

    params = get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):
        
        if project == 'neuropixels':
            folders, folder = concatenate_sessions(SESSION, params)
        elif project == 'tetrodes':
            folder = get_folder(params['Alldates'][SESSION], params, animal)
            folders = [folder]
        elif project == 'optogenetics':
            folder = "/mnt/fk-fileserver/Project_PFC_optogenetics/"+animal+"/"+params['Alldates'][SESSION]+"/"
            folders = [folder]
            
            
        for folder in folders: #there are only multiple folders in the case of neuropixels concatenated sessions
                        
            nlxfolder = folder+'neuralynx/' if project=='neuropixels' else folder

            if len(glob(nlxfolder+'*.mp4'))>0:
                videoname = nlxfolder+'VT1.mpg'
                cap = cv2. VideoCapture(videoname)
                frame_count = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
                
                videonameDLC = nlxfolder+'VT1_DLC.mp4'
                capDLC = cv2. VideoCapture(videonameDLC)
                frame_countDLC = int(capDLC. get(cv2. CAP_PROP_FRAME_COUNT))
                
                if frame_count != frame_countDLC:
                    print(animal+' '+params['Alldates'][SESSION]+': misalignment '+str(frame_count-frame_countDLC)+' frames')
                else:
                    print(animal+' '+params['Alldates'][SESSION]+': video OK')
                    
                misalignment.append( animal+' '+params['Alldates'][SESSION]+' '+str(frame_count-frame_countDLC) )    
                    
                
#%% Create analysis.json (step 13)
#making a dlc analysis json file for every video to be analyzed

outputname = "params_DLC_analysis" #have one analysis json ready in the folder
count=69

# Get all video names and make a params_analysis.json file for each video

with open(outputfolder+outputname+'.json') as json_file:
    data = json.load(json_file)

for animal in animals:

    params = get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):
        if project == 'neuropixels':
            folders, folder = concatenate_sessions(SESSION, params)
        elif project == 'tetrodes':
            folder = get_folder(params['Alldates'][SESSION], params, animal)
            folders = [folder]
        elif project == 'optogenetics':
            folder = "/mnt/fk-fileserver/Project_PFC_optogenetics/"+animal+"/"+params['Alldates'][SESSION]+"/"
            folders = [folder]

        for folder in folders: #there are only multiple folders in the case of neuropixels concatenated sessions
                        
            nlxfolder = folder+'neuralynx/' if project=='neuropixels' else folder
            
            if len(glob(nlxfolder+'VT1.mpg'))>0: #if there was a video file
                
                data['Videos']=[nlxfolder+'VT1_DLC.mp4'] #
                data['AnalyzeVideo']['destfolder'] = nlxfolder
                
                with open(outputfolder+outputname+str(count)+'.json', 'w') as outfile:
                    json.dump(data, outfile)
                count+=1    

#%% Check if the duration of the labeled DLC video matches (after analysis)
    
# Check all videos to see if the DLC output video is the same length as the original video

misalignment=[]
for animal in animals:

    params = get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):
        
        if project == 'neuropixels':
            folders, folder = concatenate_sessions(SESSION, params)
        elif project == 'tetrodes':
            folder = get_folder(params['Alldates'][SESSION], params, animal)
            folders = [folder]
        elif project == 'optogenetics':
            folder = "/mnt/fk-fileserver/Project_PFC_optogenetics/"+animal+"/"+params['Alldates'][SESSION]+"/"
            folders = [folder]
            
        for folder in folders: #there are only multiple folders in the case of neuropixels concatenated sessions
                        
            nlxfolder = folder+'neuralynx/' if project=='neuropixels' else folder

            if len(glob(nlxfolder+'*.mp4'))>0:
                videoname = nlxfolder+'VT1.mpg'
                cap = cv2.VideoCapture(videoname)
                frame_count = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))
                
                if len(glob(nlxfolder+'*labeled.mp4'))==0:
                    print(animal+' '+params['Alldates'][SESSION]+': no labeled video')
                    misalignment.append(  animal+' '+params['Alldates'][SESSION]+' no labeled video' )
                else:
                    videonameDLC = glob(nlxfolder+'*labeled.mp4')[0]
                    capDLC = cv2. VideoCapture(videonameDLC)
                    frame_countDLC = int(capDLC. get(cv2. CAP_PROP_FRAME_COUNT))
                        
                    if frame_count != frame_countDLC:
                        print(animal+' '+params['Alldates'][SESSION]+': misalignment '+str(frame_count-frame_countDLC)+' frames')
                    else:
                        print(animal+' '+params['Alldates'][SESSION]+': video OK')
                        
                    misalignment.append( animal+' '+params['Alldates'][SESSION]+' '+str(frame_count-frame_countDLC) ) 

                  
#%% adding the datafile-line to info.yaml in order use DLC output in localize (step 14)

for animal in animals:

    params = get_parameters(animal)
    
    for SESSION in range(len(params['Alldates'])):
        
        if project == 'neuropixels':
            folders, folder = concatenate_sessions(SESSION, params)
        elif project == 'tetrodes':
            folder = get_folder(params['Alldates'][SESSION], params, animal)
            folders = [folder]
        elif project == 'optogenetics':
            folder = "/mnt/fk-fileserver/Project_PFC_optogenetics/"+animal+"/"+params['Alldates'][SESSION]+"/"
            folders = [folder]
            
        for folder in folders: #there are only multiple folders in the case of neuropixels concatenated sessions
                        
            nlxfolder = folder+'neuralynx/' if project=='neuropixels' else folder
            
            if len(glob(nlxfolder+'*.h5'))>0 and len(glob(nlxfolder+'info.yaml'))>0: #if there is a Deeplabcut output file
                
                with open(nlxfolder+'info.yaml',"r") as file: 
                    info = yaml.safe_load(file)
                
                #add the datafile line to the info file
                info['datafile']=glob(nlxfolder+'*500000.h5')[0] #take the non filtered file
                
                with open(nlxfolder+'info.yaml', 'w') as file:
                    yaml.dump(info, file)                