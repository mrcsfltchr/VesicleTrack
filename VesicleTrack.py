############################################################################
# This class object controls the tracking of GUVs in fluorescent microscopy videos
############################################################################
# This software piggy back's off other projects I put together, MemDetect and TrapAnalysis, so I didn't have to reinvent methods
# I had already written before!
# However to make this project more useful I will isolate them in the future.


import sys
import numpy as np
import os


sys.path.append('/path/to/TrapAnalysis/project')
print(sys.path)
from trapanalysis import TrapGetter
from MemDetect.MemFret import load_tif
from matplotlib import pyplot as plt
from skimage.draw import circle

class VesicleTracker(TrapGetter):
    
    
    def __init__(self,exp_vid):
        
        
        super(self.__class__,self).__init__()
        
        # save reference to array of video data
        
        self.exp_vid = exp_vid
        
        # initialise data store for vesicle trajectories
        
        self.vesicle_trajectories = {}
        self.ceased_vesicle_trajectories = {}
        
        # set a threshold for how far a vesicle could have moved between two frames
        
        self.dist_threshold = 10
        
        
    def match_vesicles(self,t):
        
        # current tracked vesicles are stored in self.current vesicles
        # new detections should be temporarily in self.trap_positions
        # in this function we take a previous position and look for a
        # new detection nearby. Once all the old detections are updated 
        # we can add new vesicles being tracked.
        keys_to_delete = []
        
        for key in self.vesicle_trajectories.keys():
            
            last_ves_position = self.vesicle_trajectories[key][-1][1:]
            
            ret,new_position,self.trap_positions = self.match_vesicle(last_ves_position,self.trap_positions)
            
            if ret == -1:
                
                # if no new position is found the trajectory is removed from the active
                # dictionary of trajectories and saved in the final data store
                
                
                self.ceased_vesicle_trajectories[key] = self.vesicle_trajectories[key]
                keys_to_delete.append(key)
                
                
            else:
                # bundle position with time

                txy = np.concatenate((np.array([t]),new_position))

                
                print('txy: ',txy)
                self.vesicle_trajectories[key] = np.vstack((self.vesicle_trajectories[key],txy))
            
        self.deactivate_trajectories(keys_to_delete)
        
        
    def deactivate_trajectories(self,keys):
        
        for key in keys:
            self.vesicle_trajectories.pop(key)
         
    def start_new_trajectories(self,t):
        
        # now start new trajectories
        
        # the new detections that have been matched to old detections should have 
        # been deleted from the self.trap_positions list
        if t != 0:
            
            current_labels = np.array(list(self.vesicle_trajectories.keys()))
            
            if np.array(list(self.ceased_vesicle_trajectories.keys())).shape[0] > 0:
                
                current_labels = np.concatenate((current_labels,np.array(list(self.ceased_vesicle_trajectories.keys()))))
                
            # make integers
            current_labels = current_labels.astype(int)

            new_label = current_labels[-1]+1

        else:
            new_label = 0
            
            
        for position in self.trap_positions:
            
            self.vesicle_trajectories[str(new_label)] = np.array([np.concatenate((np.array([t]),position))])
            
            new_label += 1
            
        
            
            
    def match_vesicle(self,position, new_positions):
    
        # a simple identification between vesicles based on their closeness in the image plane
        
        distances = np.linalg.norm(new_positions - position,axis = 1)
        

        print('x_t :', new_positions)
        print('x_t-1 :', position)
        
        print('distances: ',distances)
        
        if distances[distances <= self.dist_threshold].shape[0] == 1:
            # vesicle has been tracked
            
            updated_position = new_positions[distances <= self.dist_threshold][0]
            print('updated_position: ',updated_position)
            new_positions = new_positions[distances > self.dist_threshold]
            
            return 0, updated_position, new_positions
        
        else:
            print('lost vesicle')
            return -1 , None, new_positions
            
            
            
    def extend_truncated_trajectories(self):
        
        # this method extends the trajectory from the time at which the track was lost to the end of the video by
        # repeating the last position until the end
        
        for key in self.ceased_vesicle_trajectories.keys():
            
            last_position = self.ceased_vesicle_trajectories[key][-1]
            
            t0 = last_position[0]
            
            last_position = last_position[1:]
            t = np.arange(t0, self.exp_vid.shape[0])
            
            print('last position: ',last_position)

            position = np.tile(last_position,(t.shape[0],1))
            print('tiled position: ',position)
            
            final_trajectory = np.vstack((t,position[:,0],position[:,1])).T
            
            self.ceased_vesicle_trajectories[key] = np.vstack((self.ceased_vesicle_trajectories[key],final_trajectory))
            
    def join_trajectories(self,vesicle1,vesicle2):
        
        # this method allows the user to spot two separate trajectories that really are tracks of the same vesicle
        # delayed in time and join them together.
        
        # vesicle1, vesicle2 should be string labels i.e. the keys in the dictionary self.ceased_vesicle_trajectories
        
        # validate the user input labels
        
        self.label_validation(vesicle1, self.ceased_vesicle_trajectories)
        self.label_validation(vesicle2, self.ceased_vesicle_trajectories)
        
        # reference the trajectories
        
        trajectory1 = self.ceased_vesicle_trajectories[vesicle1]
        trajectory2 = self.ceased_vesicle_trajectories[vesicle2]
        
        t_start = trajectory1[0][0]
        
        if trajectory2[0][0] > t_start:
            trajectory = np.vstack((trajectory1,trajectory2))
        else:
            trajectory = np.vstack((trajectory2,trajectory1))
            
            
    def label_validation(self,label, dictionary):
        
                
        try:
            assert label in dictionary.keys()
        except AssertionError:
            print(str(label)+ ' is not a label in the trajectory database')
            
            
            
         
    def label_overlays(self):
        
        self.label_positions = {}
        
        #assert bool(self.vesicle_trajectories)
        
        #assert bool(self.ceased_vesicle_trajectories)
        
        
        for key in self.vesicle_trajectories.keys():
            
            positions = self.vesicle_trajectories[key][:,1:]
            
            mean_position_1 = np.median(positions[:,0])
            
            mean_position_2 = np.median(positions[:,1])
            
            self.label_positions[key] = np.array([mean_position_1, mean_position_2])
            
        for key in self.ceased_vesicle_trajectories.keys():
            
            positions = self.ceased_vesicle_trajectories[key][:,1:]
            
            mean_position_1 = np.median(positions[:,0])
            
            mean_position_2 = np.median(positions[:,1])
            
            self.label_positions[key] = np.array([mean_position_1, mean_position_2])
            
    def remove_small_traces(self,min_trace = 50):
        

        # remove traces from ceased_vesicle_trajectories which are below a min_threshold
        pop_list = []
        
        for key in self.ceased_vesicle_trajectories.keys():
            track_len = len(self.ceased_vesicle_trajectories[key])
#             print(track_len)

            if track_len < 50:
                pop_list.append(key)

        for key in pop_list:
            self.ceased_vesicle_trajectories.pop(key)
            
            
        pop_list = []
        
        for key in self.vesicle_trajectories.keys():
            track_len = len(self.vesicle_trajectories[key])
#             print(track_len)

            if track_len < 50:
                pop_list.append(key)

        for key in pop_list:
            self.vesicle_trajectories.pop(key)

    def remove_traces(self,list_of_keys):
        
        # remove a specified list of traces. This is usually done after the user notices a bad trace in the plot
        
        for key in list_of_keys:
            
            if key in self.vesicle_trajectories.keys():
                self.vesicle_trajectories.pop(key)
                
            elif key in self.ceased_vesicle_trajectories.keys():
                self.ceased_vesicle_trajectories.pop(key)
                
                
            else:
                print(str(key) + ' is not in the database')
                
           
       
class FluorescenceExtractor(object):
    
    def __init__(self,radius,exp_vid):
        
        self.radius = radius # radius of disk to average intensity over
        
        self.exp_vid = exp_vid # optical experiment data
        
        self.vesicle_Is = {} # record of intensity time traces by vesicle
        
        
        
    def generate_time_series(self,vesicle_positions):
        
        # vesicle positions should be the dictionary, vesicle_trajectories, from the VesicleTracker object
        # regardless it should have the formate {label: np.array([[t_i,x_i,y_i],[t_i+1,x_i+1,y_i+1],...), label2: ...}
        
        for key in vesicle_positions.keys():
            
            vesicle_track = vesicle_positions[key]
            
            I = []
            for timed_position in vesicle_track:
                
                t = timed_position[0]
                position = timed_position[1:]
                
                I_t = self.get_mean_I(position,t)
                
                I.append([t,I_t])
                
            I = np.array(I)
                
            self.vesicle_Is[key] = I
            
    def get_mean_I(self,vesicle_position, t):
        
        ROI = circle(vesicle_position[0],vesicle_position[1],self.radius)
        
        print(ROI)
        internal_signal = self.exp_vid[t][ROI]
        
        return np.mean(internal_signal)
    
   
# Example use

if __name__ == '__main__':

    # create references to the experiment folder

    EXP_DIR = '/path/to/experimental/tif/videos'

    tif_name = 'TiffStack.tif'

    tif_path = os.path.join(EXP_DIR,tif_name)
    
    
    # load full video as numpy array

    images,_ = load_tif(tif_path)
    
    
    
    # the vesicle detector allows us to detect vesicles in a given frame.
    # We want to choose this frame to be the first frame at which point 
    # the GUVs seem to remain close by thereafter
    # this is manually input in this experiment

    # set experiment bounds

    t0 = 48

    t_end = images.shape[1]


    exp_vid = images[0][t0:t_end]
    print(exp_vid.shape)
    
    
    
    #Use vesicle detection class from the TrapAnalysis project

    vesicle_detector = VesicleTracker(exp_vid)

    for t in np.arange(0,exp_vid.shape[0]):

        vesicle_detector.get_vesicle_positions(exp_vid[t])
        vesicle_detector.remove_duplicates()

        if t >0:
            vesicle_detector.match_vesicles(t)

        vesicle_detector.start_new_trajectories(t)
        
        
        
    #vesicle_detector.vesicle_trajectories
    
    
    
    # example visualisation of trajectories
    plt.plot(vesicle_detector.vesicle_trajectories['4'][:,2],vesicle_detector.vesicle_trajectories['4'][:,1])
    plt.xlim([0,512])
    plt.ylim([0,512])