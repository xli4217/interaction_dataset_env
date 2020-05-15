#!/usr/bin/env python

import numpy as np
from rl_pipeline.configuration.configuration import Configuration

from future.utils import viewitems
import time
import os
import ctypes
import matplotlib.pyplot as plt
import matplotlib.animation as manimation

from examples.postdoc.risk_aware_rl.api.utils import map_vis_without_lanelet
from examples.postdoc.risk_aware_rl.api.utils import dataset_reader
from examples.postdoc.risk_aware_rl.api.utils import dataset_types
from examples.postdoc.risk_aware_rl.api.utils import map_vis_lanelet2
from examples.postdoc.risk_aware_rl.api.utils import tracks_vis_rl as tracks_vis
from examples.postdoc.risk_aware_rl.api.utils import dict_utils
from examples.postdoc.risk_aware_rl.api.utils.dataset_types import MotionState, Track

default_config = {
    # Common to all envs
    "seed": 10,
    "synchronous": True,
    "debug": False,
    "state_space": None,
    "action_space": None,
    "get_state": None,
    "get_reward": None,
    "reset": None,
    "is_done": None,
    # specific to this env
    "headless": False,
    "video_output_path": os.path.join(os.getcwd(), 'video.mp4'),
    "tracks_dir": "../recorded_trackfiles",
    "maps_dir": "../maps",
    "lanelet_map_ending": ".osm",
    "scenario_name": 'DR_CHN_Roundabout_LN',
    "track_file_prefix": "vehicle_tracks_",
    "track_file_ending": ".csv",
    "track_file_number": "000"
}

class InteractionDatasetEnv(object):

    def __init__(self, config=default_config, seed=None, logger=None):
        self.config = config
        self.name = "InteractionDatasetEnv"

        self.seed = seed
        self.set_seed(seed)
        
        self.logger = logger
        self.all_info = {}

        #### data file path ####
        self.lanelet_map_file =self.config['maps_dir'] + "/" + self.config['scenario_name'] + self.config['lanelet_map_ending']
        self.scenario_dir = self.config['tracks_dir'] + "/" + self.config['scenario_name']
        self.track_file_name = self.scenario_dir + "/" + self.config['track_file_prefix'] + str(self.config['track_file_number']) + self.config['track_file_ending']

        #### load tracks ####
        print("Loading tracks...")
        self.track_dictionary = dataset_reader.read_tracks(self.track_file_name)

        self.timestamp_min = self.config['timestamp_min']
        self.timestamp_max = self.config['timestamp_max']
        for key, track in dict_utils.get_item_iterator(self.track_dictionary):
            self.timestamp_min = min(self.timestamp_min, track.time_stamp_ms_first)
            self.timestamp_max = max(self.timestamp_max, track.time_stamp_ms_last)

        self.start_timestamp = self.timestamp_min
        self.timestamp = self.start_timestamp
        
        # storage of information
        self.patches_dict = dict()
        self.text_dict = dict()
        
        if not self.config['headless']:
            self.render()
            if self.config['video_output_path'] is not None:
                pass
                # FFMpegWriter = manimation.writers['ffmpeg']
                # metadata = dict(title='video', artist='Matplotlib', comment='video')
                # self.writer = FFMpegWriter(fps=15, metadata=metadata)
        else:
            self.fig = None
            self.axes = None
            self.title_text = None

        #### define ego ####
        self.ego_motion_state = MotionState(self.start_timestamp)
        self.ego_motion_state.x = 1000
        self.ego_motion_state.y = 1000
        self.ego_motion_state.vx = 0
        self.ego_motion_state.vy = 0
        self.ego_motion_state.psi_rad = 0
        
        self.ego_track = Track(-1)
        self.ego_track.agent_type = 'ego'
        self.ego_track.length = 4.9
        self.ego_track.width = 2.2
        self.ego_track.time_stamp_ms_first = self.timestamp_min
        self.ego_track.timestamp_ms_last = self.timestamp_max
        self.ego_track.motion_state = self.ego_motion_state

        # for rendering
        self.ego_rect = None
        self.ego_text = None
        
    def ego_track_to_ego_states(self, ego_track):
        ego_state = []

    def states_to_ego_track(self):
        pass
        
        
    def update(self,
               timestamp,
               track_dictionary,
               patches_dict,
               title_text=None,
               text_dict=None,
               ego_rect=None,
               ego_text=None,
               ego_track=None,
               fig=None,
               axes=None):
        
        start_time = time.time()
        
        # update text and tracks based on current timestamp
        assert(timestamp <= self.timestamp_max), "timestamp=%i" % timestamp
        assert(timestamp >= self.timestamp_min), "timestamp=%i" % timestamp
        assert(timestamp % dataset_types.DELTA_TIMESTAMP_MS == 0), "timestamp=%i" % timestamp
        
        self.ego_rect, self.ego_text = tracks_vis.update_objects_plot(timestamp,
                                                                      track_dictionary,
                                                                      patches_dict,
                                                                      text_dict,
                                                                      ego_rect,
                                                                      ego_text,
                                                                      ego_track,
                                                                      axes)
        
        if not self.config['headless']:
            title_text.set_text("\nts = {}".format(timestamp))
            end_time = time.time()
            diff_time = end_time - start_time
            fig.canvas.draw()
            plt.pause(max(0.001, dataset_types.DELTA_TIMESTAMP_MS / 1000. - diff_time))
            if self.config['video_output_path'] is not None:
                #self.writer.grab_frame()
                pass
            
    def render(self):
        plt.ion()

        self.fig, self.axes = plt.subplots(1,1)
        self.fig.canvas.set_window_title("Environment Visualization")

        # load and draw the lanelet2 map, either with or without the lanelet2 library
        self.lat_origin = 0.  # origin is necessary to correctly project the lat lon values in the osm file to the local
        self.lon_origin = 0.  # coordinates in which the tracks are provided; we decided to use (0|0) for every scenario
        print("Loading map...")

        map_vis_without_lanelet.draw_map_without_lanelet(self.lanelet_map_file, self.axes, self.lat_origin, self.lon_origin)
        
        # visualize tracks
        print("Plotting...")
        self.title_text = self.fig.suptitle("")
        self.playback_stopped = True
    
        
    def update_all_info(self):
        pass

    def get_info(self):
        return self.all_info

    def get_state(self, all_info):
        self.update_all_info()
        return np.random.rand(1)

    def get_reward(self, state=None, action=None, next_state=None, all_info={}):
        pass

    def is_done(self, state=None, action=None, next_state=None, all_info={}):
        pass

    def reset(self, **kwargs):
        pass

    def step(self, action: np.ndarray , axis=0):
        '''
        action[0] is the forward velocity in m/s
        action[1] is steering angle in rad
        '''

        action = np.clip(action, self.action_space['lower_bound'], self.action_space['upper_bound'])
        
        dt = dataset_types.DELTA_TIMESTAMP_MS / 1000
        
        self.timestamp += dataset_types.DELTA_TIMESTAMP_MS
        self.timestamp = min(self.timestamp, self.timestamp_max)
        self.timestamp = max(self.timestamp, self.timestamp_min)

        current_vel = np.array([self.ego_track.motion_state.vx,
                                self.ego_track.motion_state.vy])

        #### calculate new vel and pose ####
        x_vel = action[0] * np.cos(self.ego_track.motion_state.psi_rad)
        y_vel = action[0] * np.sin(self.ego_track.motion_state.psi_rad)
        steering_rate = (action[0] / self.ego_track.length) * np.tan(action[1])

        new_x = self.ego_track.motion_state.x + x_vel * dt
        new_y = self.ego_track.motion_state.y + y_vel * dt
        new_psi = self.ego_track.motion_state.psi_rad + steering_rate * dt
        
        #### update ego_track ####
        self.ego_track.motion_state.x = new_x
        self.ego_track.motion_state.y = new_y
        self.ego_track.motion_state.psi_rad = new_psi
        self.ego_track.motion_state.vx = x_vel
        self.ego_track.motion_state.vy = y_vel
        
        #### update ego_track ####
        self.update(self.timestamp,
                    self.track_dictionary,
                    self.patches_dict,
                    text_dict=self.text_dict,
                    title_text = self.title_text,
                    fig = self.fig,
                    ego_rect=self.ego_rect,
                    ego_text=self.ego_text,
                    ego_track=self.ego_track,
                    axes = self.axes)

            
    @property
    def state_space(self):
        return self.config.get('state_space')

    @property
    def action_space(self):
        return self.config.get('action_space')

    def teleop(self, cmd):
        pass

    def stop(self):
        pass

    def pause(self):
        pass

    def close(self):
        if not self.config['headless']:
            plt.ioff()

    def set_seed(self, seed):
        pass

if __name__ == "__main__":
    config = {
        # Common to all envs
        "seed": 10,
        "synchronous": True,
        "debug": False,
        "state_space": None,
        "action_space": {'type': 'float', 'shape': (2, ), 'upper_bound': [60, np.pi/2], 'lower_bound': [0, -np.pi/2]},
        "get_state": None,
        "get_reward": None,
        "reset": None,
        "is_done": None,
        "headless": False,
        "video_output_path": os.path.join(os.getcwd(), 'video.mp4'),
        # specific to this env
        "tracks_dir": "../recorded_trackfiles",
        "maps_dir": "../maps",
        "lanelet_map_ending": ".osm",
        "scenario_name": 'DR_CHN_Roundabout_LN',
        "track_file_prefix": "vehicle_tracks_",
        "track_file_ending": ".csv",
        "track_file_number": "000",
        "timestamp_min": 1e9,
        "timestamp_max": 0
    }

    cls = InteractionDatasetEnv(config)
    
    for i in range(1000):
        # print(i)
        cls.step(action=np.array([20, np.pi/6]))
        