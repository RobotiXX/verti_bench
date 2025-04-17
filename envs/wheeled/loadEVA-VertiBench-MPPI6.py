import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os
import random
import math
import pickle
import time

from verti_bench.envs.utils.utils import SetChronoDataDirectories
import pychrono.sensor as sens
from verti_bench.envs.utils.asset_utils import *

from PIL import Image, ImageDraw
import os
import shutil
import matplotlib.pyplot as plt
import copy
import yaml
import logging
import heapq
from scipy.ndimage import binary_dilation
import glob
import csv
import json
from scipy.special import comb

import torch
import cv2
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.parallel as parallel
from collections import defaultdict
from scipy.interpolate import PchipInterpolator
from scipy.signal import savgol_filter

from models import TAL, ElevMapEncDec, PatchDecoder
from utilities import utils
from Grid import MapProcessor
from traversabilityEstimator import TravEstimator
from MPPIModels import KrModel

MAX_VEL = 0.6
MIN_VEL = 0.45
wm_vct = False

class mppi_planner:
    def __init__(self, T, K, sigma=[0.5, 0.1], _lambda=0.5):
        #Robot Limits
        self.max_vel = MAX_VEL
        self.min_vel = MIN_VEL
        self.max_del = 1.0
        self.min_del = -1.0
        self.robot_length = 3.4
        self.min_speed = MIN_VEL
        self.max_speed = MAX_VEL
        self.min_steer_angle = -1.0
        self.max_steer_angle = 1.0

        #External class definations 
        self.mp = MapProcessor()                                                                    # Only for crawler
        self.util = utils()                                                                         # Utility class
        #variables 
        self.goal_tensor = None                                                                     # Goal tensor
        self.lasttime = None                                                                        # Last time
        model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "TAL_13_14_14.torch")                                     # Default Model path                                               # Default Model path
        enc_dec_path = "rock_LN_TH_03-07-20-08.torch"                                                    # Default Model path
        
        elev_map_encoder = ElevMapEncDec()
        elev_map_encoder = elev_map_encoder.cuda()

        elev_map_decoder = PatchDecoder(elev_map_encoder.map_encode)
        elev_map_decoder = elev_map_decoder.cuda()

        self.model = TAL(elev_map_encoder.map_encode, elev_map_decoder)
        self.model = self.model.cuda()
        
        state_dict = torch.load(model_path).state_dict()
        self.model.load_state_dict(state_dict)                                                      # Motion model
        self.model.eval()                                                                           # Model ideally runs faster in eval mode
        self.dtype = torch.float32                                                                  # Data type
        
        #------MPPI variables and constants----
        #Parameters
        self.T = T                                                                                  # Length of rollout horizon
        self.K = K                                                                                  # Number of sample rollouts
        self.dt = 1
        self._lambda = float(_lambda)                                                               # 
        self.sigma = torch.Tensor(sigma).type(self.dtype).expand(self.T, self.K, 2).cuda()                 # (T, K, 2)
        self.inv_sigma = 1/ self.sigma[0,0,:]                                                       # (2, )

        with open('/home/chenhui/Documents/verti_bench/baselines/stats_rock.pickle', 'rb') as f:
            scale = pickle.load(f)
        
        self.scale_state = scale['pose_dot']
        self.offset_scale = scale['map_offset']

        self.robot_pose = None                                                                    # Robot pose
        self.noise = torch.Tensor(self.T, self.K, 2).type(self.dtype).cuda()                               # (T,K,2)
        self.poses = torch.Tensor(self.K, self.T, 6).type(self.dtype).cuda()                               # (K,T,6)
        self.fpose = torch.Tensor(self.K, 6).type(self.dtype)                                    # (K,6)
        self.last_pose = None
        self.at_goal = True
        self.curr_pose = None
        self.robot_velcities = torch.zeros(6).type(self.dtype) 

        self.pose_dot = torch.zeros(6).type(self.dtype)
        self.last_t = 0
        self.map_origin = torch.Tensor([-64.5, -64.5]).type(self.dtype)
        self.map_resolution = 1
        self.current_trajectories = None

        #Cost variables
        self.running_cost = torch.zeros(self.K).type(self.dtype)                                    # (K, )
        self.pose_cost = torch.Tensor(self.K).type(self.dtype)                                      # (K, )
        self.bounds_check = torch.Tensor(self.K).type(self.dtype)                                   # (K, )
        self.height_check = torch.Tensor(self.K).type(self.dtype)                                   # (K, )#ony for crawler
        self.ctrl_cost = torch.Tensor(self.K, 2).type(self.dtype)                                   # (K,2)
        self.ctrl_change = torch.Tensor(self.T,2).type(self.dtype)                                  # (T,2)
        self.euclidian_distance = torch.Tensor(self.K).type(self.dtype)                             # (K, )
        self.dist_to_goal = torch.Tensor(self.K, 6).type(self.dtype)                                # (K, )
        self.dtg_w_init_pose = torch.Tensor(self.K, 6).type(self.dtype)                             # (K, )

        self.map_embedding = None
        self.recent_controls = np.zeros((3,2))
        self.control_i = 0
        self.msgid = 0
        self.speed = 0
        self.steering_angle = 0
        self.prev_ctrl = None
        self.ctrl = torch.zeros((self.T, 2)).cuda()  # Initial speed = 5.0 m/s
        self.rate_ctrl = 0
        self.cont_ctrl = True

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.m_idx = [1,    3 ,     5,      7,      8,      9,      10,    11,        12,         13]
        # Weights for the 8 traversability maps
        self.weights = torch.tensor([0.15, 0.15, 0.1, 0.1, 0.2, 0.2, 0.15, 0.15, 0.3, 0.3 ], dtype=self.dtype).cuda().unsqueeze(-1).unsqueeze(-1)
        # Initialize the traversability model
        self.model.to(self.device)
        self.traversability_model = TravEstimator(output_dim=(321,260)).to(self.device)
        self.traversability_model.load_state_dict(torch.load('/home/chenhui/Documents/verti_bench/baselines/best_traversability_model.pth'))
        self.traversability_model.eval()  # Set to evaluation mode
        self.traversability_model.requires_grad_(False)  # Disable gradient computation
        self.traversability_mask = None
        self.trav_map_combined = None
        self.traversability_model(torch.zeros(2, 1, 160, 130).to(self.device))

        #6DOF model Initialization
        load_dir = '/home/chenhui/Documents/verti_bench/baselines/Kr_model_50.pth'
        checkpoint = torch.load(load_dir)

        self.kr_model = KrModel(5)
        self.kr_model.load_state_dict(checkpoint, strict=True)
        self.kr_model.eval()
        self.kr_model.to(self.device)
        print("Loading:", load_dir)                                                               
        print("Model:\n",self.kr_model)
        print("Torch Datatype:", self.dtype)

        self.times_mppi_called = 0
        self.images_for_poses = None

        with open('/home/chenhui/Documents/verti_bench/baselines/stats_kr_bench.pkl', 'rb') as f:
            self.stats = pickle.load(f)
        
        self.stats['cmd_vel_mean'] = torch.tensor(self.stats['cmd_vel_mean']).cuda().float()
        self.stats['cmd_vel_std'] = torch.tensor(self.stats['cmd_vel_std']).cuda().float()
        self.stats['velocity_mean'] = torch.tensor(self.stats['velocity_mean']).cuda().float()
        self.stats['velocity_std'] = torch.tensor(self.stats['velocity_std']).cuda().float()
        self.stats['res_roll_pitch_mean'] = torch.tensor(self.stats['res_roll_pitch_mean']).cuda().float()
        self.stats['res_roll_pitch_std'] = torch.tensor(self.stats['res_roll_pitch_std']).cuda().float()
        self.stats['roll_pitch_yaw_mean'] = torch.tensor(self.stats['roll_pitch_yaw_mean']).cuda().float()
        self.stats['roll_pitch_yaw_std'] = torch.tensor(self.stats['roll_pitch_yaw_std']).cuda().float()


    def gridMap_callback(self, m_vehicle, m_vehicle_pos):
        """
        Callback to process the elevation map and update the map embedding.
        :param bmp_elevation_map: Ground truth elevation map as a [129, 129] BMP array
        """
        if self.robot_pose is not None:

            # Crop [29, 29] region around the vehicle
            crop_size = 29
            cropped_map, _ = get_cropped_map(m_vehicle, (m_vehicle_pos.x, m_vehicle_pos.y, m_vehicle_pos.z), crop_size, 5)
            input_size = (360, 360)
            resized_map = cv2.resize(cropped_map, input_size, interpolation=cv2.INTER_LANCZOS4)

            # Debugging: Check cropped and resized map values
            # print(f"Cropped map min: {cropped_map.min()}, max: {cropped_map.max()}")
            # print(f"Resized map min: {resized_map.min()}, max: {resized_map.max()}")

            # Dynamically rescale elevation values to [0, 0.5]
            map_min = obstacle_array.min()
            map_max = obstacle_array.max()
            resized_map = (resized_map - map_min) / (map_max - map_min) * 0.5  # Normalize to [0, 0.5]

            # Debugging: Check normalized map values
            # print(f"Normalized map min: {resized_map.min()}, max: {resized_map.max()}")
            # cv2.imshow("Elevation Map", resized_map)
            # cv2.waitKey(1)
            
            # Convert to tensor
            map_d = torch.tensor(resized_map, dtype=self.dtype).cuda().unsqueeze(dim=0).unsqueeze(dim=0)

            # Debugging: Print the tensor before processing
            # print(f"map_d shape: {map_d.shape}, dtype: {map_d.dtype}, min: {map_d.min().item()}, max: {map_d.max().item()}")
            
            # Process map through TAL model
            with torch.no_grad():
                # Generate traversability maps
                # map_t = F.center_crop(map_d, (320, 260))
                # map_t = F.gaussian_blur(map_t, kernel_size=3, sigma=0.2)
                # map_t = F.resize(map_t, (160, 130))
                # start_time = time.time()
                # traversability_maps = self.traversability_model(map_t).squeeze()  # Shape: [14, H, W]
                self.map_embedding = self.model.process_map(map_d).repeat(self.K, 1, 1, 1).cuda()
                # print("Map embedding updated:", self.map_embedding.shape)
                # print(f"Map embedding min: {self.map_embedding.min().item()} max: {self.map_embedding.max().item()}")
                
                # Combine 14 maps into a single map
                # You can adjust this combination method based on your specific requirements
                # traversability_maps = traversability_maps[self.m_idx]
                # traversability_maps = traversability_maps * self.weights
                # combined_traversability_map = torch.sum(traversability_maps, dim=0).squeeze()  # Shape: [1, H, W]
                
                # elevmap_n = F.center_crop(map_d, (320, 260)).squeeze()
                # combined_traversability_map[elevmap_n > 0.25] = combined_traversability_map.max()*1.5
                # self.traversability_mask = combined_traversability_map
                # print("Traversability mask generated")
                # self.trav_map_combined = combined_traversability_map
                # cv2.imshow("Traversability Map", self.trav_map_combined.cpu().numpy())
                # cv2.waitKey(1)

        else:
            print("Warning: robot_pose is not set. Cannot process elevation map.")

    def goal_cb(self, local_goal):
        """
        Callback to set the goal directly using Chrono's goal representation.
        :param chrono_goal: Goal as a Chrono `ChVectorD` (x, y, z)
        """
        # print("Goal received")
        # Convert Chrono goal to PyTorch tensor
        goal_tensor_new = torch.Tensor([
            local_goal[0],  # x
            local_goal[1],  # y
            1.6,  # z
            0,            # roll (default, not provided by Chrono goal)
            0,            # pitch (default, not provided by Chrono goal)
            0             # yaw (default, not provided by Chrono goal)
        ]).type(self.dtype)

        # Set the goal tensor and update status
        self.goal_tensor = goal_tensor_new
        self.at_goal = False

        # Debugging information
        # print("Setting local goal:", local_goal)
        # print("SETTING Goal: ", self.goal_tensor.cpu().numpy())

    def cost(self, pose, goal, ctrl, noise, roll, pitch, t, initial_pose):
        self.fpose.copy_(pose)
        self.dist_to_goal.copy_(self.fpose).sub_(goal)
        self.dtg_w_init_pose.copy_(initial_pose).sub_(goal) 
        self.dist_to_goal[:,3] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,3])
        self.dist_to_goal[:,4] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,4])
        self.dist_to_goal[:,5] = self.util.clamp_angle_tensor_(self.dist_to_goal[:,5])
        
        xy_to_goal = self.dist_to_goal[:,:2]
        self.euclidian_distance = torch.norm(xy_to_goal, p=2, dim=1)
        euclidian_distance_squared = self.euclidian_distance.pow(2)
        theta_distance_to_goal = self.dist_to_goal[:,5] % (2*np.pi)

        xy_to_goal_init = self.dtg_w_init_pose[:,:2]
        euclidian_distance_squared_init = torch.norm(xy_to_goal_init, p=2, dim=1).pow(2)
        # pdb.set_trace()

        self.ctrl_cost.copy_(ctrl).mul_(self._lambda).mul_(self.inv_sigma.cpu()).mul_(noise.cpu())
        running_cost_temp = self.ctrl_cost.abs_().sum(dim=1)
        self.running_cost.copy_(running_cost_temp)
        
        # pdb.set_trace()
        self.running_cost.add_(euclidian_distance_squared).add((theta_distance_to_goal * 30)/euclidian_distance_squared_init[0]).add_(roll.cpu()*1.0).add_(pitch.cpu()*0.5) #.add_(low_traversability_penalty*10)#.add_(r*2).add_(p)#.add_(theta_distance_to_goal)#.add_(p)#.add_(stuck_cost).add_(p)

    def get_control(self):
        # Apply the first control values, and shift your control trajectory
        run_ctrl = self.ctrl[0].clone()

        # shift all controls forward by 1, with last control replicated
        self.ctrl = torch.roll(self.ctrl, shifts=-1, dims=0)
        #self.ctrl[:-1] = self.ctrl[1:]
        return run_ctrl

    def mppi(self, init_pose, init_velocities, init_angles):
        t0 = time.time()
        self.times_mppi_called += 1
        dt = self.dt

        self.running_cost.zero_()                                                  # Zero running cost
        cur_pose = init_pose.repeat(self.K, 1).cuda()                                  # Repeat the init pose to sample size 
        init_pose_rep = cur_pose.clone() 
        pure_pose = cur_pose.clone()

        ctrl_temp = torch.zeros(self.K, 2).cuda()                                  # Initialize control temp
        nn_velocities = init_velocities.repeat(self.K, 1, 1).cuda().type(self.dtype)  # Repeat the init velocities to sample size
        nn_angles = init_angles.repeat(self.K, 1).cuda().type(self.dtype)            # Repeat the init angles to sample size

        speed = init_velocities[0].repeat(self.K, 1).cuda()                          # Repeat the init speed to sample size
        torch.normal(0, self.sigma, out=self.noise)                                 # Generate noise based on the sigma

        # Loop the forward calculation till the horizon
        for t in range(self.T):
            ctrl_temp = (self.ctrl[t] + self.noise[t])                               # Add noise to previous control input 
            ctrl_temp[:, 0].clamp_(self.min_speed, self.max_speed)                   # Clamp control velocity
            ctrl_temp[:, 1].clamp_(self.min_steer_angle, self.max_steer_angle)       # Clamp control steering
            # pdb.set_trace()
            delta_pose = mppi_planner.ackermann_bench(ctrl_temp[:, 0].unsqueeze(-1).cuda(), ctrl_temp[:, 1].unsqueeze(-1).cuda(), self.robot_length, dt) # Calculate the change in pose
            cur_pose = mppi_planner.to_world_torch(init_pose_rep, delta_pose.cuda())    # Convert the pose to world frame
            self.poses[:, t, :] = cur_pose.clone()                                  # Update the poses
            
            if torch.isnan(cur_pose).any():
                print("NaN is in poses")

            if self.times_mppi_called == 1 or (t == 0 and self.times_mppi_called % 50 == 0):
                self.images_for_poses = (get_cropped_map_torch(self.poses[:, t, :], 8, self.K).unsqueeze(1) / 255.0).to(torch.float32)

            nn_input_norm = (nn_velocities - self.stats['velocity_mean']) / (self.stats['velocity_std'] + 0.000006)
            nn_angles_norm = (nn_angles -self.stats['roll_pitch_yaw_mean']) / (self.stats['roll_pitch_yaw_std'] + 0.000006)
            ctrl_temp_norm = (ctrl_temp - self.stats['cmd_vel_mean']) / (self.stats['cmd_vel_std'] + 0.000006) 
            # Model query for next pose caalculation
            with torch.no_grad():
                model_otpt = self.kr_model(nn_input_norm.squeeze(1), nn_angles_norm, self.images_for_poses, ctrl_temp_norm)
            
            roll = (model_otpt[:, 0] * self.stats['res_roll_pitch_std'][0] + 0.000006) + self.stats['res_roll_pitch_mean'][0]
            pitch = (model_otpt[:, 1] * self.stats['res_roll_pitch_std'][1] + 0.000006) + self.stats['res_roll_pitch_mean'][1]

            self.poses[:, t, [3]] = roll.unsqueeze(1)
            self.poses[:, t, [4]] = pitch.unsqueeze(1)
            change_in_robot = mppi_planner.to_robot_torch(init_pose_rep, self.poses[:, t, :])
            init_pose_rep = self.poses[:, t, :].clone()

            nn_velocities = change_in_robot / dt
                       
            # Add to self poses
            # nn_angles[:, 1] = pitch
            # nn_angles[:, 0] = roll
            nn_angles[:, 2] = self.poses[:, t, 5]

            # Calculate the cost for each pose
            self.cost(self.poses[:, t, :], self.goal_tensor, self.ctrl[t], self.noise[t],roll, pitch, t, pure_pose)

        # MPPI weighing 
        self.running_cost -= torch.min(self.running_cost)
        self.running_cost /= -self._lambda
        torch.exp(self.running_cost, out=self.running_cost)
        weights = self.running_cost / torch.sum(self.running_cost)

        weights = weights.unsqueeze(1).expand(self.T, self.K, 2)
        weights_temp = weights.mul(self.noise.cpu())
        self.ctrl_change.copy_(weights_temp.sum(dim=1))
        # pdb.set_trace()
        self.ctrl += self.ctrl_change.cuda()

        self.ctrl[:,0].clamp_(self.min_speed, self.max_speed)
        self.ctrl[:,1].clamp_(self.min_steer_angle, self.max_steer_angle)
        
        # print("MPPI: %4.5f ms" % ((time.time()-t0)*1000.0))
        return self.poses
    
    def odom_cb(self, vehicle, m_system):
        """
        Callback to update the robot's pose using Chrono's simulation data.
        :param chrono_pose: Chrono `ChVectorD` representing the robot's position (x, y, z)
        :param chrono_orientation: Chrono `ChQuaternionD` representing the robot's orientation
        """
        timenow = m_system.GetChTime() # Replace with your simulation time variable

        pos = vehicle.GetVehicle().GetPos()
        euler_angles = vehicle.GetVehicle().GetRot().Q_to_Euler123() #Global coordinate
        roll = euler_angles.x
        pitch = euler_angles.y
        yaw = euler_angles.z

        # Update current pose first
        self.robot_pose = torch.Tensor([
            pos.x, pos.y, pos.z,
            roll, pitch, yaw    
        ]).type(self.dtype)
        
        self.curr_pose = torch.Tensor([
                    pos.x, pos.y, pos.z,
                    roll, pitch, yaw
                ]).type(self.dtype)
        
        self.velocities = torch.Tensor([
                vehicle.GetVehicle().GetSpeed(), 0.0, 0.0,  # vel_x, vel_y, vel_z
                vehicle.GetVehicle().GetRollRate(),
                vehicle.GetVehicle().GetPitchRate(),
                vehicle.GetVehicle().GetYawRate()]).type(self.dtype)

        # Then handle initialization
        if self.last_pose is None:
            self.last_pose = torch.Tensor([
                pos.x, pos.y, pos.z,
                roll, pitch, yaw
            ]).type(self.dtype)
            self.lasttime = timenow
            return
        
        

        difference_from_goal = np.sqrt(((self.curr_pose.cpu())[0] - (self.goal_tensor.cpu())[0])**2 + ((self.curr_pose.cpu())[1] - (self.goal_tensor.cpu())[1])**2)
        # Adjust velocity limits based on distance to goal
        if difference_from_goal < 2:
            self.min_vel = -1  # Adjusted limits
            self.max_vel = 1
        else:
            self.min_vel = MIN_VEL
            self.max_vel = MAX_VEL

        # Update pose_dot if enough time has elapsed
        t_diff = timenow - self.last_t
        if t_diff >= 1:
            self.pose_dot = self.curr_pose - self.last_pose
            self.pose_dot[5] = self.util.clamp_angle(self.pose_dot[5])  # Clamp yaw difference
            self.last_pose = self.curr_pose
            self.last_t = timenow

    def mppi_cb(self) :
        if self.curr_pose is None or self.goal_tensor is None or self.velocities is None:
            return
        
        roll, pitch, yaw = self.curr_pose[3], self.curr_pose[4], self.curr_pose[5]
        cs_angles = torch.Tensor([roll, pitch, yaw]).type(self.dtype)
        poses = self.mppi(self.curr_pose, self.velocities, cs_angles) 

        run_ctrl = None
        if not self.cont_ctrl:
            run_ctrl = self.get_control().cpu().numpy()
            self.recent_controls[self.control_i] = run_ctrl
            self.control_i = (self.control_i + 1) % self.recent_controls.shape[0]
            pub_control = self.recent_controls.mean(0)
            self.speed = pub_control[0]
            self.steering_angle = pub_control[1]

    def send_controls(self):
        """
        Sends control commands to the Chrono vehicle.
        :param vehicle: The Chrono HMMWV_Reduced vehicle object
        :param delta_steer: The steering angle from MPPI
        """
        if not self.at_goal:
            if self.cont_ctrl:  # Check if continuous control is enabled
                run_ctrl = self.get_control()
                if self.prev_ctrl is None:
                    self.prev_ctrl = run_ctrl

                # Update speed and steering
                speed = run_ctrl[0]  # Throttle value
                steer = -float(run_ctrl[1])  # Ensure steer is a float
                self.prev_ctrl = (speed, steer)
            else:
                speed = self.speed
                steer = -float(self.steering_angle)  # Ensure delta_steer is a float
        else:
            speed = 0.0
            steer = 0.0

        return speed, steer

    def check_at_goal(self):
        if self.at_goal or self.last_pose is None:
            #print ("Already at goal")
            return

        # TODO: if close to the goal, there's nothing to do
        XY_THRESHOLD = 4
        THETA_THRESHOLD = 0.35 # about 10 degrees
        difference_from_goal = (self.last_pose - self.goal_tensor).abs()
        xy_distance_to_goal = difference_from_goal[:2].norm()
        theta_distance_to_goal = difference_from_goal[5] % (2 * np.pi)
        if xy_distance_to_goal < XY_THRESHOLD and theta_distance_to_goal < THETA_THRESHOLD:
            print ('Goal achieved')
            self.at_goal = True
            self.speed = 0
            self.steering_angle = 0
            return

    @staticmethod
    #function to find the change in robot frame given two poses
    def to_robot_torch(pose_batch1, pose_batch2):
        if pose_batch1.shape != pose_batch2.shape:
            raise ValueError("Input tensors must have same shape")

        if pose_batch1.shape[-1] != 6:
            raise ValueError(f"Input tensors must have last dim equal to 6 for SE3, got {pose_batch1.shape[-1]}")

        batch_size = pose_batch1.shape[0]
        ones = torch.ones_like(pose_batch2[:, 0])
        transform = torch.zeros_like(pose_batch1)
        T1 = torch.zeros((batch_size, 4, 4), device=pose_batch1.device, dtype=pose_batch1.dtype)
        T2 = torch.zeros((batch_size, 4, 4), device=pose_batch2.device, dtype=pose_batch2.dtype)

        T1[:, :3, :3] = mppi_planner.euler_to_rotation_matrix(pose_batch1[:, 3:])
        T2[:, :3, :3] = mppi_planner.euler_to_rotation_matrix(pose_batch2[:, 3:])
        T1[:, :3,  3] = pose_batch1[:, :3]
        T2[:, :3,  3] = pose_batch2[:, :3]
        T1[:,  3,  3] = 1
        T2[:,  3,  3] = 1

        T1_inv = torch.inverse(T1)
        tf3_mat = torch.matmul(T2, T1_inv)

        transform[:, :3] = torch.matmul(T1_inv, torch.cat((pose_batch2[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()[:, :3]
        transform[:, 3:] = mppi_planner.extract_euler_angles_from_se3_batch(tf3_mat)

        return transform
         
    @staticmethod 
    def to_world_torch(Robot_frame, P_relative):
        SE3 = True

        if not isinstance(Robot_frame, torch.Tensor):
            Robot_frame = torch.tensor(Robot_frame, dtype=torch.float32)
        if not isinstance(P_relative, torch.Tensor):
            P_relative = torch.tensor(P_relative, dtype=torch.float32)

        if len(Robot_frame.shape) == 1:
            Robot_frame = Robot_frame.unsqueeze(0)

        if len(P_relative.shape) == 1:
            P_relative = P_relative.unsqueeze(0)
    
        if len(Robot_frame.shape) > 2 or len(P_relative.shape) > 2:
            raise ValueError(f"Input must be 1D for  unbatched and 2D for batched got input dimensions {Robot_frame.shape} and {P_relative.shape}")

        # pdb.set_trace()
        if Robot_frame.shape != P_relative.shape:
            raise ValueError("Input tensors must have same shape")
        
        if Robot_frame.shape[-1] != 6 and Robot_frame.shape[-1] != 3:
            raise ValueError(f"Input tensors must have last dim equal to 6 for SE3 and 3 for SE2 got {Robot_frame.shape[-1]}")
        
        if Robot_frame.shape[-1] == 3:
            SE3 = False
            Robot_frame_ = torch.zeros((Robot_frame.shape[0], 6), device=Robot_frame.device, dtype=Robot_frame.dtype)
            Robot_frame_[:, [0,1,5]] = Robot_frame
            Robot_frame = Robot_frame_
            P_relative_ = torch.zeros((P_relative.shape[0], 6), device=P_relative.device, dtype=P_relative.dtype)
            P_relative_[:, [0,1,5]] = P_relative
            P_relative = P_relative_
            
        """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
        batch_size = Robot_frame.shape[0]
        ones = torch.ones_like(P_relative[:, 0])
        transform = torch.zeros_like(Robot_frame)
        T1 = torch.zeros((batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype)
        T2 = torch.zeros((batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype)

        R1 = mppi_planner.euler_to_rotation_matrix(Robot_frame[:, 3:])
        R2 = mppi_planner.euler_to_rotation_matrix(P_relative[:, 3:])
        
        T1[:, :3, :3] = R1
        T2[:, :3, :3] = R2
        T1[:, :3,  3] = Robot_frame[:, :3]
        T2[:, :3,  3] = P_relative[:, :3]
        T1[:,  3,  3] = 1
        T2[:,  3,  3] = 1 

        T_tf = torch.matmul(T2, T1)
        transform[:, :3] = torch.matmul(T1, torch.cat((P_relative[:, :3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze(dim=2)[:, :3]
        transform[:, 3:] = mppi_planner.extract_euler_angles_from_se3_batch(T_tf)

        if not SE3:
            transform = transform[:, [0,1,5]]

        return transform
           
    @staticmethod 
    def euler_to_rotation_matrix(euler_angles):
        """ Convert Euler angles to a rotation matrix """
        # Compute sin and cos for Euler angles
        cos = torch.cos(euler_angles)
        sin = torch.sin(euler_angles)
        zero = torch.zeros_like(euler_angles[:, 0])
        one = torch.ones_like(euler_angles[:, 0])
        # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
        R_x = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], dim=1).view(-1, 3, 3)
        R_y = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], dim=1).view(-1, 3, 3)
        R_z = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], dim=1).view(-1, 3, 3)

        return torch.matmul(torch.matmul(R_z, R_y), R_x)

    @staticmethod 
    def extract_euler_angles_from_se3_batch(tf3_matx):
        # Validate input shape
        if tf3_matx.shape[1:] != (4, 4):
            raise ValueError("Input tensor must have shape (batch, 4, 4)")

        # Extract rotation matrices
        rotation_matrices = tf3_matx[:, :3, :3]

        # Initialize tensor to hold Euler angles
        batch_size = tf3_matx.shape[0]
        euler_angles = torch.zeros((batch_size, 3), device=tf3_matx.device, dtype=tf3_matx.dtype)

        # Compute Euler angles
        euler_angles[:, 0] = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Roll
        euler_angles[:, 1] = torch.atan2(-rotation_matrices[:, 2, 0], torch.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2))  # Pitch
        euler_angles[:, 2] = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Yaw

        return euler_angles

    @staticmethod
    def ackermann_bench(throttle, steering, wheel_base=2.85, dt=0.1):
        if not isinstance(throttle, torch.Tensor):
            throttle = torch.tensor(throttle, dtype=torch.float32)
        if not isinstance(steering, torch.Tensor):
            steering = torch.tensor(steering, dtype=torch.float32)
        if not isinstance(wheel_base, torch.Tensor):
            wheel_base = torch.tensor(wheel_base, dtype=torch.float32)
        if not isinstance(dt, torch.Tensor):
            dt = torch.tensor(dt, dtype=torch.float32)
        if throttle.shape != steering.shape:
            raise ValueError("throttle and steering must have the same shape")
        if len(throttle.shape) == 0:
            throttle = throttle.unsqueeze(0)
        
        deltaPose = torch.zeros(throttle.shape[0], 6, dtype=torch.float32)

        dtheta = (throttle / wheel_base) * torch.tan(steering) * dt
        dx = throttle * torch.cos(dtheta) * dt
        dy = throttle * torch.sin(dtheta) * dt
        deltaPose[:, 0], deltaPose[:, 1], deltaPose[:, 5] = dx.squeeze(), dy.squeeze(), dtheta.squeeze()

        return deltaPose.squeeze()
    
class SCMParameters:
    def __init__(self):
        self.Bekker_Kphi = 0    # Kphi, frictional modulus in Bekker model
        self.Bekker_Kc = 0      # Kc, cohesive modulus in Bekker model
        self.Bekker_n = 0       # n, exponent of sinkage in Bekker model (usually 0.6...1.8)
        self.Mohr_cohesion = 0  # Cohesion in, Pa, for shear failure
        self.Mohr_friction = 0  # Friction angle (in degrees!), for shear failure
        self.Janosi_shear = 0   # J , shear parameter, in meters, in Janosi-Hanamoto formula (usually few mm or cm)
        self.elastic_K = 0      # elastic stiffness K (must be > Kphi very high values gives the original SCM model)
        self.damping_R = 0      # vertical damping R, per unit area (vertical speed proportional, it is zero in original SCM model)

    # Set the parameters of the terrain
    def SetParameters(self, terrain):
        terrain.SetSoilParameters(
            self.Bekker_Kphi,    # Bekker Kphi
            self.Bekker_Kc,      # Bekker Kc
            self.Bekker_n,       # Bekker n exponent
            self.Mohr_cohesion,  # Mohr cohesive limit (Pa)
            self.Mohr_friction,  # Mohr friction limit (degrees)
            self.Janosi_shear,   # Janosi shear coefficient (m)
            self.elastic_K,      # Elastic stiffness (Pa/m), before plastic yield, must be > Kphi
            self.damping_R)      # Damping (Pa s/m), proportional to negative vertical speed (optional)

    # Soft default parameters
    def InitializeParametersAsSoft(self): # snow
        self.Bekker_Kphi = 0.2e6
        self.Bekker_Kc = 0
        self.Bekker_n = 1.1
        self.Mohr_cohesion = 0
        self.Mohr_friction = 30
        self.Janosi_shear = 0.01
        self.elastic_K = 4e7
        self.damping_R = 3e4

    # Middle default parameters
    def InitializeParametersAsMid(self): # mud
        self.Bekker_Kphi = 2e6
        self.Bekker_Kc = 0
        self.Bekker_n = 1.1
        self.Mohr_cohesion = 0
        self.Mohr_friction = 30
        self.Janosi_shear = 0.01
        self.elastic_K = 2e8
        self.damping_R = 3e4
    
    # Hard default parameters
    def InitializeParametersAsHard(self): # sand
        self.Bekker_Kphi = 5301e3
        self.Bekker_Kc = 102e3
        self.Bekker_n = 0.793
        self.Mohr_cohesion = 1.3e3
        self.Mohr_friction = 31.1
        self.Janosi_shear = 1.2e-2
        self.elastic_K = 4e8
        self.damping_R = 3e4
        
def deformable_params(terrain_type):
    """
    Initialize SCM parameters based on terrain type.
    Returns initialized SCMParameters object.
    """
    terrain_params = SCMParameters()
    
    if terrain_type == 'snow':
        terrain_params.InitializeParametersAsSoft()
    elif terrain_type == 'mud':
        terrain_params.InitializeParametersAsMid()
    elif terrain_type == 'sand':
        terrain_params.InitializeParametersAsHard()
    else:
        raise ValueError(f"Unknown deformable terrain type: {terrain_type}")
        
    return terrain_params

class AStarPlanner:
    def __init__(self, grid_map, start, goal, resolution=1.0):
        self.grid_map = grid_map
        self.start = tuple(start)
        self.goal = tuple(goal)
        self.resolution = resolution
        self.open_list = []
        self.closed_list = set()
        self.parent = {}
        self.g_costs = {}

    def heuristic(self, node):
        return ((node[0] - self.goal[0])**2 + (node[1] - self.goal[1])**2)**0.5

    def get_neighbors(self, node):
        directions = [(0, 1), (1, 0), (0, -1), (-1, 0),
                      (1, 1), (-1, 1), (-1, -1), (1, -1)]
        neighbors = []
        for dx, dy in directions:
            neighbor = (node[0] + dx, node[1] + dy)
            if (0 <= neighbor[0] < self.grid_map.shape[0] and
                0 <= neighbor[1] < self.grid_map.shape[1] and
                self.grid_map[neighbor] == 0):
                neighbors.append(neighbor)
        return neighbors

    def plan(self):
        start_node = self.start
        goal_node = self.goal
        heapq.heappush(self.open_list, (0, start_node))
        self.g_costs[start_node] = 0

        while self.open_list:
            _, current = heapq.heappop(self.open_list)

            if current in self.closed_list:
                continue

            self.closed_list.add(current)

            if current == goal_node:
                return self.reconstruct_path(current)

            for neighbor in self.get_neighbors(current):
                tentative_g_cost = self.g_costs[current] + self.resolution
                if (neighbor not in self.g_costs or
                        tentative_g_cost < self.g_costs[neighbor]):
                    self.g_costs[neighbor] = tentative_g_cost
                    f_cost = tentative_g_cost + self.heuristic(neighbor)
                    heapq.heappush(self.open_list, (f_cost, neighbor))
                    self.parent[neighbor] = current

        return None

    def reconstruct_path(self, current):
        path = []
        while current in self.parent:
            path.append(current)
            current = self.parent[current]
        path.append(self.start)
        return path[::-1]

# Use Bezier curve to smooth the path
def bernstein_poly(i, n, t):
    return comb(n, i) * (t ** i) * ((1 - t) ** (n - i))

def bezier_curve(points, num_samples=100):
    n = len(points) - 1
    t = np.linspace(0, 1, num_samples)
    curve = np.zeros((num_samples, 2))
    for i in range(n + 1):
        curve += np.outer(bernstein_poly(i, n, t), points[i])
    return curve

def smooth_path_bezier(path, num_samples=100):
    if path is None or len(path) < 3:
        return path
     
    points = np.array(path)
    curve = bezier_curve(points, num_samples)
    return curve
  
def astar_path(obs_path, m_terrain_length, m_terrain_width, start_pos, goal_pos, inflation_radius=5):
    """
    A* path on the obstacle map
    """
    # Load obstacle map
    obs_image = Image.open(obs_path)
    obs_map = np.array(obs_image.convert('L'))
    grid_map = np.where(obs_map == 255, 1, 0)
    structure = np.ones((2 * inflation_radius + 2, 2 * inflation_radius + 2))

    inflated_map = grid_map.copy()
    for _ in range(inflation_radius):
        inflated_map = binary_dilation(grid_map == 1, structure=structure).astype(int)
    
    # Convert Chrono coordinates to grid coordinates
    def chrono_to_grid(pos):
        grid_height, grid_width = inflated_map.shape
        x = int((pos[0] + m_terrain_length) * grid_width / (2 * m_terrain_length))
        y = int((-pos[1] + m_terrain_width) * grid_height / (2 * m_terrain_width))
        return (min(max(y, 0), grid_height-1), min(max(x, 0), grid_width-1))
    
    # Convert start and goal positions
    start_grid = chrono_to_grid((start_pos[0], start_pos[1]))
    goal_grid = chrono_to_grid((goal_pos[0], goal_pos[1]))
    
    # Plan path
    planner = AStarPlanner(inflated_map, start_grid, goal_grid)
    path = planner.plan()
    
    if path is None:
        print("No valid path found!")
        return None
    
    smoothed_path = smooth_path_bezier(path)

    # # Plot figure
    # plt.figure(figsize=(8, 8), dpi=150)
    # # Set white background
    # ax = plt.gca()
    # ax.set_facecolor('white')
    # plt.gcf().set_facecolor('white')

    # # Plot the obstacle map
    # plt.imshow(grid_map, cmap='binary', alpha=1.0, vmin=0, vmax=1, extent=[0, 129, 129, 0])

    # # Plot the path
    # path_y = [p[0] for p in path]
    # path_x = [p[1] for p in path]
    # # plt.plot(path_x, path_y, color='red', linewidth=2, label='Planned Path', zorder=3)
    # plt.plot(smoothed_path[:, 1], smoothed_path[:, 0], '-', label='Smoothed Path', color='blue')
    # # Plot start and goal
    # plt.scatter(start_grid[1], start_grid[0], color='green', s=150, label='Start', zorder=4, edgecolor='black')
    # plt.scatter(goal_grid[1], goal_grid[0], color='red', marker='*', s=250, label='Goal', zorder=4, edgecolor='black')

    # # Set axis limits
    # plt.xlim(0, 129)
    # plt.ylim(129, 0)
    
    # # Axis labels
    # plt.xlabel('X Position (m)', fontsize=12)
    # plt.ylabel('Y Position (m)', fontsize=12)
    
    # # Set aspect ratio
    # plt.axis('equal')
    # plt.tight_layout()
    # plt.show()
    
    return smoothed_path

def astar_replan(obs_path, m_terrain_length, m_terrain_width, current_pos, goal_pos, inflation_radius=5):
    """
    A* replan from current position to goal
    """
    # Load obstacle map
    obs_image = Image.open(obs_path)
    obs_map = np.array(obs_image.convert('L'))
    grid_map = np.where(obs_map == 255, 1, 0)
    structure = np.ones((2 * inflation_radius + 2, 2 * inflation_radius + 2))
    inflated_map = binary_dilation(grid_map == 1, structure=structure).astype(int)
    
    # Convert coordinates
    def chrono_to_grid(pos):
        grid_height, grid_width = inflated_map.shape
        x = int((pos[0] + m_terrain_length) * grid_width / (2 * m_terrain_length))
        y = int((-pos[1] + m_terrain_width) * grid_height / (2 * m_terrain_width))
        return (min(max(y, 0), grid_height-1), min(max(x, 0), grid_width-1))
    
    current_grid = chrono_to_grid((current_pos[0], current_pos[1]))
    goal_grid = chrono_to_grid((goal_pos[0], goal_pos[1]))
    
    # Replan path
    planner = AStarPlanner(inflated_map, current_grid, goal_grid)
    path = planner.plan()

    if path is None:
        print("No valid path found in replanning!")
        return None
    
    if len(path) >= 3:
        path = smooth_path_bezier(path)
        
    bitmap_points = [(point[1], point[0]) for point in path]
    chrono_path = transform_to_chrono(bitmap_points)

    return chrono_path

class PIDController:
    def __init__(self, kp, ki, kd, setpoint=0.0):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.setpoint = setpoint
        self.previous_error = 0.0
        self.integral = 0.0

    def compute(self, current_value, dt):
        error = self.setpoint - current_value
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        output = self.kp * error + self.ki * self.integral + self.kd * derivative
        self.previous_error = error
        return output

def get_cropped_map_torch(m_vehicle_pos, region_size, batch_size):
    pos_bmp = transform_to_bmp_torch(m_vehicle_pos, bmp_dim_x, bmp_dim_y)
    pos_bmp_x = torch.round(torch.clamp(pos_bmp[:, 0], 0, bmp_dim_x - 1)).int()
    pos_bmp_y = torch.round(torch.clamp(pos_bmp[:, 1], 0, bmp_dim_y - 1)).int()

    center_x = bmp_dim_x // 2
    center_y = bmp_dim_y // 2
    shift_x = center_x - pos_bmp_x
    shift_y = center_y - pos_bmp_y

    grid_y = torch.arange(bmp_dim_y, device='cuda').view(1, bmp_dim_y, 1)
    grid_x = torch.arange(bmp_dim_x, device='cuda').view(1, 1, bmp_dim_x)

    shifted_y = (grid_y + shift_y.view(-1, 1, 1)) % bmp_dim_y
    shifted_x = (grid_x + shift_x.view(-1, 1, 1)) % bmp_dim_x

    shifted_y = shifted_y.to(torch.int32)
    shifted_x = shifted_x.to(torch.int)

    shifted_terrain_array_torch = terrain_array_torch[
        torch.arange(m_vehicle_pos.shape[0], dtype=torch.int64).unsqueeze(1).unsqueeze(2), shifted_y, shifted_x
    ]

    center_y, center_x = bmp_dim_y // 2, bmp_dim_x // 2
    half_region_size = region_size // 2
    under_vehicle_start_y = center_y - half_region_size
    under_vehicle_end_y = center_y + half_region_size
    under_vehicle_start_x = center_x - half_region_size
    under_vehicle_end_x = center_x + half_region_size
    under_vehicle = shifted_terrain_array_torch[:,
                                under_vehicle_start_y:under_vehicle_end_y,
                                under_vehicle_start_x:under_vehicle_end_x
                            ]
    
    return under_vehicle

def transform_to_bmp(chrono_positions):
    bmp_dim_y, bmp_dim_x = terrain_array.shape
    
    # Normalization factors
    s_norm_x = bmp_dim_x / (2 * m_terrain_length)
    s_norm_y = bmp_dim_y / (2 * m_terrain_width)
    
    # Transformation matrix
    T = np.array([
        [s_norm_x, 0, 0],
        [0, s_norm_y, 0],
        [0, 0, 1]
    ])

    bmp_positions = []
    for pos in chrono_positions:
        # Adjust PyChrono coordinates
        vehicle_x = pos[0]  # PyChrono X (Forward)
        vehicle_y = -pos[1]  # PyChrono Y (Left) → invert for BMP
        pos_chrono = np.array([vehicle_x + m_terrain_length, vehicle_y + m_terrain_width, 1])

        # Transform to BMP coordinates
        pos_bmp = np.dot(T, pos_chrono)
        bmp_positions.append((pos_bmp[0], pos_bmp[1]))

    return bmp_positions

def transform_to_bmp_torch(chrono_positions, bmp_dim_x, bmp_dim_y):
    # Convert input to PyTorch tensor if not already
    # pdb.set_trace()
    # chrono_positions = torch.tensor(chrono_positions, dtype=torch.float32).to('cuda')
    
    # Get dimensions of terrain array
    # bmp_dim_y, bmp_dim_x = terrain_array.shape
    

    # Normalization factors
    s_norm_x = bmp_dim_x / (2 * m_terrain_length)
    s_norm_y = bmp_dim_y / (2 * m_terrain_width)
    
    # Transformation matrix
    T = torch.tensor([
        [s_norm_x, 0, 0],
        [0, s_norm_y, 0],
        [0, 0, 1]
    ], dtype=torch.float32, device='cuda')
    # pdb.set_trace()
    # Adjust PyChrono coordinates and add a homogeneous coordinate for matrix multiplication
    vehicle_x = chrono_positions[:, 0].cuda()  # PyChrono X (Forward)
    vehicle_y = -chrono_positions[:, 1].cuda()  # PyChrono Y (Left) → invert for BMP
    pos_chrono = torch.stack((vehicle_x + m_terrain_length, vehicle_y + m_terrain_width, torch.ones_like(vehicle_x)), dim=1)
    
    # Transform to BMP coordinates using batch matrix multiplication
    pos_bmp = torch.matmul(pos_chrono, T.t())  # Transpose T to align dimensions for multiplication

    # Return the transformed positions
    return pos_bmp[:, :2]  # Only x and y are needed

def transform_to_chrono(bmp_positions):
    bmp_dim_y, bmp_dim_x = terrain_array.shape
        
    # Inverse normalization factors
    s_norm_x = bmp_dim_x / (2 * m_terrain_length)
    s_norm_y = bmp_dim_y / (2 * m_terrain_width)

    # Inverse transformation matrix
    T_inv = np.array([
        [1 / s_norm_x, 0, 0],
        [0, 1 / s_norm_y, 0],
        [0, 0, 1]
    ])

    chrono_positions = []
    for pos in bmp_positions:
        # Transform back to PyChrono coordinates
        pos_bmp = np.array([pos[0], pos[1], 1])
        pos_chrono = np.dot(T_inv, pos_bmp)

        # Adjust to PyChrono coordinate system
        x = pos_chrono[0] - m_terrain_length
        y = -(pos_chrono[1] - m_terrain_width)
        chrono_positions.append((x, y))

    return chrono_positions

def initialize_vw_pos(m_vehicle, start_pos, m_isFlat):
    if m_isFlat:
        start_height = start_pos[2]
    else:
        bmp_dim_y, bmp_dim_x = terrain_array.shape
        pos_bmp = transform_to_bmp([start_pos])[0]
        
        def get_interpolated_height(terrain_array, x_float, y_float, bmp_dim_x, bmp_dim_y):
            """
            Get interpolated height value using bilinear interpolation.
            x_float, y_float are floating-point bitmap coordinates.
            """
            # Get central point indices
            x_center = int(np.floor(x_float))
            y_center = int(np.floor(y_float))
            
            # Calculate safe indices for all 8 points
            x_left = max(0, x_center - 1)
            x_right = min(bmp_dim_x - 1, x_center + 1)
            y_up = max(0, y_center - 1)
            y_down = min(bmp_dim_y - 1, y_center + 1)
            
            # Get all 8 surrounding points and the center
            v1 = terrain_array[y_up, x_left]      # Top-left
            v2 = terrain_array[y_up, x_center]    # Top-center
            v3 = terrain_array[y_up, x_right]     # Top-right
            v4 = terrain_array[y_center, x_left]  # Middle-left
            c  = terrain_array[y_center, x_center]# Center
            v5 = terrain_array[y_center, x_right] # Middle-right
            v6 = terrain_array[y_down, x_left]    # Bottom-left
            v7 = terrain_array[y_down, x_center]  # Bottom-center
            v8 = terrain_array[y_down, x_right]   # Bottom-right
            
            # Calculate relative position within the central cell
            dx = x_float - x_center
            dy = y_float - y_center
            
            # Weight calculation based on distance
            # Points closer to the target position get higher weights
            weights = np.zeros(9)
            
            # Corner points (diagonal neighbors)
            weights[0] = max(0, (1 - abs(dx + 1)) * (1 - abs(dy + 1)))  # v1
            weights[2] = max(0, (1 - abs(dx - 1)) * (1 - abs(dy + 1)))  # v3
            weights[6] = max(0, (1 - abs(dx + 1)) * (1 - abs(dy - 1)))  # v6
            weights[8] = max(0, (1 - abs(dx - 1)) * (1 - abs(dy - 1)))  # v8
            
            # Edge points (direct neighbors)
            weights[1] = max(0, (1 - abs(dx)) * (1 - abs(dy + 1)))      # v2
            weights[3] = max(0, (1 - abs(dx + 1)) * (1 - abs(dy)))      # v4
            weights[4] = max(0, (1 - abs(dx - 1)) * (1 - abs(dy)))      # v5
            weights[7] = max(0, (1 - abs(dx)) * (1 - abs(dy - 1)))      # v7
            
            # Center point
            weights[4] = max(0, (1 - abs(dx)) * (1 - abs(dy)))          # c
            
            # Normalize weights to sum to 1
            weights = weights / np.sum(weights)
            
            # Calculate interpolated value using all points
            value = (weights[0] * v1 + weights[1] * v2 + weights[2] * v3 +
                    weights[3] * v4 + weights[4] * c  + weights[5] * v5 +
                    weights[6] * v6 + weights[7] * v7 + weights[8] * v8)
            
            return value / 255.0
        
        pos_bmp_x = np.clip(pos_bmp[0], 0, bmp_dim_x - 1)
        pos_bmp_y = np.clip(pos_bmp[1], 0, bmp_dim_y - 1)
        # Check if pos_bmp_x and pos_bmp_y are within bounds
        assert 0 <= pos_bmp_x < bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
        assert 0 <= pos_bmp_y < bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"
        
        height_ratio = get_interpolated_height(terrain_array, pos_bmp_x, pos_bmp_y, bmp_dim_x, bmp_dim_y)
        start_height = m_min_terrain_height + height_ratio * (m_max_terrain_height - m_min_terrain_height)
        
    start_pos = (start_pos[0], start_pos[1], start_height + start_pos[2])
    dx = goal_pos[0] - start_pos[0]
    dy = goal_pos[1] - start_pos[1]
    start_yaw = np.arctan2(dy, dx)
    m_initLoc = chrono.ChVectorD(*start_pos)
    m_initRot = chrono.Q_from_AngZ(start_yaw)
    m_vehicle.SetInitPosition(chrono.ChCoordsysD(m_initLoc, m_initRot))
    return m_initLoc, m_initRot, start_yaw

def set_goal(m_system, goal_pos, m_isFlat):
    if m_isFlat:
        goal_height = goal_pos[2]
    else:
        bmp_dim_y, bmp_dim_x = terrain_array.shape  # height (rows), width (columns)
        pos_bmp = transform_to_bmp([goal_pos])[0]
        pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, bmp_dim_x - 1)))
        pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, bmp_dim_y - 1)))
        # Check if pos_bmp_x and pos_bmp_y are within bounds
        assert 0 <= pos_bmp_x < bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
        assert 0 <= pos_bmp_y < bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"
        
        pixel_value = terrain_array[pos_bmp_y, pos_bmp_x] / 255.0
        goal_height = m_min_terrain_height + pixel_value * (m_max_terrain_height - m_min_terrain_height)
        
    goal_pos = (goal_pos[0], goal_pos[1], goal_height + goal_pos[2])
    m_goal = chrono.ChVectorD(*goal_pos)

    # Create goal sphere with visualization settings
    goal_contact_material = chrono.ChMaterialSurfaceNSC()
    goal_body = chrono.ChBodyEasySphere(0.5, 1000, True, False, goal_contact_material)
    goal_body.SetPos(m_goal)
    goal_body.SetBodyFixed(True)
    
    # Apply red visualization material
    goal_mat = chrono.ChVisualMaterial()
    goal_mat.SetAmbientColor(chrono.ChColor(1, 0, 0)) 
    goal_mat.SetDiffuseColor(chrono.ChColor(1, 0, 0))
    goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)
    
    # Add the goal body to the system
    m_system.Add(goal_body)
    return m_goal

def get_cropped_map(m_vehicle, m_vehicle_pos, region_size, num_front_regions):
    bmp_dim_y, bmp_dim_x = obstacle_array.shape  # height (rows), width (columns)
    pos_bmp = transform_to_bmp([m_vehicle_pos])[0]
    pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, bmp_dim_x - 1)))
    pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, bmp_dim_y - 1)))
    # Check if pos_bmp_x and pos_bmp_y are within bounds
    assert 0 <= pos_bmp_x < bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
    assert 0 <= pos_bmp_y < bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"

    center_x = bmp_dim_x // 2
    center_y = bmp_dim_y // 2
    shift_x = center_x - pos_bmp_x
    shift_y = center_y - pos_bmp_y

    # Shift the map to center the vehicle position
    shifted_map = np.roll(obstacle_array, shift_y, axis=0)  # y shift affects rows (axis 0)
    shifted_map = np.roll(shifted_map, shift_x, axis=1)    # x shift affects columns (axis 1)
    
    # Rotate the map based on vehicle heading
    vehicle_heading_global = -m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z
    angle = np.degrees(vehicle_heading_global)
    rotated_map = np.array((F.rotate(torch.tensor(shifted_map).unsqueeze(0), angle)).squeeze().cpu(), dtype=np.uint8)

    # Extract the part under the vehicle
    # Vehicle's x-forward direction becomes negative Y in BMP space
    center_y, center_x = rotated_map.shape[0] // 2, rotated_map.shape[1] // 2
    under_vehicle_start_y = center_y - region_size // 2
    under_vehicle_end_y = center_y + region_size // 2
    under_vehicle_start_x = center_x - region_size // 2
    under_vehicle_end_x = center_x + region_size // 2
    
    # Handle boundary conditions for under_vehicle
    under_vehicle_start_x = max(0, under_vehicle_start_x)
    under_vehicle_end_x = min(rotated_map.shape[1], under_vehicle_end_x)
    under_vehicle_start_y = max(0, under_vehicle_start_y)
    under_vehicle_end_y = min(rotated_map.shape[0], under_vehicle_end_y)
    under_vehicle = rotated_map[
        under_vehicle_start_y:under_vehicle_end_y,
        under_vehicle_start_x:under_vehicle_end_x
    ]
    
    # Extract the part in front of the vehicle
    front_regions = []
    offset = num_front_regions // 2
    for i in range(-offset, offset+1):
        front_start_y = under_vehicle_start_y - region_size
        front_end_y = under_vehicle_start_y
        front_start_x = center_x - region_size // 2 + i * region_size
        front_end_x = front_start_x + region_size
        
        # Handle boundary conditions for front regions
        front_start_x = max(0, front_start_x)
        front_end_x = min(rotated_map.shape[1], front_end_x)
        front_start_y = max(0, front_start_y)
        front_end_y = min(rotated_map.shape[0], front_end_y)       
        
        front_region = rotated_map[
            front_start_y:front_end_y,
            front_start_x:front_end_x
        ]
        front_regions.append(front_region)
        
    return under_vehicle, front_regions

def get_current_label(m_vehicle, m_vehicle_pos, region_size, terrain_labels):
    bmp_dim_y, bmp_dim_x = terrain_array.shape  # height (rows), width (columns)
    pos_bmp = transform_to_bmp([m_vehicle_pos])[0]
    pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, bmp_dim_x - 1)))
    pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, bmp_dim_y - 1)))
    # Check if pos_bmp_x and pos_bmp_y are within bounds
    assert 0 <= pos_bmp_x < bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
    assert 0 <= pos_bmp_y < bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"

    center_x = bmp_dim_x // 2
    center_y = bmp_dim_y // 2
    shift_x = center_x - pos_bmp_x
    shift_y = center_y - pos_bmp_y

    # Shift the map to center the vehicle position
    shifted_labels = np.roll(terrain_labels, shift_y, axis=0)  # y shift affects rows (axis 0)
    shifted_labels = np.roll(shifted_labels, shift_x, axis=1)  # x shift affects columns (axis 1)
    
    # Rotate the map based on vehicle heading
    vehicle_heading_global = -m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z
    angle = np.degrees(vehicle_heading_global)
    rotated_labels = np.array((F.rotate(torch.tensor(shifted_labels).unsqueeze(0), angle)).squeeze().cpu(), dtype=np.int32)
    
    # Extract the part under the vehicle
    center_y, center_x = rotated_labels.shape[0] // 2, rotated_labels.shape[1] // 2
    start_y = center_y - region_size // 2
    end_y = center_y + region_size // 2
    start_x = center_x - region_size // 2
    end_x = center_x + region_size // 2
    
    # Handle boundary conditions
    start_y = max(0, start_y)
    end_y = min(rotated_labels.shape[0], end_y)
    start_x = max(0, start_x)
    end_x = min(rotated_labels.shape[1], end_x)
    
    cropped_labels = rotated_labels[start_y:end_y, start_x:end_x]
    return cropped_labels

def find_lowest_position(under_vehicle, front_regions, region_size):
    under_mean = np.mean(under_vehicle)
    under_var = np.var(under_vehicle)
    region_stats = [(np.mean(region), np.var(region)) for region in front_regions]

    # Sort regions by similarity to under_vehicle
    # Primary criterion: mean similarity; Secondary criterion: variance similarity
    sorted_regions = sorted(
        enumerate(region_stats), 
        key=lambda stat: (abs(stat[1][0] - under_mean), abs(stat[1][1] - under_var))
    )
    # Select the best region (most similar)
    target_index, (target_mean, target_variance) = sorted_regions[0]
    
    # Calculate the relative x position of the target region
    center_index = len(front_regions) // 2
    relative_x = (target_index - center_index) * region_size + region_size / 2
    relative_y = region_size / 2

    return relative_x, relative_y

def find_regular_shape(patch_size, max_dim):
    if patch_size > max_dim:
        return []
    
    shapes = []
    max_patches = (max_dim - 1) // (patch_size - 1)
    
    for width_patches in range(1, max_patches + 1):
        for height_patches in range(1, max_patches + 1):
            # Convert to actual dimensions
            width = width_patches * patch_size
            height = height_patches * patch_size
            
            # Check if shape fits within maximum dimension
            if width <= max_dim and height <= max_dim:
                shape = (width, height)
                if shape not in shapes:
                    shapes.append(shape)
                    
    # Sort shapes by area in descending order                
    shapes.sort(key=lambda x: x[0] * x[1], reverse=True)
    return shapes

def best_shape_fit(shapes, patch_size, available_patches):
    if not available_patches:
        return None, set()
        
    max_i = max(i for i, _ in available_patches)
    max_j = max(j for _, j in available_patches)
    
    # Try each shape from largest to smallest
    for width, height in shapes:
        patches_width = width // patch_size
        patches_height = height // patch_size
        
        # Skip if shape is too big for available grid
        if patches_width > max_j + 1 or patches_height > max_i + 1:
            continue
        
        # Try each possible top-left starting position
        for i_start in range(max_i - patches_height + 2):
            for j_start in range(max_j - patches_width + 2):
                # Check if all patches in the rectangle are available
                current_patches = {(i_start + i, j_start + j) 
                                  for i in range(patches_height) 
                                  for j in range(patches_width)}
                if current_patches.issubset(available_patches):
                    return (width, height), current_patches
                    
    return None, set()

def terrain_patch_bmp(terrain_array, start_y, end_y, start_x, end_x, idx):
    # Boundary check
    if (start_y < 0 or end_y > terrain_array.shape[0] or
        start_x < 0 or end_x > terrain_array.shape[1]):
        raise ValueError("Indices out of bounds for terrain array")
    
    # Extract the patch
    patch_array = terrain_array[start_y:end_y, start_x:end_x]

    # Normalize and convert to uint8
    if patch_array.dtype != np.uint8:
        patch_array = ((patch_array - patch_array.min()) * (255.0 / (patch_array.max() - patch_array.min()))).astype(np.uint8)
    # Convert to PIL Image
    patch_image = Image.fromarray(patch_array, mode='L')
    
    # Create file path
    patch_file = f"terrain_patch_{idx}.bmp"
    terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "../data/terrain_bitmaps/BenchMaps/sampled_maps/Worlds")
    
    os.makedirs(terrain_dir, exist_ok=True)
    terrain_path = os.path.join(terrain_dir, patch_file)
    
    # Save the image
    try:
        patch_image.save(terrain_path, format="BMP")
        logging.info(f"Saved terrain patch to {terrain_path}")
    except Exception as e:
        logging.error(f"Failed to save terrain patch: {e}")
        raise
    
    return terrain_path

def combine_rigid(m_system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size, m_isFlat):
    rigid_sections = []
    max_dim = terrain_labels.shape[0]
    
    rigid_patches = defaultdict(set)
    for patch_file, i, j, center_pos in terrain_patches:
        label = terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]
        if not texture_options[label]['is_deformable']:
            rigid_patches[label].add((i, j, center_pos))
            
    processed_patches = set()
    shapes = find_regular_shape(patch_size, max_dim)
    
    for label, patches in rigid_patches.items():
        patch_coords = {(i, j) for i, j, _ in patches}
        
        while patch_coords:
            best_shape, selected_patches = best_shape_fit(shapes, patch_size, patch_coords)
            
            if not best_shape or not selected_patches:
                break

            width, height = best_shape
            patches_width = (width - 1) // (patch_size - 1)
            patches_height = (height - 1) // (patch_size - 1)
            
            # Calculate bounds for this section
            min_i = min(i for i, j in selected_patches)
            min_j = min(j for i, j in selected_patches)
            max_i = max(i for i, j in selected_patches)
            max_j = max(j for i, j in selected_patches)
            
            # Find corner positions
            valid_corner_positions = []
            corner_coords = [(min_i, min_j), (min_i, max_j), (max_i, min_j), (max_i, max_j)]
            for patch in patches:
                i, j, pos = patch
                if (i, j) in corner_coords and (i, j) in selected_patches:
                    valid_corner_positions.append(pos)
            
            # Calculate center position
            avg_x = sum(pos[0] for pos in valid_corner_positions) / len(valid_corner_positions)
            avg_y = sum(pos[1] for pos in valid_corner_positions) / len(valid_corner_positions)
            section_pos = chrono.ChVectorD(avg_x, avg_y, 0)
            
            # Check if all patches have the same properties
            if not selected_patches:
                raise ValueError("No patches selected for merging.")
            
            first_patch = next(iter(selected_patches))
            first_properties = property_dict[(first_patch[0], first_patch[1])]
            first_type = first_properties['terrain_type']
            first_texture = first_properties['texture_file']
            for patch in selected_patches:
                properties = property_dict[(patch[0], patch[1])]
                if properties['terrain_type'] != first_type:
                    raise ValueError(f"Terrain type mismatch: expected {first_type}, found {properties['terrain_type']}.")
                if properties['texture_file'] != first_texture:
                    raise ValueError(f"Texture file mismatch: expected {first_texture}, found {properties['texture_file']}.")
            
            # Create terrain section
            rigid_terrain = veh.RigidTerrain(m_system)
            patch_mat = chrono.ChMaterialSurfaceNSC()
            patch_mat.SetFriction(properties['friction'])
            patch_mat.SetRestitution(properties['restitution'])
            
            if m_isFlat:
                patch = rigid_terrain.AddPatch(patch_mat, 
                                             chrono.ChCoordsysD(section_pos, chrono.CSYSNORM.rot),
                                             width, height)
            else:
                start_i = min_i * (patch_size - 1)
                end_i = max_i * (patch_size - 1) + patch_size
                start_j = min_j * (patch_size - 1)
                end_j = max_j * (patch_size - 1) + patch_size
                
                file = terrain_patch_bmp(terrain_array,
                                       start_i, end_i,
                                       start_j, end_j,
                                       len(rigid_sections))
                                       
                patch = rigid_terrain.AddPatch(patch_mat,
                                             chrono.ChCoordsysD(section_pos, chrono.CSYSNORM.rot),
                                             file,
                                             width, height,
                                             m_min_terrain_height,
                                             m_max_terrain_height)
                os.remove(file)
            
            # Set texture
            patch.SetTexture(veh.GetDataFile(properties['texture_file']), patches_width, patches_height)
            rigid_terrain.Initialize()
            rigid_sections.append(rigid_terrain)
            
            # Update processed patches and remaining patches
            processed_patches.update(selected_patches)
            patch_coords -= selected_patches
    
    # Convert any remaining small patches individually
    for patch_file, i, j, center_pos in terrain_patches:
        if (i, j) not in processed_patches and not texture_options[terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]]['is_deformable']:
            properties = property_dict[(i, j)]
            patch_pos = chrono.ChVectorD(*center_pos)
            
            rigid_terrain = veh.RigidTerrain(m_system)
            patch_mat = chrono.ChMaterialSurfaceNSC()
            patch_mat.SetFriction(properties['friction'])
            patch_mat.SetRestitution(properties['restitution'])
            
            if m_isFlat:
                patch = rigid_terrain.AddPatch(patch_mat,
                                             chrono.ChCoordsysD(patch_pos, chrono.CSYSNORM.rot),
                                             patch_size, patch_size)
            else:
                patch = rigid_terrain.AddPatch(patch_mat,
                                             chrono.ChCoordsysD(patch_pos, chrono.CSYSNORM.rot),
                                             patch_file,
                                             patch_size, patch_size,
                                             m_min_terrain_height,
                                             m_max_terrain_height)
                                             
            patch.SetTexture(veh.GetDataFile(properties['texture_file']), patch_size, patch_size)
            rigid_terrain.Initialize()
            rigid_sections.append(rigid_terrain)
    
    return rigid_sections, property_dict, terrain_labels

def combine_deformation(m_system, terrain_patches, property_dict, texture_options, m_isFlat):
    type_to_label = {}
    deform_terrains = []
    
    for label, info in texture_options.items():
        type_to_label[info['terrain_type']] = label
    
    deformable_terrains = set(
        property_dict[(i, j)]['terrain_type']
        for _, i, j, _ in terrain_patches
        if property_dict[(i, j)]['is_deformable']
    )
    terrain_types = sorted(deformable_terrains)
    num_textures = len(terrain_types)
    bmp_width, bmp_height = terrain_array.shape
    
    if num_textures == 1:
        terrain_type = terrain_types[0]
        center_x, center_y = bmp_width // 2, bmp_height // 2
        chrono_center_x, chrono_center_y = transform_to_chrono([(center_x, center_y)])[0]
        section_pos = chrono.ChVectorD(chrono_center_x, chrono_center_y, 0)
            
        # Create terrain section
        deform_terrain = veh.SCMTerrain(m_system)
        
        # Set SCM parameters
        terrain_params = deformable_params(terrain_type)
        terrain_params.SetParameters(deform_terrain)
        
        # Enable bulldozing
        deform_terrain.EnableBulldozing(True)
        deform_terrain.SetBulldozingParameters(
            55,  # angle of friction for erosion
            1,   # displaced vs downward pressed material
            5,   # erosion refinements per timestep
            10   # concentric vertex selections
        )
        
        # Initialize terrain with regular shape dimensions
        deform_terrain.SetPlane(chrono.ChCoordsysD(section_pos, chrono.CSYSNORM.rot))
        deform_terrain.SetMeshWireframe(False)
        if m_isFlat:
            deform_terrain.Initialize(2 * m_terrain_length, 2 * m_terrain_width, terrain_delta)
        else:
            deform_terrain.Initialize(
                terrain_path,
                2 * m_terrain_length,
                2 * m_terrain_width,
                m_min_terrain_height,
                m_max_terrain_height,
                terrain_delta
            )
        
        label = type_to_label[terrain_type]
        texture_file = texture_options[label]['texture_file']
        deform_terrain.SetTexture(veh.GetDataFile(texture_file), bmp_width, bmp_height)
        deform_terrains.append(deform_terrain)
            
    elif num_textures == 2:
        # Two textures: 1/2 for the first, 1/2 for the second
        split_height = bmp_height // 2

        for idx, terrain_type in enumerate(terrain_types):
            if idx == 0: # First texture
                start_y = 0
                end_y = split_height + 1
                section_height = end_y - start_y
                center_x, center_y = bmp_width // 2, start_y + (section_height - 1) // 2 
            else:  # Second texture
                start_y = split_height
                end_y = bmp_height
                section_height = end_y - start_y
                center_x, center_y = bmp_width // 2, start_y + (section_height + 1) // 2
                
            chrono_center_x, chrono_center_y = transform_to_chrono([(center_x, center_y)])[0]
            section_pos = chrono.ChVectorD(chrono_center_x, chrono_center_y, 0)
            
            # Create terrain section
            deform_terrain = veh.SCMTerrain(m_system)
            
            # Set SCM parameters
            terrain_params = deformable_params(terrain_type)
            terrain_params.SetParameters(deform_terrain)
            
            # Enable bulldozing
            deform_terrain.EnableBulldozing(True)
            deform_terrain.SetBulldozingParameters(
                55,  # angle of friction for erosion
                1,   # displaced vs downward pressed material
                5,   # erosion refinements per timestep
                10   # concentric vertex selections
            )
            
            # Initialize terrain with regular shape dimensions
            deform_terrain.SetPlane(chrono.ChCoordsysD(section_pos, chrono.CSYSNORM.rot))
            deform_terrain.SetMeshWireframe(False)
            if m_isFlat:
                deform_terrain.Initialize(
                    2 * m_terrain_length,
                    section_height * (2 * m_terrain_width / bmp_height),
                    terrain_delta
                )
            else:
                file = terrain_patch_bmp(terrain_array, start_y, end_y, 0, bmp_width, idx, )
                deform_terrain.Initialize(
                    file,
                    2 * m_terrain_length,
                    section_height * (2 * m_terrain_width / bmp_height),
                    m_min_terrain_height,
                    m_max_terrain_height,
                    terrain_delta
                )
                os.remove(file)
            
            label = type_to_label[terrain_type]
            texture_file = texture_options[label]['texture_file']
            deform_terrain.SetTexture(veh.GetDataFile(texture_file), bmp_width, bmp_height)
            deform_terrains.append(deform_terrain)
            
    elif num_textures == 3:
        split_1 = bmp_height // 3
        
        for idx, terrain_type in enumerate(terrain_types):
            if idx == 0:  # Top texture
                start_y = 0
                end_y = split_1 + 1
                section_height = end_y - start_y
                center_x, center_y = bmp_width // 2, start_y + (section_height - 1) // 2

            elif idx == 1:  # Middle texture
                start_y = split_1
                end_y = split_1 * 2 + 1
                section_height = end_y - start_y
                center_x, center_y = bmp_width // 2, start_y + (section_height) // 2

            else:  # Bottom texture
                start_y = split_1 * 2
                end_y = bmp_height
                section_height = end_y - start_y
                center_x, center_y = bmp_width // 2, start_y + (section_height + 2) // 2 + 0.5
            
            chrono_center_x, chrono_center_y = transform_to_chrono([(center_x, center_y)])[0]
            section_pos = chrono.ChVectorD(chrono_center_x, chrono_center_y, 0)
            
            # Create terrain section
            deform_terrain = veh.SCMTerrain(m_system)
            
            # Set SCM parameters
            terrain_params = deformable_params(terrain_type)
            terrain_params.SetParameters(deform_terrain)
            
            # Enable bulldozing
            deform_terrain.EnableBulldozing(True)
            deform_terrain.SetBulldozingParameters(
                55,  # angle of friction for erosion
                1,   # displaced vs downward pressed material
                5,   # erosion refinements per timestep
                10   # concentric vertex selections
            )
            
            # Initialize terrain with regular shape dimensions
            deform_terrain.SetPlane(chrono.ChCoordsysD(section_pos, chrono.CSYSNORM.rot))
            deform_terrain.SetMeshWireframe(False)
            
            width = 2 * m_terrain_length
            height = section_height * (2 * m_terrain_width / bmp_height)

            if m_isFlat:
                deform_terrain.Initialize(width, height, terrain_delta)
            else:
                file = terrain_patch_bmp(terrain_array, start_y, end_y, 0, bmp_width, idx)
                deform_terrain.Initialize(
                    file,
                    width,
                    height,
                    m_min_terrain_height,
                    m_max_terrain_height,
                    terrain_delta
                )
                os.remove(file)
                
            label = type_to_label[terrain_type]
            texture_file = texture_options[label]['texture_file']
            deform_terrain.SetTexture(veh.GetDataFile(texture_file), bmp_width, bmp_height)
            deform_terrains.append(deform_terrain)
            
    return deform_terrains

def mixed_terrain(m_system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size, m_isFlat):
    deformable_sections = []
    max_dim = terrain_labels.shape[0]
    
    deformable_patches = defaultdict(set)
    for patch_file, i, j, center_pos in terrain_patches:
        label = terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]
        if texture_options[label]['is_deformable']:
            deformable_patches[label].add((i, j, center_pos))
            
    processed_patches = set()
    shapes = find_regular_shape(patch_size, max_dim)
    
    for label, patches in deformable_patches.items():
        patch_coords = {(i, j) for i, j, _ in patches}
        best_shape, selected_patches = best_shape_fit(shapes, patch_size, patch_coords)
        
        if not best_shape or not selected_patches:
            continue

        width, height = best_shape
        patches_width = (width - 1) // (patch_size - 1)
        patches_height = (height - 1) // (patch_size - 1)
        
        # Create deformable terrain for this shape
        deform_terrain = veh.SCMTerrain(m_system)
        terrain_type = texture_options[label]['terrain_type']
        terrain_params = deformable_params(terrain_type)
        terrain_params.SetParameters(deform_terrain)
        
        # Enable bulldozing
        deform_terrain.EnableBulldozing(True)
        deform_terrain.SetBulldozingParameters(
            55,  # angle of friction for erosion
            1,   # displaced vs downward pressed material
            5,   # erosion refinements per timestep
            10   # concentric vertex selections
        )
        
        # Calculate center in BMP coordinates
        min_i = min(i for i, j in selected_patches)
        min_j = min(j for i, j in selected_patches)
        max_i = max(i for i, j in selected_patches)
        max_j = max(j for i, j in selected_patches)
        
        valid_corner_positions = []
        corner_coords = [(min_i, min_j), (min_i, max_j), (max_i, min_j), (max_i, max_j)]
        for patch in patches:
            i, j, pos = patch
            if (i, j) in corner_coords and (i, j) in selected_patches:
                valid_corner_positions.append(pos)
        
        # Calculate average center position
        avg_x = sum(pos[0] for pos in valid_corner_positions) / len(valid_corner_positions)
        avg_y = sum(pos[1] for pos in valid_corner_positions) / len(valid_corner_positions)
        section_pos = chrono.ChVectorD(avg_x, avg_y, 0)
        
        # Initialize terrain section
        deform_terrain.SetPlane(chrono.ChCoordsysD(section_pos, chrono.CSYSNORM.rot))
        deform_terrain.SetMeshWireframe(False)
        
        if m_isFlat:
            deform_terrain.Initialize(width, height, terrain_delta)
        else:
            start_i = min_i * (patch_size - 1)
            end_i = max_i * (patch_size - 1) + patch_size
            start_j = min_j * (patch_size - 1)
            end_j = max_j * (patch_size - 1) + patch_size
            file = terrain_patch_bmp(terrain_array, 
                            start_i, end_i,
                            start_j, end_j,
                            len(deformable_sections))
            deform_terrain.Initialize(
                file,
                width, height,
                m_min_terrain_height,
                m_max_terrain_height,
                terrain_delta
            )
            os.remove(file)
        
        # Set texture
        texture_file = texture_options[label]['texture_file']
        deform_terrain.SetTexture(veh.GetDataFile(texture_file), patches_width, patches_height)
        deformable_sections.append(deform_terrain)
        processed_patches.update(selected_patches)
            
    # Convert remaining deformable patches to first rigid texture
    first_rigid_label = min(label for label, info in texture_options.items() if not info['is_deformable'])
    first_rigid_info = next(info for info in textures if info['terrain_type'] == texture_options[first_rigid_label]['terrain_type'])
    
    updated_property_dict = property_dict.copy()
    for patch_file, i, j, center_pos in terrain_patches:
        if (i, j) not in processed_patches and texture_options[terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]]['is_deformable']:
            updated_property_dict[(i, j)] = {
                'is_deformable': False,
                'terrain_type': first_rigid_info['terrain_type'],
                'texture_file': texture_options[first_rigid_label]['texture_file'],
                'friction': first_rigid_info['friction'],
                'restitution': first_rigid_info.get('restitution', 0.01)
            }
            terrain_labels[i * (patch_size - 1):(i + 1) * (patch_size - 1), 
                         j * (patch_size - 1):(j + 1) * (patch_size - 1)] = first_rigid_label

    return deformable_sections, updated_property_dict, terrain_labels

def load_texture_config():
    property_dict = {}
    terrain_patches = []
    
    labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps/configTest/Final", f"labels{world_id}_*.npy")
    matched_labels = glob.glob(labels_path)
    labels_path = matched_labels[0]
    terrain_labels = np.load(labels_path)
    
    texture_options = {}
    terrain_type_to_label = {
        'clay': 0, 'concrete': 1, 'dirt': 2, 'grass': 3, 
        'gravel': 4, 'rock': 5, 'wood': 6,
        'mud': 7, 'sand': 8, 'snow': 9
    }
    
    # Process each texture configuration
    for texture_info in textures:
        i, j = texture_info['index']
        terrain_type = texture_info['terrain_type']
        label = terrain_type_to_label[terrain_type]
        
        center_pos = (
            texture_info['center_position']['x'],
            texture_info['center_position']['y'],
            texture_info['center_position']['z']
        )
        patch_filename = f"patch_{i}_{j}.bmp"
        terrain_patches.append((patch_filename, i, j, center_pos))
        
        # Update texture options
        texture_options[label] = {
            'texture_file': texture_info['texture_file'],
            'terrain_type': terrain_type,
            'is_deformable': texture_info['is_deformable']
        }
        
        # Update property dictionary
        if texture_info['is_deformable']:
            property_dict[(i, j)] = {
                'is_deformable': True,
                'terrain_type': terrain_type,
                'texture_file': texture_info['texture_file']
            }
        else:
            property_dict[(i, j)] = {
                'is_deformable': False,
                'terrain_type': terrain_type,
                'texture_file': texture_info['texture_file'],
                'friction': texture_info['friction'],
                'restitution': texture_info.get('restitution', 0.01)
            }
        
    return property_dict, terrain_labels, texture_options, terrain_patches    

def add_obstacles(m_system, m_isFlat=False):
    m_assets = SimulationAssets(m_system, m_terrain_length * 1.8, m_terrain_width * 1.8,
                                terrain_path, m_min_terrain_height, m_max_terrain_height, m_isFlat)
    
    # Add rocks
    for rock_info in config['obstacles']['rocks']:
        rock_scale = rock_info['scale']
        rock_pos = chrono.ChVectorD(rock_info['position']['x'],
                                  rock_info['position']['y'],
                                  rock_info['position']['z'])
        
        rock = Asset(visual_shape_path="sensor/offroad/rock.obj",
                    scale=rock_scale,
                    bounding_box=chrono.ChVectorD(4.4, 4.4, 3.8))
        
        asset_body = rock.Copy()
        asset_body.UpdateAssetPosition(rock_pos, chrono.ChQuaternionD(1, 0, 0, 0))
        m_system.GetCollisionSystem().BindItem(asset_body.body)
        m_system.Add(asset_body.body)
    
    # Add trees
    for tree_info in config['obstacles']['trees']:
        tree_pos = chrono.ChVectorD(tree_info['position']['x'],
                                  tree_info['position']['y'],
                                  tree_info['position']['z'])
        
        tree = Asset(visual_shape_path="sensor/offroad/tree.obj",
                    scale=1.0,
                    bounding_box=chrono.ChVectorD(1.0, 1.0, 5.0))
        
        asset_body = tree.Copy()
        asset_body.UpdateAssetPosition(tree_pos, chrono.ChQuaternionD(1, 0, 0, 0))
        m_system.GetCollisionSystem().BindItem(asset_body.body)
        m_system.Add(asset_body.body)
    
    return m_assets

def gen_modified_terrain(terrain_array, config, m_terrain_length, m_terrain_width, world_id):
    """
    Generate modified terrain bitmap with obstacles marked
    """
    bmp_dim_y, bmp_dim_x = terrain_array.shape
    modified_terrain = terrain_array.copy()
    
    # Consider obstacles in the elevation map
    for obstacle in config['obstacles']['rocks'] + config['obstacles']['trees']:
        # Get obstacle position and dimensions
        pos = obstacle['position']
        obstacle_pos = chrono.ChVectorD(pos['x'], pos['y'], pos['z'])
        
        # Transform obstacle position to bitmap coordinates
        obstacle_bmp = transform_to_bmp([(obstacle_pos.x, obstacle_pos.y, obstacle_pos.z)])[0]
        obs_x = int(np.round(np.clip(obstacle_bmp[0], 0, bmp_dim_x - 1)))
        obs_y = int(np.round(np.clip(obstacle_bmp[1], 0, bmp_dim_y - 1)))
        
        if 'scale' in obstacle:  # Rocks
            # Rock bounding box is 4.4 x 4.4 x 3.8
            box_width = 4.4 * obstacle['scale']
            box_length = 4.4 * obstacle['scale']
        else:  # Trees
            # Tree bounding box is 1.0 x 1.0 x 5.0
            box_width = 1.0
            box_length = 1.0
            
        # Create a mask for the obstacle
        width_pixels = int(box_width * bmp_dim_x / (2 * m_terrain_length))
        length_pixels = int(box_length * bmp_dim_x / (2 * m_terrain_width))
        
        # Calculate bounds for the obstacle footprint
        x_min = max(0, obs_x - width_pixels // 2)
        x_max = min(bmp_dim_x, obs_x + width_pixels // 2 + 1)
        y_min = max(0, obs_y - length_pixels // 2)
        y_max = min(bmp_dim_y, obs_y + length_pixels // 2 + 1)
        
        modified_terrain[y_min:y_max, x_min:x_max] = 255
    
    # Save modified terrain
    modified_terrain_image = Image.fromarray(modified_terrain.astype(np.uint8), mode='L')
    modified_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                            "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps/configTest/Final", f"modified{world_id}_{difficulty}.bmp")
    modified_terrain_image.save(modified_path)
    
    return modified_path

def run_simulation(render=False, use_gui=False, m_isFlat = False, is_rigid=False, 
                   is_deformable=False, obstacles_flag=False, inflation_radius=5.0):
    if not (is_rigid or is_deformable):
        raise ValueError("At least one terrain type must be enabled")
    
    # System and Terrain Setup
    m_system = chrono.ChSystemNSC()
    m_system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
    m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    
    # Visualization frequencies
    m_vis_freq = 100.0  # Hz
    m_vis_dur = 1.0 / m_vis_freq
    last_vis_time = 0.0
    
    # Vehicle setup
    m_vehicle = veh.HMMWV_Reduced(m_system)
    m_vehicle.SetContactMethod(chrono.ChContactMethod_NSC)
    m_vehicle.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)
    m_vehicle.SetChassisFixed(False)
    m_vehicle.SetEngineType(veh.EngineModelType_SIMPLE_MAP) # This offers higher max torques 
    m_vehicle.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
    m_vehicle.SetDriveType(veh.DrivelineTypeWV_AWD)
    m_vehicle.SetTireType(veh.TireModelType_RIGID)
    m_vehicle.SetTireStepSize(m_step_size)
    m_vehicle.SetInitFwdVel(2.0)
    m_initLoc, m_initRot, m_initYaw = initialize_vw_pos(m_vehicle, start_pos, m_isFlat)
    m_goal = set_goal(m_system, goal_pos, m_isFlat)
    m_vehicle.Initialize()

    m_vehicle.LockAxleDifferential(0, False)    
    m_vehicle.LockAxleDifferential(1, False)
    m_vehicle.LockCentralDifferential(0, False)
    m_vehicle.GetVehicle().EnableRealtime(False)
    
    m_vehicle.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetWheelVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_chassis_body = m_vehicle.GetChassisBody()
    
    # Terrain textures from config
    property_dict, terrain_labels, texture_options, terrain_patches = load_texture_config()
    
    if terrain_type == 'rigid':
        original_labels = terrain_labels.copy()
        rigid_terrains, property_dict, _ = combine_rigid(
            m_system, terrain_patches, terrain_labels.copy(), property_dict,
            texture_options, patch_size, m_isFlat
        )
        terrain_labels = original_labels
        
        if obstacles_flag:
            add_obstacles(m_system, m_isFlat=m_isFlat)
                
    elif terrain_type == 'deformable':
        original_labels = terrain_labels.copy()
        deform_terrains = combine_deformation(m_system, terrain_patches, property_dict, texture_options, m_isFlat)
        terrain_labels = original_labels
        
        if obstacles_flag:
            add_obstacles(m_system, m_isFlat=m_isFlat)
                
    else: 
        original_labels = terrain_labels.copy()
        deform_terrains, property_dict, _ = mixed_terrain(
            m_system, terrain_patches, terrain_labels.copy(), property_dict,
            texture_options, patch_size, m_isFlat
        )
        rigid_terrains, property_dict, _ = combine_rigid(
            m_system, terrain_patches, original_labels, property_dict,
            texture_options, patch_size, m_isFlat
        )
        terrain_labels = original_labels
        
        if obstacles_flag:
            add_obstacles(m_system, m_isFlat=m_isFlat)

    if is_deformable:
        for deform_terrain in deform_terrains:
            deform_terrain.AddMovingPatch(m_chassis_body, chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(5, 3, 3))
            deform_terrain.SetPlotType(veh.SCMTerrain.PLOT_NONE, 0, 1)
    
    # Visualization
    if render:
        vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
        vis.SetWindowTitle('vws in the wild')
        vis.SetWindowSize(3840, 2160)
        trackPoint = chrono.ChVectorD(-3, 0.0, 1.0)
        vis.SetChaseCamera(trackPoint, 3, 1)
        vis.Initialize()
        vis.AddLightDirectional()
        vis.AddSkyBox()
        vis.AttachVehicle(m_vehicle.GetVehicle())
    
    # Set the driver
    if use_gui:
        # GUI-based interactive driver system
        m_driver = veh.ChInteractiveDriverIRR(vis)
        m_driver.SetSteeringDelta(0.1) # Control sensitivity
        m_driver.SetThrottleDelta(0.02)
        m_driver.SetBrakingDelta(0.06)
        m_driver.Initialize()
    else:
        # Automatic driver
        m_driver = veh.ChDriver(m_vehicle.GetVehicle())
        
    m_driver_inputs = m_driver.GetInputs()
    # Set PID controller for speed
    m_speedController = veh.ChSpeedController()
    m_speedController.Reset(m_vehicle.GetRefFrame())
    m_speedController.SetGains(1.0, 0.0, 0.0)
    # Initialize the custom PID controller for steering
    m_steeringController = PIDController(kp=1.0, ki=0.0, kd=0.0)
    
    local_goal_idx = 0
    chrono_path = initial_chrono_path

    def find_local_goal(vehicle_pos, chrono_path, local_goal_idx, look_ahead_distance):
        if local_goal_idx >= len(chrono_path):
            return len(chrono_path) - 1, chrono_path[-1]
        
        for idx in range(local_goal_idx, len(chrono_path)):
            path_point = chrono_path[idx]
            distance = ((vehicle_pos[0] - path_point[0])**2 + (vehicle_pos[1] - path_point[1])**2)**0.5
            
            if distance >= look_ahead_distance:
                return idx, path_point
                
        # If no point is far enough, return the last point
        return len(chrono_path) - 1, chrono_path[-1]
    
    # Continuous speed
    # speed = 4.0 if not use_gui else 0.0 
    start_time = m_system.GetChTime()
    
    # Lists to store metrics
    roll_angles = []
    pitch_angles = []
    throttle_data = []
    steering_data = []
    vehicle_states = []  # store vehicle states (x, y, z, roll, pitch, yaw)
    
    # Check if the vehicle is stuck
    last_position = None
    stuck_counter = 0
    STUCK_DISTANCE = 0.01
    STUCK_TIME = 10.0

    # Create a folder for the current world
    worldEle_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"./res/MPPI6/world{world_id}/elevation")
    os.makedirs(worldEle_folder, exist_ok=True)
    worldSem_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), f"./res/MPPI6/world{world_id}/semantics")
    os.makedirs(worldSem_folder, exist_ok=True) 

    m_mppi_freq = 20.0  # Hz
    m_mppi_dur = 1.0 / m_mppi_freq
    last_mppi_time = 0.0

    last_replan_time = 0.0
    replan_interval = 0.1

    while True:
        if render and not vis.Run():
            break

        new_time = m_system.GetChTime()
        time = m_system.GetChTime()
        # Draw at low frequency
        if render and (last_vis_time==0 or (time - last_vis_time) > m_vis_dur):
            vis.BeginScene()
            vis.Render()
            vis.EndScene()
            last_vis_time = time
            
        m_vehicle_pos = m_vehicle.GetVehicle().GetPos() #Global coordinate
        m_vector_to_goal = m_goal - m_vehicle_pos 
            
        if use_gui:
            m_driver_inputs = m_driver.GetInputs()
        else:        
            euler_angles = m_vehicle.GetVehicle().GetRot().Q_to_Euler123() #Global coordinate
            roll = euler_angles.x
            pitch = euler_angles.y
            vehicle_heading = euler_angles.z
            roll_angles.append(np.degrees(abs(roll)))
            pitch_angles.append(np.degrees(abs(pitch)))
            
            region_size = 8
            under_vehicle, front_regions = get_cropped_map(m_vehicle, (m_vehicle_pos.x, m_vehicle_pos.y, m_vehicle_pos.z), 
                                                           region_size, 5)
            current_label = get_current_label(m_vehicle, (m_vehicle_pos.x, m_vehicle_pos.y, m_vehicle_pos.z), 
                                          8, terrain_labels)
            # print(f"Current label:\n {current_label}")
            
            # Save under_vehicle as BMP
            under_vehicle_image = Image.fromarray(under_vehicle.astype(np.uint8), mode='L')
            under_vehicle_path = os.path.join(worldEle_folder, f"cropped_{time:.2f}.bmp")
            under_vehicle_image.save(under_vehicle_path)
            
            # Save current_label as .npy
            current_label_path = os.path.join(worldSem_folder, f"label_{time:.2f}.npy")
            np.save(current_label_path, current_label)
            
            # Record 6DOF vehicle state (x, y, z, roll, pitch, yaw)
            vehicle_states.append({
                'time': time,
                'x': m_vehicle_pos.x,
                'y': m_vehicle_pos.y,
                'z': m_vehicle_pos.z,
                'roll': np.degrees(roll),
                'pitch': np.degrees(pitch),
                'yaw': np.degrees(vehicle_heading)
            })

            if time - last_replan_time >= replan_interval:
                print("Replanning path...")
                new_path = astar_replan(
                    obs_path,
                    m_terrain_length, 
                    m_terrain_width,
                    (m_vehicle_pos.x, m_vehicle_pos.y),
                    (goal_pos[0], goal_pos[1]),
                    inflation_radius
                )
                
                if new_path is not None:
                    chrono_path = new_path
                    local_goal_idx = 0
                    print("Path replanned successfully")
                else:
                    print("Failed to replan path!")
                
                last_replan_time = time
            
            if last_mppi_time == 0 or (time - last_mppi_time) >= m_mppi_dur:
                
                local_goal_idx, local_goal = find_local_goal(
                    (m_vehicle_pos.x, m_vehicle_pos.y), 
                    chrono_path, 
                    local_goal_idx, 
                    look_ahead_distance
                )
                
                mppi.goal_cb(local_goal)
                mppi.odom_cb(m_vehicle, m_system)
                mppi.gridMap_callback(m_vehicle, m_vehicle_pos)
                
                mppi.mppi_cb()

                # print("Pose Dot", mppi.pose_dot)
                # print("Last Pose: ", mppi.last_pose)
                # print("Current Pose: ", mppi.curr_pose)

                # Get new control values
                speed, steer = mppi.send_controls()
                steer = - steer
                speed = (speed) * 10
                # print(f"Speed: {speed}, Steer: {steer}")
                # Update the last MPPI computation time
                last_mppi_time = time

            m_driver_inputs.m_steering = np.clip(steer, m_driver_inputs.m_steering - 0.05, 
                                                 m_driver_inputs.m_steering + 0.05)

            # Desired throttle/braking value
            out_throttle = m_speedController.Advance(m_vehicle.GetRefFrame(), float(speed), time, m_step_size)
            out_throttle = np.clip(out_throttle, -1, 1)
            if out_throttle > 0:
                m_driver_inputs.m_braking = 0
                m_driver_inputs.m_throttle = out_throttle
            else:
                m_driver_inputs.m_braking = 0
                m_driver_inputs.m_throttle = 0
        

        throttle_data.append(m_driver_inputs.m_throttle)
        steering_data.append(m_driver_inputs.m_steering)
        current_position = (m_vehicle_pos.x, m_vehicle_pos.y, m_vehicle_pos.z)
        # print(f"Current position: {m_vehicle_pos.z}")
        
        if last_position:
            position_change = np.sqrt(
                (current_position[0] - last_position[0])**2 +
                (current_position[1] - last_position[1])**2 +
                (current_position[2] - last_position[2])**2
            )
            
            if position_change < STUCK_DISTANCE:
                stuck_counter += m_step_size
            else:
                stuck_counter = 0
                
            if stuck_counter >= STUCK_TIME:
                print('--------------------------------------------------------------')
                print('Vehicle stuck!')
                print(f'Stuck time: {stuck_counter:.2f} seconds')
                print(f'Position change: {position_change:.3f} m')
                print(f'Initial position: {m_initLoc}')
                print(f'Current position: {m_vehicle_pos}')
                print(f'Goal position: {m_goal}')
                print(f'Distance to goal: {m_vector_to_goal.Length():.2f} m')
                print('--------------------------------------------------------------')
                
                if render:
                    vis.Quit()
                    
                return time - start_time, False, roll_angles, pitch_angles, throttle_data, steering_data, vehicle_states
        
        last_position = current_position
        
        if m_vector_to_goal.Length() < 8:
            print('--------------------------------------------------------------')
            print('Goal Reached')
            print(f'Initial position: {m_initLoc}')
            print(f'Goal position: {m_goal}')
            print('--------------------------------------------------------------')
            if render:
                vis.Quit()
                
            return time - start_time, True, roll_angles, pitch_angles, throttle_data, steering_data, vehicle_states
        
        if m_system.GetChTime() > m_max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', m_initLoc)
            dist = m_vector_to_goal.Length()
            print('Final position of vw: ', m_chassis_body.GetPos())
            print('Goal position: ', m_goal)
            print('Distance to goal: ', dist)
            print('--------------------------------------------------------------')
            if render:
                vis.Quit()
                
            return time - start_time, False, roll_angles, pitch_angles, throttle_data, steering_data, vehicle_states
        
        if is_rigid:
            # print("Rigid terrain", len(rigid_terrains))
            for rigid_terrain in rigid_terrains:
                rigid_terrain.Synchronize(time)
                m_vehicle.Synchronize(time, m_driver_inputs, rigid_terrain)
                rigid_terrain.Advance(m_step_size)
            
        if is_deformable:
            # print("Deform terrain", len(deform_terrains))
            for deform_terrain in deform_terrains:
                deform_terrain.Synchronize(time)
                m_vehicle.Synchronize(time, m_driver_inputs, deform_terrain)
                deform_terrain.Advance(m_step_size)
        
        m_driver.Advance(m_step_size)
        m_vehicle.Advance(m_step_size)
        
        if render:
            vis.Synchronize(time, m_driver_inputs)
            vis.Advance(m_step_size)
        
        m_system.DoStepDynamics(m_step_size)

    return None, False, 0, 0, [], [], [] # Return None if goal not reached

if __name__ == '__main__':
    # Terrain parameters
    SetChronoDataDirectories()
    
    mppi = mppi_planner(
            T=20, 
            K=800,
            sigma=[0.2, 0.5],
            _lambda=0.1,
        )
    
    # Create csv and json files to store results
    res_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "./res/MPPI6")
    os.makedirs(res_dir, exist_ok=True)
    csv_path = os.path.join(res_dir, 'MPPI6_eval.csv')
    json_path = os.path.join(res_dir, 'MPPI6_eval.json')
    fieldnames = [
            'world_id', 'success_rate', 'mean_traversal_time', 'variance_traversal_time',
            'mean_roll', 'variance_roll', 'mean_pitch', 'variance_pitch',
            'mean_throttle', 'variance_throttle', 'mean_steering', 'variance_steering'
        ]
    
    with open(csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    with open(json_path, 'w') as jsonfile:
        json.dump([], jsonfile)

    for world_id in range(1, 101):
        print(f"Evaluating World {world_id}/{100}")

        # Load configuration
        config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps/configTest/Final",
                                f"config{world_id}_*.yaml")
        matched_file = glob.glob(config_path)
        config_path = matched_file[0]
        print(f"Using config file: {config_path}")
        
        with open(config_path, 'r') as file:
            config = yaml.safe_load(file)
        
        m_terrain_length = config['terrain']['length']
        m_terrain_width = config['terrain']['width']
        m_min_terrain_height = config['terrain']['min_height']
        m_max_terrain_height = config['terrain']['max_height']
        difficulty = config['terrain']['difficulty']
        m_isFlat = config['terrain']['is_flat']
        positions = config['positions']
        terrain_type = config['terrain_type']
        obstacle_flag = config['obstacles_flag']
        obstacle_density = config['obstacle_density']
        textures = config['textures']
        terrain_delta = 0.1 # mesh resolution for SCM terrain
        
        # Load terrain bitmap
        terrain_file = f"{world_id}.bmp"
        terrain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                    "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps/Worlds", terrain_file)
        
        terrain_image = Image.open(terrain_path)
        terrain_array = np.array(terrain_image)
        bmp_dim_y, bmp_dim_x = terrain_array.shape 
        if (bmp_dim_y, bmp_dim_x) != (129, 129):
            raise ValueError("Check terrain file and dimensions")

        terrain_array_torch = torch.tensor(terrain_array, dtype=torch.float32, device='cuda').repeat(800, 1, 1)

        # Simulation step sizes
        m_max_time = 60
        m_step_size = 5e-3 # simulation update every num seconds
        
        world_results = []

        for pos_id in range(len(positions)):
            # Start and goal positions
            selected_pair = positions[pos_id]
            start_pos = selected_pair['start']
            goal_pos = selected_pair['goal']
            
            # Small patches folder and size
            patches_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                        "../data/terrain_bitmaps/BenchMaps/sampled_maps/patches")
            patch_size = 9
            
            # A* path planning in chrono
            obs_path = gen_modified_terrain(terrain_array, config, m_terrain_length, m_terrain_width, world_id)
            obstacle_array = np.array(Image.open(obs_path))
            inflation_radius = int(3)
            planned_path = astar_path(
                obs_path,
                m_terrain_length,
                m_terrain_width,
                start_pos,
                goal_pos,
                inflation_radius=inflation_radius
            )
            bitmap_points = [(point[1], point[0]) for point in planned_path]
            initial_chrono_path = transform_to_chrono(bitmap_points)
            look_ahead_distance = 10.0
            
            if terrain_type == 'rigid':
                is_rigid = True
                is_deformable = False
            elif terrain_type == 'deformable':
                is_rigid = False
                is_deformable = True
            else:
                is_rigid = True
                is_deformable = True

            time_to_goal, success, roll_angles, pitch_angles, throttle_data, steering_data, vehicle_states = run_simulation(
                                                                                            render=False, use_gui=False, m_isFlat=m_isFlat,
                                                                                            is_rigid=is_rigid, is_deformable=is_deformable, 
                                                                                            obstacles_flag=obstacle_flag, inflation_radius=inflation_radius
                                                                                        )
            
            world_results.append({
                'time_to_goal': time_to_goal if success else None,
                'success': success,
                'roll_data': roll_angles,
                'pitch_data': pitch_angles,
                'throttle_data': throttle_data if throttle_data else None,
                'steering_data': steering_data if steering_data else None,
                'vehicle_states': vehicle_states
            })
            
        # Calculate statistics for this world
        success_count = sum(1 for r in world_results if r['success'])
        if success_count > 0:
            successful_times = [r['time_to_goal'] for r in world_results if r['success']]
            mean_time = np.mean(successful_times)
            var_time = np.var(successful_times)
        else:
            mean_time = 0
            var_time = 0
            
        all_roll_angles = [angle for r in world_results for angle in r['roll_data']]
        all_pitch_angles = [angle for r in world_results for angle in r['pitch_data']]
        all_throttle = [t for r in world_results for t in r['throttle_data']]
        all_steering = [s for r in world_results for s in r['steering_data']]
        
        world_result = {
            'world_id': world_id,
            'success_rate': success_count / 10,
            'mean_traversal_time': mean_time,
            'variance_traversal_time': var_time,
            'mean_roll': np.mean(all_roll_angles) if all_roll_angles else None,
            'variance_roll': np.var(all_roll_angles) if all_roll_angles else None,
            'mean_pitch': np.mean(all_pitch_angles) if all_pitch_angles else None,
            'variance_pitch': np.var(all_pitch_angles) if all_pitch_angles else None,
            'mean_throttle': np.mean(all_throttle) if all_throttle else None,
            'variance_throttle': np.var(all_throttle) if all_throttle else None,
            'mean_steering': np.mean(all_steering) if all_steering else None,
            'variance_steering': np.var(all_steering) if all_steering else None
        }
        
        with open(csv_path, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(world_result)
            
        with open(json_path, 'r') as jsonfile:
            results = json.load(jsonfile)
        
        results.append(world_result)
        with open(json_path, 'w') as jsonfile:
            json.dump(results, jsonfile, indent=4)

        world_results_path = os.path.join(res_dir, f'world{world_id}_data.json')
        with open(world_results_path, 'w') as world_jsonfile:
            json.dump(world_results, world_jsonfile, indent=4)
            
        print(f"Results for World {world_id} saved to CSV and JSON.")

    print("Evaluation complete!")