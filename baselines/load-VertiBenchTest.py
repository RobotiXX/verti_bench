import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os
import random
import math

from verti_bench.envs.utils.utils import SetChronoDataDirectories
import pychrono.sensor as sens
from verti_bench.envs.utils.asset_utils import *

from PIL import Image, ImageDraw
import shutil
import matplotlib.pyplot as plt
import copy
import yaml
import logging
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

import torch.nn.parallel as parallel
import cv2
from collections import defaultdict
from verti_bench.baselines.TNT.models import TAL, ElevMapEncDec, PatchDecoder
from verti_bench.baselines.TNT.traversabilityEstimator import TravEstimator

class TALTNTPredictor:
    def __init__(self, tal_model_path, tnt_model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dt = 0.1
        
        # Load TAL model
        self.elev_map_encoder = ElevMapEncDec().to(self.device)
        self.elev_map_decoder = PatchDecoder(self.elev_map_encoder.map_encode).to(self.device)
        self.tal_model = TAL(self.elev_map_encoder.map_encode, self.elev_map_decoder).to(self.device)
        
        # Load TNT model
        self.tnt_model = TravEstimator(output_dim=(320, 260)).to(self.device)
        
        # Load model weights
        tal_state_dict = torch.load(tal_model_path)
        tnt_state_dict = torch.load(tnt_model_path)
        
        if isinstance(tal_state_dict, torch.nn.Module):
            tal_state_dict = tal_state_dict.state_dict()
        
        self.tal_model.load_state_dict(tal_state_dict)
        self.tnt_model.load_state_dict(tnt_state_dict)
        
        self.tal_model.eval()
        self.tnt_model.eval()
        
        # Load and set up scales
        with open('/home/tong/Documents/verti_bench/baselines/TNT/stats_rock.pickle', 'rb') as f:
            scale = pickle.load(f)
            
        pose_dot_scale = scale['pose_dot']
        self.state_scale = torch.tensor(
            [pose_dot_scale[0]] * 3 + [pose_dot_scale[1]] * 3,
            dtype=torch.float32
        ).to(self.device)
        
        self.offset_scale = torch.tensor(
            scale['map_offset'],
            dtype=torch.float32
        ).to(self.device)
        
        # TNT specific parameters
        self.m_idx = [1, 3, 5, 7, 8, 9, 10, 11, 12, 13]
        self.weights = torch.tensor(
            [0.15, 0.15, 0.1, 0.1, 0.2, 0.2, 0.15, 0.15, 0.3, 0.3],
            dtype=torch.float32
        ).cuda().unsqueeze(-1).unsqueeze(-1)

    def process_elevation_map(self, current_state, elevation_map_obj):
        """Process elevation map for both TAL and TNT"""
        # Convert current_state tensor to numpy for height value
        current_height = current_state[2].cpu().numpy()
        
        # Get elevation map data directly
        elevation_data = elevation_map_obj.elevation_map
        
        # Center map on vehicle height
        map_centered = elevation_data - current_height
        map_normalized = (np.clip(map_centered, -0.5, 0.5) + 0.5)
        
        # Convert to tensor and add dimensions
        map_tensor = torch.tensor(map_normalized, dtype=torch.float32).to(self.device)
        map_tensor = map_tensor.unsqueeze(0).unsqueeze(0)
        
        # Get original dimensions
        orig_h, orig_w = map_tensor.shape[2:]
        
        # Resize map for TNT input
        map_t = F.interpolate(map_tensor, size=(320, 260), mode='bilinear', align_corners=False)
        
        # Apply Gaussian blur
        kernel_size = 3
        sigma = 0.2
        padding = kernel_size // 2
        
        # Create Gaussian kernel
        kernel_x = torch.arange(kernel_size, device=self.device) - (kernel_size - 1) / 2
        kernel_x = torch.exp(-kernel_x.pow(2) / (2 * sigma ** 2))
        kernel_x = kernel_x / kernel_x.sum()
        kernel_x = kernel_x.view(1, 1, kernel_size, 1)
        kernel_y = kernel_x.transpose(2, 3)
        
        # Apply separable Gaussian blur
        map_t = F.pad(map_t, (padding, padding, padding, padding), mode='reflect')
        map_t = F.conv2d(map_t, kernel_x, padding=(0, padding), groups=1)
        map_t = F.conv2d(map_t, kernel_y, padding=(padding, 0), groups=1)
        
        # Resize for TNT input (160, 130)
        map_t = F.interpolate(map_t, size=(160, 130), mode='bilinear', align_corners=False)
        
        # Get traversability prediction
        with torch.no_grad():
            trav_maps = self.tnt_model(map_t).squeeze()
            trav_maps = trav_maps[self.m_idx]
            trav_maps = trav_maps * self.weights
            combined_trav = torch.sum(trav_maps, dim=0)
            
            # Process elevation for penalty
            elev_norm = F.interpolate(map_tensor, size=combined_trav.shape, mode='bilinear', align_corners=False)
            elev_norm = elev_norm.squeeze() - torch.tensor(current_height).to(self.device)
            
            # Apply penalty for elevation values greater than 4
            combined_trav[elev_norm > 3] = combined_trav.max() * 2.0  # Adjust factor as needed

        return map_tensor, combined_trav

    def predict_state(self, current_state, control, elevation_map_obj):
        """Predict next state using TAL model"""
        current_state_tensor = torch.tensor(current_state, dtype=torch.float32).to(self.device)
        
        # Process maps
        map_tensor, traversability = self.process_elevation_map(current_state_tensor, elevation_map_obj)
        
        # Prepare map for TAL embedding
        map_for_tal = F.interpolate(map_tensor, size=(40, 100), mode='bilinear', align_corners=False)
        
        # Get map embedding for TAL
        with torch.no_grad():
            map_embedding = self.elev_map_encoder(map_for_tal)
            
            # The embedding is [40, 96], we need to reshape it to [1, 160, 6, 4]
            # First reshape to have the correct channel dimension
            map_embedding = map_embedding.transpose(0, 1)  # Now [96, 40]
            map_embedding = map_embedding.reshape(1, 160, -1)  # Now [1, 160, 24]
            
            # Then interpolate to get the desired spatial dimensions
            map_embedding = map_embedding.view(1, 160, 6, 4)
            # Interpolate to get final 6x6 spatial dimensions
            map_embedding = F.interpolate(map_embedding, size=(6, 6), mode='bilinear', align_corners=False)
        
        # Prepare state input
        state = torch.zeros(1, 6, dtype=torch.float32).to(self.device)
        state[0, 3] = current_state_tensor[3]  # roll
        state[0, 4] = current_state_tensor[4]  # pitch
        
        # Convert control to tensor
        cmd_vel = torch.tensor(control, dtype=torch.float32).to(self.device).unsqueeze(0)
        
        # Prepare map offset
        map_offset = torch.zeros(1, 4, dtype=torch.float32).to(self.device)
        yaw = current_state_tensor[5]
        map_offset[0, 0] = 0.1 / self.offset_scale[0]
        map_offset[0, 1] = 0.1 / self.offset_scale[1]
        map_offset[0, 2] = torch.sin(yaw)
        map_offset[0, 3] = torch.cos(yaw)
        
        # Scale state
        state = state / self.state_scale.unsqueeze(0)
        
        # Get prediction
        with torch.no_grad():
            delta_state = self.tal_model.predict(state, cmd_vel, map_offset, map_embedding)
        
        # Scale output back
        delta_state = delta_state * self.state_scale
        delta_state = delta_state.cpu().numpy()[0]
        
        # Integrate to get next state
        next_state = np.copy(current_state)
        next_state[0] += delta_state[0] * self.dt
        next_state[1] += delta_state[1] * self.dt
        next_state[2] += delta_state[2] * self.dt
        next_state[3] = current_state[3] + delta_state[3]
        next_state[4] = current_state[4] + delta_state[4]
        next_state[5] = current_state[5] + delta_state[5] * self.dt
        
        return next_state, traversability

    def scale_input(self, x, scale):
        """Scale input values with tensor operations"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32).to(self.device)
        return x / scale
        
    def scale_output(self, x, scale):
        """Scale output values with tensor operations"""
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32).to(self.device)
        if not isinstance(scale, torch.Tensor):
            scale = torch.tensor(scale, dtype=torch.float32).to(self.device)
        return x * scale

class TraversabilityAwareMPPI:
    def __init__(self, num_samples, horizon, dt, wheelbase, predictor):
        self.num_samples = num_samples
        self.horizon = horizon
        self.dt = dt
        self.robot_wheelbase = wheelbase
        self.state_predictor = predictor
        self.goal_tolerance = 2.0
        
        # Control limits
        self.steering_limits = [-np.pi/4, np.pi/4]
        self.velocity_limits = [0, 5.0]
        
        # MPPI parameters - adjusted for better stability
        self.lambda_ = 1.0  # Increased lambda for more conservative behavior
        self.control_std = np.array([0.2, 0.4])  # Reduced noise for more stable behavior
        
        # Initialize nominal control sequence
        self.nominal_sequence = np.zeros((self.horizon, 2))
        
        # Adjusted cost weights for better balance
        self.cost_weights = {
            'distance': 50.0,        # Increased weight for goal-reaching
            'progress': 30.0,        # Increased weight for forward progress
            'traversability': 40.0,  # Increased penalty for difficult terrain
            'roll': 30.0,           # Increased stability weights
            'pitch': 30.0,
            'control': 0.5          # Increased penalty for aggressive controls
        }

    def get_desired_controls(self, current_state, goal_pos):
        """Calculate desired controls based on goal position"""
        dx = goal_pos[0] - current_state[0]
        dy = goal_pos[1] - current_state[1]
        dz = goal_pos[2] - current_state[2]
        
        # Calculate desired heading
        desired_heading = np.arctan2(dy, dx)
        current_heading = current_state[5]
        
        # Calculate heading error
        heading_error = np.arctan2(np.sin(desired_heading - current_heading), 
                                 np.cos(desired_heading - current_heading))
        
        # Calculate steering based on heading error
        desired_steering = np.clip(0.5 * heading_error, *self.steering_limits)
        
        # Calculate velocity based on distance and heading alignment
        dist_to_goal = np.sqrt(dx**2 + dy**2 + dz**2)
        heading_factor = np.cos(heading_error) ** 2  # 1 when aligned, 0 when perpendicular
        desired_velocity = self.velocity_limits[1] * min(1.0, dist_to_goal/5.0) * heading_factor
        desired_velocity = max(0.5, desired_velocity)
        
        return np.array([desired_steering, desired_velocity])

    def sample_noise(self):
        """Generate control noise with goal-oriented bias"""
        noise = np.random.normal(0, 1.0, (self.num_samples, self.horizon, 2))
        scaled_noise = noise * self.control_std[None, None, :]
        return scaled_noise

    def compute_trajectory(self, start_state, controls):
        """Forward simulate trajectory using TAL predictions"""
        states = [start_state]
        current_state = np.copy(start_state)
        
        for t in range(len(controls)):
            # Get control with limits
            control = np.clip(controls[t], 
                            [self.velocity_limits[0], self.steering_limits[0]],
                            [self.velocity_limits[1], self.steering_limits[1]])
            
            # Get next state prediction from TAL
            next_state = self.state_predictor.predict_state(
                current_state,
                control,
                self.elevation_map.elevation_map
            )
            
            states.append(next_state)
            current_state = next_state
            
        return np.array(states)

    def compute_cost(self, trajectory, controls, traversability, goal_pos):
        """Compute cost with robust error handling"""
        try:
            if trajectory is None or len(trajectory) == 0:
                return float('inf')

            final_state = trajectory[-1]
            initial_state = trajectory[0]

            # Distance cost
            dist_to_goal = np.linalg.norm(final_state[:2] - goal_pos[:2])
            if dist_to_goal < self.goal_tolerance:
                return -1.0

            # Progress cost - with safety check
            initial_dist = np.linalg.norm(initial_state[:2] - goal_pos[:2])
            progress = initial_dist - dist_to_goal
            progress_cost = max(-100, min(100, -progress))  # Limit progress cost

            # Traversability cost - with better handling
            trav_cost = 0.0
            if isinstance(traversability, torch.Tensor):
                # Get valid values only
                valid_trav = traversability[~torch.isnan(traversability)]
                if len(valid_trav) > 0:
                    trav_cost = float(valid_trav.mean().cpu().item())
                    trav_cost = max(0, min(100, trav_cost))  # Limit traversability cost

            # Angle costs - with better handling
            roll_cost = 0.0
            pitch_cost = 0.0
            
            valid_roll = trajectory[:, 3][~np.isnan(trajectory[:, 3])]
            valid_pitch = trajectory[:, 4][~np.isnan(trajectory[:, 4])]
            
            if len(valid_roll) > 0:
                roll_threshold = np.radians(10)
                roll_violations = np.abs(valid_roll[np.abs(valid_roll) > roll_threshold])
                if len(roll_violations) > 0:
                    roll_cost = float(roll_violations.mean())
                    roll_cost = max(0, min(100, roll_cost))

            if len(valid_pitch) > 0:
                pitch_threshold = np.radians(10)
                pitch_violations = np.abs(valid_pitch[np.abs(valid_pitch) > pitch_threshold])
                if len(pitch_violations) > 0:
                    pitch_cost = float(pitch_violations.mean())
                    pitch_cost = max(0, min(100, pitch_cost))

            # Control cost - with safety check
            valid_controls = controls[~np.isnan(controls)]
            control_cost = 0.0
            if len(valid_controls) > 0:
                control_cost = float(np.sum(np.square(valid_controls)))
                control_cost = max(0, min(100, control_cost))

            # Combine costs
            total_cost = (
                self.cost_weights['distance'] * dist_to_goal ** 2 +
                self.cost_weights['progress'] * progress_cost +
                self.cost_weights['traversability'] * trav_cost +
                self.cost_weights['roll'] * roll_cost +
                self.cost_weights['pitch'] * pitch_cost +
                self.cost_weights['control'] * control_cost
            )

            return float(total_cost) if not np.isnan(total_cost) else float('inf')
            
        except Exception as e:
            print(f"Error in cost computation: {e}")
            return float('inf')

    def update_nominal_sequence(self, current_state, goal_pos, weighted_controls):
        """Update nominal sequence with goal-oriented behavior"""
        # Get desired controls based on goal
        desired_control = self.get_desired_controls(current_state, goal_pos)
        
        # Shift the sequence forward
        self.nominal_sequence[:-1] = self.nominal_sequence[1:]
        
        # Blend between weighted controls and desired controls
        alpha = 0.7  # Blend factor
        blended_control = alpha * weighted_controls + (1 - alpha) * desired_control
        self.nominal_sequence[-1] = blended_control

    def plan(self, current_state, goal_pos, elevation_map):
        """Execute MPPI planning with robust error handling and fallback controls"""
        try:
            # Get desired control as fallback
            desired_control = self.get_desired_controls(current_state, goal_pos)
            
            # Initialize nominal sequence if needed
            if np.all(self.nominal_sequence == 0):
                self.nominal_sequence = np.tile(desired_control, (self.horizon, 1))

            # Generate noise samples
            noise = self.sample_noise()
            
            # Create perturbed control sequences
            perturbed_controls = self.nominal_sequence[None, :, :] + noise
            
            # Evaluate trajectories
            costs = np.full(self.num_samples, float('inf'))
            all_trajectories = []
            valid_trajectories = []
            
            for i in range(self.num_samples):
                try:
                    # Initialize trajectory
                    trajectory = []
                    current = np.copy(current_state)
                    controls_i = []
                    
                    # Forward simulate
                    for t in range(self.horizon):
                        # Clip controls to limits
                        control = np.clip(perturbed_controls[i, t], 
                                        [self.velocity_limits[0], self.steering_limits[0]],
                                        [self.velocity_limits[1], self.steering_limits[1]])
                        
                        # Predict next state
                        try:
                            next_state, traversability = self.state_predictor.predict_state(
                                current, control, elevation_map
                            )
                            
                            if not np.any(np.isnan(next_state)):
                                trajectory.append(next_state)
                                controls_i.append(control)
                                current = next_state
                            else:
                                break
                                
                        except Exception as e:
                            print(f"State prediction error: {e}")
                            break
                    
                    # Only consider trajectories with enough valid states
                    if len(trajectory) >= self.horizon // 2:
                        trajectory = np.array(trajectory)
                        controls_i = np.array(controls_i)
                        cost = self.compute_cost(trajectory, controls_i, traversability, goal_pos)
                        
                        if not np.isnan(cost) and not np.isinf(cost):
                            costs[i] = cost
                            all_trajectories.append(trajectory)
                            valid_trajectories.append(i)
                            
                except Exception as e:
                    print(f"Trajectory evaluation error: {e}")
                    continue

            # If no valid trajectories, return fallback control
            if len(valid_trajectories) == 0:
                print("Using fallback control strategy")
                return self.get_desired_controls(current_state, goal_pos), None

            # Compute weights for valid trajectories
            valid_costs = costs[valid_trajectories]
            min_cost = np.min(valid_costs)
            exp_costs = np.exp(-(valid_costs - min_cost) / self.lambda_)
            weights = exp_costs / np.sum(exp_costs)
            
            # Compute weighted controls
            weighted_controls = np.zeros_like(self.nominal_sequence)
            for i, w in zip(valid_trajectories, weights):
                weighted_controls += w * perturbed_controls[i]

            # Update nominal sequence
            self.update_nominal_sequence(current_state, goal_pos, weighted_controls[-1])

            # Get controls
            steering = np.clip(weighted_controls[0, 0] / self.steering_limits[1], -1, 1)
            throttle = np.clip(weighted_controls[0, 1] / self.velocity_limits[1], 0, 1)

            # Get best trajectory
            best_idx = valid_trajectories[np.argmin(valid_costs)]
            best_trajectory = all_trajectories[np.argmin(valid_costs)]

            return [float(steering), float(throttle)], best_trajectory

        except Exception as e:
            print(f"Planning error: {e}")
            return self.get_fallback_control(current_state, goal_pos), None

class GroundTruthElevationMap:
    def __init__(self, bmp_path, scale_factor=0.1, height_offset=None):
        self.elevation_map = self.load_elevation_from_bmp(bmp_path, scale_factor, height_offset)
        self.scale_factor = scale_factor
        self.height_offset = height_offset

    @staticmethod
    def load_elevation_from_bmp(bmp_path, scale_factor=0.1, height_offset=None):
        bmp = cv2.imread(bmp_path, cv2.IMREAD_GRAYSCALE)
        elevation_array = bmp.astype(np.float32) * scale_factor + height_offset
        return elevation_array

    def get_elevation(self, x, y):
        h, w = self.elevation_map.shape
        # Convert position to pixel coordinates
        px = int((x + 40) * w / 80)  # Assuming 80m is map width, centered at 0
        py = int((y + 40) * h / 80)  # Assuming 80m is map height, centered at 0
        
        # Ensure coordinates are within bounds
        px = np.clip(px, 0, w - 1)
        py = np.clip(py, 0, h - 1)
        
        return self.elevation_map[py, px]

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
    shifted_map = np.roll(terrain_array, shift_y, axis=0)  # y shift affects rows (axis 0)
    shifted_map = np.roll(shifted_map, shift_x, axis=1)    # x shift affects columns (axis 1)
    
    # Rotate the map based on vehicle heading
    vehicle_heading_global = m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z
    angle = np.degrees(vehicle_heading_global)
    rotated_map = np.array((F.rotate(torch.tensor(shifted_map).unsqueeze(0), angle)).squeeze().cpu(), dtype=np.uint8)

    # Extract the part under the vehicle
    # Vehicle's x-forward direction becomes negative Y in BMP space
    center_y, center_x = rotated_map.shape[0] // 2, rotated_map.shape[1] // 2
    under_vehicle_start_y = center_y - region_size
    under_vehicle_end_y = center_y
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
    vehicle_heading_global = m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z
    angle = np.degrees(vehicle_heading_global)
    rotated_labels = np.array((TF.rotate(torch.tensor(shifted_labels).unsqueeze(0), angle)).squeeze().cpu(), dtype=np.int32)
    
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
                              "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps")
    
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
                               "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps", "terrain_labels.npy")
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
                    bounding_box=chrono.ChVectorD(4.2, 4.2, 3.8))
        
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

def run_simulation(render=False, use_gui=False, m_isFlat = False, is_rigid=False, is_deformable=False, obstacles_flag=False):
    if not (is_rigid or is_deformable):
        raise ValueError("At least one terrain type must be enabled")
    
    # System and Terrain Setup
    m_system = chrono.ChSystemNSC()
    m_system.SetNumThreads(16)
    m_system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
    m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    
    # Visualization frequencies
    m_vis_freq = 50.0  # Hz
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
    m_vehicle.SetInitFwdVel(0.0)
    m_initLoc, m_initRot, m_initYaw = initialize_vw_pos(m_vehicle, start_pos, m_isFlat)
    m_goal = set_goal(m_system, goal_pos, m_isFlat)
    m_vehicle.Initialize()

    m_vehicle.LockAxleDifferential(0, True)    
    m_vehicle.LockAxleDifferential(1, True)
    m_vehicle.LockCentralDifferential(0, True)
    m_vehicle.GetVehicle().EnableRealtime(False)
    
    m_vehicle.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetWheelVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.GetSystem().SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
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
            deform_terrain.AddMovingPatch(m_chassis_body, chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(5, 3, 1))
            deform_terrain.SetPlotType(veh.SCMTerrain.PLOT_NONE, 0, 1)
    
    # Visualization
    if render:
        vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
        vis.SetWindowTitle('vws in the wild')
        vis.SetWindowSize(3840, 2160)
        trackPoint = chrono.ChVectorD(-3, 0.0, 2)
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
        
        # Initialize planning components
    predictor = TALTNTPredictor(model_path, 
                                tnt_model_path)
    
    elevation_map = GroundTruthElevationMap(bmp_path, scale_factor=0.1, height_offset=m_min_terrain_height)
    
    planner = TraversabilityAwareMPPI(
        num_samples=30,
        horizon=5,
        dt=0.1,
        wheelbase=2.85,
        predictor=predictor
    )
    
    # Initialize state tracking
    actual_path = []
    roll_history = []
    pitch_history = []
    
    # Main simulation loop
    while vis.Run() if render else True:
        time = m_system.GetChTime()

        # Draw at low frequency
        if last_vis_time==0 or (time - last_vis_time) > m_vis_dur:
            last_vis_time = time
            vis.BeginScene()
            vis.Render()
            vis.EndScene()

        # Get current vehicle state
        current_pos = m_vehicle.GetVehicle().GetPos()
        current_rot = m_vehicle.GetVehicle().GetRot()
        current_vel = m_vehicle.GetVehicle().GetSpeed()
        euler_angles = current_rot.Q_to_Euler123()

        m_vector_to_goal = m_goal - current_pos 

        # Create current state vector for planner
        current_state = np.array([
            float(current_pos.x),
            float(current_pos.y),
            float(current_pos.z),
            float(euler_angles.x),  # roll
            float(euler_angles.y),  # pitch
            float(euler_angles.z)   # yaw
        ])

        # Plan trajectory and get control inputs
        control, trajectory = planner.plan(current_state, goal_pos, elevation_map)
        steering, throttle = control

        # Print control inputs for debugging
        print(f"Control Inputs - Steering: {steering:.2f}, Throttle: {throttle:.2f}")

        # Record state for analysis
        actual_path.append(current_state[:3])
        roll_history.append(float(euler_angles.x))
        pitch_history.append(float(euler_angles.y))

        # Check if goal is reached
        dist_to_goal = np.linalg.norm(current_state[:2] - goal_pos[:2])
        if dist_to_goal < planner.goal_tolerance:
            print("Goal reached!")
            break
    
        # Apply control inputs
        m_driver_inputs = m_driver.GetInputs()
        m_driver_inputs.m_steering = np.clip(steering, -1, 1)
        m_driver_inputs.m_throttle = np.clip(throttle, 0, 1)
        m_driver_inputs.m_braking = 0
            
        if m_vector_to_goal.Length() < 4:
            print('--------------------------------------------------------------')
            print('Goal Reached')
            print(f'Initial position: {m_initLoc}')
            print(f'Goal position: {m_goal}')
            print('--------------------------------------------------------------')
            if render:
                vis.Quit()
                
            avg_roll = np.mean(roll_angles) if roll_angles else 0 
            avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
            return time - start_time, True, avg_roll, avg_pitch
        
        if m_system.GetChTime() > m_max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', m_initLoc)
            dist = m_vector_to_goal.Length()
            print('Final position of art: ', m_chassis_body.GetPos())
            print('Goal position: ', m_goal)
            print('Distance to goal: ', dist)
            print('--------------------------------------------------------------')
            if render:
                vis.Quit()
                
            avg_roll = np.mean(roll_angles) if roll_angles else 0 
            avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
            return time - start_time, False, avg_roll, avg_pitch
        
        current_label = get_current_label(m_vehicle, (current_pos.x, current_pos.y, current_pos.z), 
                                          8, terrain_labels)
        print(f"Current label:\n {current_label}")
        
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
    
    return None, False, 0, 0  # Return None if goal not reached

if __name__ == '__main__':
    # Terrain parameters
    SetChronoDataDirectories()
    
    # Load configuration
    config_id = 0
    config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps",
                               f"terrain_config{config_id}.yaml")
    
    terrain_file_options = ["1.bmp"]
    terrain_file = terrain_file_options[0]
    terrain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps", terrain_file)
    bmp_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps", terrain_file)
    model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "../baselines/TNT/TAL_13_14_14.torch")
    enc_dec_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                              "../baselines/TNT/rock_LN_TH_03-07-20-08.torch")
    tnt_model_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                  "../baselines/TNT/best_traversability_model.pth")
    terrain_image = Image.open(terrain_path)
    terrain_array = np.array(terrain_image)
    bmp_dim_y, bmp_dim_x = terrain_array.shape 
    if (bmp_dim_y, bmp_dim_x) != (129, 129):
        raise ValueError("Check terrain file and dimensions")
    
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
        
    m_terrain_length = config['terrain']['length']
    m_terrain_width = config['terrain']['width']
    m_min_terrain_height = config['terrain']['min_height']
    m_max_terrain_height = config['terrain']['max_height']
    difficulty = config['terrain']['difficulty']
    m_isFlat = config['terrain']['is_flat']
    start_pos = config['positions']['start']
    goal_pos = config['positions']['goal']
    terrain_type = config['terrain_type']
    obstacle_flag = config['obstacles_flag']
    obstacle_density = config['obstacle_density']
    textures = config['textures']
    terrain_delta = 0.1 # mesh resolution for SCM terrain

    # Simulation step sizes
    m_max_time = 40
    m_step_size = 5e-3 # simulation update every num seconds
    
    # Small patches folder and size
    patches_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                  "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps/patches")
    patch_size = 17
    
    if terrain_type == 'rigid':
        is_rigid = True
        is_deformable = False
    elif terrain_type == 'deformable':
        is_rigid = False
        is_deformable = True
    else:
        is_rigid = True
        is_deformable = True
        
    # Run multiple experiments 
    num_experiments = 1
    results = [] 

    for i in range(num_experiments):
        print(f"Running experiment {i + 1}/{num_experiments}")
        time_to_goal, success, avg_roll, avg_pitch = run_simulation(render=True, use_gui=False, m_isFlat=m_isFlat,
                                                                    is_rigid=is_rigid, is_deformable=is_deformable, 
                                                                    obstacles_flag=obstacle_flag)
        results.append({
            'time_to_goal': time_to_goal if success else None,
            'success': success,
            'avg_roll': avg_roll,
            'avg_pitch': avg_pitch
        })

    # Process results 
    success_count = sum(1 for r in results if r['success'])
    successful_times = [r['time_to_goal'] for r in results if r['time_to_goal'] is not None]
    avg_rolls = [r['avg_roll'] for r in results if r['success']]
    avg_pitchs = [r['avg_pitch'] for r in results if r['success']]

    mean_traversal_time = np.mean(successful_times) if successful_times else None
    roll_mean = np.mean(avg_rolls) if avg_rolls else None
    roll_variance = np.var(avg_rolls) if avg_rolls else None
    pitch_mean = np.mean(avg_pitchs) if avg_pitchs else None
    pitch_variance = np.var(avg_pitchs) if avg_pitchs else None

    # Print results for the current terrain label
    print("--------------------------------------------------------------")
    print(f"Success rate: {success_count}/{num_experiments}")
    if success_count > 0:
        print(f"Mean traversal time (successful trials): {mean_traversal_time:.2f} seconds")
        print(f"Average roll angle: {roll_mean:.2f} degrees, Variance: {roll_variance:.2f}")
        print(f"Average pitch angle: {pitch_mean:.2f} degrees, Variance: {pitch_variance:.2f}")
    else:
        print("No successful trials")
    print("--------------------------------------------------------------")
    