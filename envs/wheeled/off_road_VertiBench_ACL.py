import gymnasium as gym
import numpy as np
import os
from verti_bench.envs.utils.terrain_utils import SCMParameters
from verti_bench.envs.utils.asset_utils import *
from verti_bench.envs.utils.utils import SetChronoDataDirectories
from verti_bench.envs.ChronoBase import ChronoBaseEnv

import pychrono.vehicle as veh 
import pychrono as chrono
from typing import Any
import logging
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import yaml
from collections import defaultdict
import uuid
import shutil
import glob
import logging
import heapq
from scipy.ndimage import binary_dilation
from scipy.special import comb
import glob

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms
import torchvision.transforms.functional as F
from collections import defaultdict
from verti_bench.train.custom_networks.swae_model import SWAE, LatentSpaceMapper

try:
    from pychrono import irrlicht as chronoirr # type: ignore
except:
    print('Could not import ChronoIrrlicht')
try:
    import pychrono.sensor as sens # type: ignore
except:
    print('Could not import Chrono Sensor')

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

class off_road_art(ChronoBaseEnv):

    # Supported render modes
    metadata = {'additional_render.modes': ['agent_pov', 'None']}

    def __init__(self, world_id=1, pos_id=0, additional_render_mode='None'):
        try:
            # Check if render mode is suppoerted
            if additional_render_mode not in off_road_art.metadata['additional_render.modes']:
                raise Exception(f'Render mode: {additional_render_mode} not supported')
            ChronoBaseEnv.__init__(self, additional_render_mode)

            # Set the chrono data directories for all the terrain
            SetChronoDataDirectories()

            # Set camera frame size
            self.m_camera_width = 80
            self.m_camera_height = 45

            # -----------------------------------
            # Simulation specific class variables
            # -----------------------------------
            self.m_system = None  # Chrono system
            self.m_vehicle = None  # Vehicle set in reset method
            self.m_vehicle_pos = None  # Vehicle position
            self.m_driver = None  # Driver set in reset method
            self.m_driver_input = None  # Driver input set in reset method
            self.m_chassis_body = None  # Chassis body of the vehicle
            
            # Initial location and rotation of the vehicle
            self.m_initLoc = None
            self.m_initRot = None

            # Simulation step sizes
            self.m_max_time = 60 # seconds
            self.m_step_size = 5e-3 # seconds per step
                
            # Visualize frequency
            self.m_vis_freq = 100.0 # Visualization Hz
            self.m_vis_dur = 1.0 / self.m_vis_freq
            self.last_vis_time = 0.0

            # Steer and speed controller
            self.m_speedController = None
            self.max_speed = 4.0
            
            # Initialize world_id and pos_id
            self.world_id = world_id
            self.pos_id = pos_id
            print(f"World ID: {self.world_id}, Position ID: {self.pos_id}")
            
            # Terrain files
            self.terrain_file = f"{self.world_id}.bmp"
            self.terrain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../data/terrain_bitmaps/BenchMaps/sampled_maps/Worlds", self.terrain_file)
            self.terrain_image = Image.open(self.terrain_path)
            self.terrain_array = np.array(self.terrain_image)
            self.bmp_dim_y, self.bmp_dim_x = self.terrain_array.shape 
            if (self.bmp_dim_y, self.bmp_dim_x) != (129, 129):
                raise ValueError("Check terrain file and dimensions")

            # Terrain configuration
            self.config = None
            tmp_config_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                               "../data/terrain_bitmaps/BenchMaps/sampled_maps/configTest/Final",
                               f"config{self.world_id}_*.yaml")
            matched_file = glob.glob(tmp_config_path)
            self.config_path = matched_file[0]
            
            # Load terrain configuration
            with open(self.config_path, 'r') as file:
                self.config = yaml.safe_load(file)
            print(f"Loaded config: {self.config_path}")
                
            # Terrain parameters
            self.m_terrain_length = self.config['terrain']['length']
            self.m_terrain_width = self.config['terrain']['width']
            self.m_min_terrain_height = self.config['terrain']['min_height']
            self.m_max_terrain_height = self.config['terrain']['max_height']
            self.difficulty = self.config['terrain']['difficulty']
            self.m_isFlat = self.config['terrain']['is_flat']
            self.positions = self.config['positions']
            self.terrain_type = self.config['terrain_type']
            self.obstacle_flag = self.config['obstacles_flag']
            self.obstacle_density = self.config['obstacle_density']
            self.textures = self.config['textures']
            self.terrain_delta = 0.1  # mesh resolution for SCM terrain
            
            # Small patches folder and size
            self.patches_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                  "../data/terrain_bitmaps/BenchMaps/sampled_maps/patches")
            self.patch_size = 9
            
            # Terrain type flags
            if self.terrain_type == 'rigid':
                self.is_rigid = True
                self.is_deformable = False
            elif self.terrain_type == 'deformable':
                self.is_rigid = False
                self.is_deformable = True
            else:
                self.is_rigid = True
                self.is_deformable = True
                
            #Start and goal positions
            selected_pair = self.positions[self.pos_id]
            self.start_pos = selected_pair['start']
            self.goal_pos = selected_pair['goal']
            
            # Terrain variables
            self.m_assets = []
            self.submap_shape_x = 64 
            self.submap_shape_y = 64

            # Network params
            self.features_dim = 16
            self.input_size = self.submap_shape_x * self.submap_shape_y
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load the SWAE model
            self.swae = SWAE(in_channels=1, latent_dim=64)
            self.swae.load_state_dict(torch.load(os.path.dirname(
                os.path.realpath(__file__)) + "/../utils/BenchElev.pth", weights_only=True))
            self.swae.freeze_encoder()
            self.swae.to(self.device)
            self.swae.eval()
            
            # Fully Connected Layer to map 64*1 to 16*1
            self.latent_space_mapper = LatentSpaceMapper(64, self.features_dim).to(self.device)
            
            # Min/Max normalize for latent space
            self.min_vector = torch.tensor(np.load(os.path.dirname(
                os.path.realpath(__file__)) + "/../utils/min_vectorBench.npy")).to(self.device)
            self.max_vector = torch.tensor(np.load(os.path.dirname(
                os.path.realpath(__file__)) + "/../utils/max_vectorBench.npy")).to(self.device)
            
            # ----------------------------------------------------
            # Observation space:
            #   1.Cropped array for elevation map: [-1, 1]
            #   2.Difference of Vehicle heading & Heading to goal: [-pi, pi] -> [-1, 1]
            #   3.Velocity of the vehicle [-max_speed, max_speed] -> [-1, 1]
            # ----------------------------------------------------
            # Observation space with elevation map => normalize
            low_bound = np.concatenate(([-1] * self.features_dim, [-1, -1]))
            high_bound = np.concatenate(([1] * self.features_dim, [1, 1]))
            self.observation_space = gym.spaces.Box(
                low=low_bound,
                high=high_bound,
                shape=(self.features_dim + 2,),
                dtype=np.float32
            )
            
            # ------------------------------------------------
            # Action space:
            # Steering is between -1 and 1
            # Linear velocity is: [-maxSpeed, maxSpeed] => [-1, 1]
            # ------------------------------------------------
            # Continuous steering in action space => normalize
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0]), 
                high=np.array([1.0, 1.0]), 
                shape=(2,), 
                dtype=np.float32
            )
            
            # Add A* path planning parameters
            self.look_ahead_distance = 10.0  # meters
            self.local_goal_idx = 0
            self.global_path = None
            self.local_goal = None
            self.inflation_radius = 5
            self.obs_path = None
            
            # Stuck detection parameters
            self.last_position = None  
            self.stuck_counter = 0     
            self.STUCK_DISTANCE = 0.01 
            self.STUCK_TIME = 10.0      
            
            # Sensor manager
            self.m_sens_manager = None  # Sensor manager for the simulation
            self.m_have_camera = False  # Flag to check if camera is present
            self.m_camera = None  # Camera sensor
            self.m_have_gps = False
            self.m_gps = None  # GPS sensor
            self.m_gps_origin = None  # GPS origin
            self.m_have_imu = False
            self.m_imu = None  # IMU sensor
            self.m_imu_origin = None  # IMU origin
            self.m_camera_frequency = 60
            self.m_gps_frequency = 10
            self.m_imu_frequency = 100
            
            # Gym Env specific parameters
            self.m_reward = 0  # Reward for current step
            self.m_debug_reward = 0  # Reward for the whole episode
            # Position of goal as numpy array
            self.m_goal = None
            # Distance to goal at previos time step
            self.m_vector_to_goal = None
            self.m_old_distance = None
            # Observation of the env
            self.m_observation = None
            # Flag to determine if the env has terminated -> timeOut or reach goal
            self.m_terminated = False
            # Flag to determine if the env has truncated -> crash or fallen off terrain
            self.m_truncated = False
            # Flag to check if the render setup has been done
            self.m_render_setup = False
            # Flag to count success while testing
            self.m_additional_render_mode = additional_render_mode
            self.m_episode_num = 0
            self.m_success_count = 0
            self.m_crash_count = 0
            self.m_fallen_count = 0
            self.m_timeout_count = 0
            self.m_stuck_count = 0
            
            # Add tracking variables
            self.time_to_goal = None
            self.roll_angles = []
            self.pitch_angles = []
            self.throttle_data = []
            self.steering_data = []
            self.vehicle_states = []
        
        except Exception as e:
            print(f"Failed to initialize environment: {e}")
            raise e

    def reset(self, seed=None):
        """
        Reset the environment to its initial state
        """
        try:
            # -------------------------------
            # Reset Chrono system
            # -------------------------------
            self.m_system = chrono.ChSystemNSC()
            self.m_system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
            self.m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)  
            
            # -------------------------------
            # Reset the vehicle
            # -------------------------------
            self.m_vehicle = veh.HMMWV_Reduced(self.m_system)
            self.m_vehicle.SetContactMethod(chrono.ChContactMethod_NSC)
            self.m_vehicle.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)
            self.m_vehicle.SetChassisFixed(False)
            self.m_vehicle.SetEngineType(veh.EngineModelType_SIMPLE_MAP)
            self.m_vehicle.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
            self.m_vehicle.SetDriveType(veh.DrivelineTypeWV_AWD)
            self.m_vehicle.SetTireType(veh.TireModelType_RIGID)
            self.m_vehicle.SetTireStepSize(self.m_step_size)
            self.m_vehicle.SetInitFwdVel(0.0)
            self.m_initLoc, self.m_initRot, m_initYaw = self.initialize_vw_pos(self.m_vehicle, self.start_pos, self.m_isFlat)
            self.m_goal = self.set_goal(self.m_system, self.goal_pos, self.m_isFlat)
            self.m_vehicle.Initialize()

            self.m_vehicle.LockAxleDifferential(0, True)    
            self.m_vehicle.LockAxleDifferential(1, True)
            self.m_vehicle.LockCentralDifferential(0, True)
            self.m_vehicle.LockCentralDifferential(1, True)
            self.m_vehicle.GetVehicle().EnableRealtime(False)

            self.m_vehicle.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
            self.m_vehicle.SetWheelVisualizationType(veh.VisualizationType_PRIMITIVES)
            self.m_vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
            self.m_vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
            self.m_vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
            self.m_chassis_body = self.m_vehicle.GetChassisBody()
            
            # Generate obstacle map and plan initial path
            self.obs_path = self.gen_modified_terrain(self.terrain_array, self.config, 
                                                      self.m_terrain_length, self.m_terrain_width)
            planned_path = self.astar_path(
                self.obs_path,
                self.m_terrain_length,
                self.m_terrain_width,
                self.start_pos,
                self.goal_pos,
                inflation_radius=self.inflation_radius
            )
            
            # Convert bitmap path to Chrono coordinates
            bitmap_points = [(point[1], point[0]) for point in planned_path]
            self.global_path = self.transform_to_chrono(bitmap_points)
            self.local_goal_idx = 0
            _, self.local_goal = self.find_local_goal(
                (self.m_initLoc.x, self.m_initLoc.y),
                self.m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z,
                self.global_path, 
                self.local_goal_idx,
                self.look_ahead_distance
            )
            
            # Terrain textures from config
            property_dict, terrain_labels, texture_options, terrain_patches = self.load_texture_config()
            
            # -------------------------------
            # Reset the terrain
            # ------------------------------- 
            if self.terrain_type == 'rigid':
                original_labels = terrain_labels.copy()
                self.rigid_terrains, property_dict, _ = self.combine_rigid(
                    self.m_system, terrain_patches, terrain_labels.copy(), property_dict,
                    texture_options, self.patch_size, self.m_isFlat
                )
                terrain_labels = original_labels
                
                if self.obstacle_flag:
                    self.add_obstacles(self.m_system, m_isFlat=self.m_isFlat)
                        
            elif self.terrain_type == 'deformable':
                original_labels = terrain_labels.copy()
                self.deform_terrains = self.combine_deformation(self.m_system, terrain_patches, property_dict, texture_options, self.m_isFlat)
                terrain_labels = original_labels
                
                if self.obstacle_flag:
                    self.add_obstacles(self.m_system, m_isFlat=self.m_isFlat)
                        
            else: 
                original_labels = terrain_labels.copy()
                self.deform_terrains, property_dict, _ = self.mixed_terrain(
                    self.m_system, terrain_patches, terrain_labels.copy(), property_dict,
                    texture_options, self.patch_size, self.m_isFlat
                )
                self.rigid_terrains, property_dict, _ = self.combine_rigid(
                    self.m_system, terrain_patches, original_labels, property_dict,
                    texture_options, self.patch_size, self.m_isFlat
                )
                terrain_labels = original_labels
                
                if self.obstacle_flag:
                    self.add_obstacles(self.m_system, m_isFlat=self.m_isFlat)
                    
            # ===============================
            # Add the moving terrain patches
            # ===============================
            if self.is_deformable:
                for deform_terrain in self.deform_terrains:
                    deform_terrain.AddMovingPatch(self.m_chassis_body, chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(5, 3, 1))
                    deform_terrain.SetPlotType(veh.SCMTerrain.PLOT_NONE, 0, 1)
            
            # Set the driver
            self.m_driver = veh.ChDriver(self.m_vehicle.GetVehicle())
            self.m_driver_inputs = self.m_driver.GetInputs()

            # Set PID controller for speed
            self.m_speedController = veh.ChSpeedController()
            self.m_speedController.Reset(self.m_vehicle.GetRefFrame())
            self.m_speedController.SetGains(1.0, 0.0, 0.0)

            # -------------------------------
            # Initialize the sensors
            # -------------------------------
            del self.m_sens_manager
            self.m_sens_manager = sens.ChSensorManager(self.m_system)
            # Set the lighting scene
            self.m_sens_manager.scene.AddPointLight(chrono.ChVectorF(
                100, 100, 100), chrono.ChColor(1, 1, 1), 5000.0)

            # Add all the sensors -> For now orientation is ground truth
            self.add_sensors(camera=False, gps=False, imu=False)

            # -------------------------------
            # Get the initial observation
            # -------------------------------
            self.m_observation = self.get_observation()
            self.m_old_distance = self.m_vector_to_goal.Length()
            self.m_debug_reward = 0
            self.m_reward = 0
            self.m_render_setup = False
            self.last_position = None
            self.stuck_counter = 0
            self.m_terminated = False
            self.m_truncated = False
            
            # Clear tracking variables
            self.time_to_goal = None
            self.roll_angles = []
            self.pitch_angles = []
            self.throttle_data = []
            self.steering_data = []
            self.vehicle_states = []
            
            info = {
                'current_time': 0,
                'time_to_goal': None,
                'success': False,
                'roll_angles': [],
                'pitch_angles': [],
                'throttle_data': [],
                'steering_data': [],
                'vehicle_states': []
            }

            return self.m_observation, info
        
        except Exception as e:
            logging.exception("Exception in reset method")
            print(f"Failed to reset environment: {e}")
            raise e

    def step(self, action):
        """
        One step of simulation. Get the driver input from simulation
            Steering: [-1, 1], -1 is right, 1 is left
            Speed: [-5, 5]
        """
        try:
            # normalize
            steering = float(action[0])
            normalized_speed = float(action[1])
            speed = normalized_speed * self.max_speed
            
            time = self.m_system.GetChTime()
            if not hasattr(self, 'last_replan_time'):
                self.last_replan_time = 0
            
            if time - self.last_replan_time >= 0.1:
                new_path = self.astar_replan(
                    self.obs_path,
                    self.m_terrain_length,
                    self.m_terrain_width,
                    (self.m_vehicle_pos.x, self.m_vehicle_pos.y),
                    (self.goal_pos[0], self.goal_pos[1]),
                    self.inflation_radius
                )
                
                if new_path is not None:
                    self.global_path = new_path
                    self.local_goal_idx = 0
                else:
                    print("Failed to replan path!")
                    
                self.last_replan_time = time
            
            # Find local goal
            self.local_goal_idx, self.local_goal = self.find_local_goal(
                (self.m_vehicle_pos.x, self.m_vehicle_pos.y),
                self.m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z,
                self.global_path,
                self.local_goal_idx,
                self.look_ahead_distance
            )
            
            # Desired throttle/braking value
            out_throttle = self.m_speedController.Advance(
                self.m_vehicle.GetRefFrame(), speed, time, self.m_step_size)
            out_throttle = np.clip(out_throttle, -1.0, 1.0)
            
            if out_throttle >= 0:
                self.m_driver_inputs.m_braking = 0
                self.m_driver_inputs.m_throttle = out_throttle
            else:
                self.m_driver_inputs.m_braking = -out_throttle
                self.m_driver_inputs.m_throttle = 0

            # Apply the steering input with smoothing
            self.m_driver_inputs.m_steering = np.clip(steering, -1.0, 1.0)

            # Store raw data
            euler_angles = self.m_vehicle.GetVehicle().GetRot().Q_to_Euler123()
            roll = euler_angles.x
            pitch = euler_angles.y
            vehicle_heading = euler_angles.z
            self.roll_angles.append(np.degrees(abs(roll)))
            self.pitch_angles.append(np.degrees(abs(pitch)))
            self.throttle_data.append(self.m_driver_inputs.m_throttle)
            self.steering_data.append(self.m_driver_inputs.m_steering)
            # Store vehicle state
            vehicle_state = {
                'time': self.m_system.GetChTime(),
                'x': self.m_vehicle_pos.x,
                'y': self.m_vehicle_pos.y,
                'z': self.m_vehicle_pos.z,
                'roll': np.degrees(roll),
                'pitch': np.degrees(pitch),
                'yaw': np.degrees(vehicle_heading)
                
            }
            self.vehicle_states.append(vehicle_state)
            
            # Synchronize and advance simulation for one step
            if self.is_rigid:
                # print("Rigid terrain", len(self.rigid_terrains))
                for rigid_terrain in self.rigid_terrains:
                    rigid_terrain.Synchronize(time)
                    self.m_vehicle.Synchronize(time, self.m_driver_inputs, rigid_terrain)
                    rigid_terrain.Advance(self.m_step_size)
            
            if self.is_deformable:
                # print("Deform terrain", len(self.deform_terrains))
                for deform_terrain in self.deform_terrains:
                    deform_terrain.Synchronize(time)
                    self.m_vehicle.Synchronize(time, self.m_driver_inputs, deform_terrain)
                    deform_terrain.Advance(self.m_step_size)
            
            # Advance simulation for one timestep for all modules
            self.m_driver.Advance(self.m_step_size)
            self.m_vehicle.Advance(self.m_step_size)
            
            if (self.m_render_setup and self.render_mode == 'follow'):
                self.vis.Synchronize(time, self.m_driver_inputs)
                self.vis.Advance(self.m_step_size)

            self.m_system.DoStepDynamics(self.m_step_size)
            # Sensor update
            self.m_sens_manager.Update()
            
            # Check if vehicle is stuck
            current_position = (self.m_vehicle_pos.x, self.m_vehicle_pos.y, self.m_vehicle_pos.z)
            if self.last_position:
                position_change = np.sqrt(
                    (current_position[0] - self.last_position[0])**2 +
                    (current_position[1] - self.last_position[1])**2 +
                    (current_position[2] - self.last_position[2])**2
                )
                
                if position_change < self.STUCK_DISTANCE:
                    self.stuck_counter += self.m_step_size
                else:
                    self.stuck_counter = 0
            
            self.last_position = current_position

            # Get the observation
            self.m_observation = self.get_observation()
            self.m_reward = self.get_reward()
            self.m_debug_reward += self.m_reward
        
            self._is_terminated()
            self._is_truncated()
            
            info = {
                'current_time': self.m_system.GetChTime(),
                'time_to_goal': self.time_to_goal,
                'success': self.m_success_count > 0,
                'roll_angles': self.roll_angles.copy(),
                'pitch_angles': self.pitch_angles.copy(),
                'throttle_data': self.throttle_data.copy(),
                'steering_data': self.steering_data.copy(),
                'vehicle_states': self.vehicle_states.copy()
            }
            
            self.roll_angles = []
            self.pitch_angles = []
            self.throttle_data = []
            self.steering_data = []
            self.vehicle_states = []
            
            return self.m_observation, self.m_reward, self.m_terminated, self.m_truncated, info

        except Exception as e:
            logging.exception("Exception in step method")
            print(f"Error during step execution: {e}")
            raise e

    def render(self, mode='follow'):
        """
        Render the environment
        """
        # ------------------------------------------------------
        # Add visualization - only if we want to see "human" POV
        # ------------------------------------------------------
        if mode == 'human':
            self.render_mode = 'human'

            if self.m_render_setup == False:
                self.vis = chronoirr.ChVisualSystemIrrlicht()
                self.vis.AttachSystem(self.m_system)
                self.vis.SetCameraVertical(chrono.CameraVerticalDir_Z)
                self.vis.SetWindowSize(3840, 2160)
                self.vis.SetWindowTitle('vws in the wild')
                self.vis.Initialize()
                self.vis.AddSkyBox()
                self.vis.AddCamera(chrono.ChVectorD(0, 0, 80), chrono.ChVectorD(0, 0, 1))
                self.vis.AddTypicalLights()
                self.vis.AddLightWithShadow(chrono.ChVectorD(1.5, -2.5, 5.5), chrono.ChVectorD(0, 0, 0.5), 
                                            3, 4, 10, 40, 512)
                self.m_render_setup = True
        
            time = self.m_system.GetChTime()
            # Draw at low frequency
            if self.last_vis_time==0 or (time - self.last_vis_time) > self.m_vis_dur:
                self.vis.BeginScene()
                self.vis.Render()
                self.vis.EndScene()
                self.last_vis_time = time

        elif mode == 'follow':
            self.render_mode = 'follow'
            if self.m_render_setup == False:
                self.vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
                self.vis.SetWindowTitle('vws in the wild')
                self.vis.SetWindowSize(3840, 2160)
                trackPoint = chrono.ChVectorD(-3, 0.0, 1.0)
                self.vis.SetChaseCamera(trackPoint, 3, 1)
                self.vis.Initialize()
                self.vis.AddLightDirectional()
                self.vis.AddSkyBox()
                self.vis.AttachVehicle(self.m_vehicle.GetVehicle())
                self.m_render_setup = True

            time = self.m_system.GetChTime()
            # Draw at low frequency
            if self.last_vis_time==0 or (time - self.last_vis_time) > self.m_vis_dur:
                self.vis.BeginScene()
                self.vis.Render()
                self.vis.EndScene()
                self.last_vis_time = time

        else:
            raise NotImplementedError

    def get_observation(self):
        """
        Get the observation of the environment
            1. Cropped array for elevation map
            2. Difference of Vehicle heading & Heading to goal
            3. Velocity of the vehicle     
        :return: Observation of the environment
        """
        try:
            self.m_vehicle_pos = self.m_chassis_body.GetPos()
        
            # Get GPS info: not used for now
            cur_gps_data = None
            if self.m_have_gps:
                gps_buffer = self.m_gps.GetMostRecentGPSBuffer()
                if gps_buffer.HasData():
                    cur_gps_data = gps_buffer.GetGPSData()
                    cur_gps_data = chrono.ChVectorD(
                        cur_gps_data[1], cur_gps_data[0], cur_gps_data[2])
                else:
                    cur_gps_data = chrono.ChVectorD(self.m_gps_origin)

                # Convert to cartesian coordinates
                sens.GPS2Cartesian(cur_gps_data, self.m_gps_origin)
            else:  # If there is no GPS use ground truth
                cur_gps_data = self.m_vehicle_pos

            if self.m_have_imu:
                raise NotImplementedError('IMU not implemented yet')
            
            # Get observation
            pos_bmp = self.transform_to_bmp([(self.m_vehicle_pos.x, self.m_vehicle_pos.y, self.m_vehicle_pos.z)])[0]
            pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, self.bmp_dim_x - 1)))
            pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, self.bmp_dim_y - 1)))
            # Check if pos_bmp_x and pos_bmp_y are within bounds
            assert 0 <= pos_bmp_x < self.bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
            assert 0 <= pos_bmp_y < self.bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"
            
            modified_terrain = self.terrain_array.copy()
            center_x = self.bmp_dim_x // 2
            center_y = self.bmp_dim_y // 2
            shift_x = center_x - pos_bmp_x
            shift_y = center_y - pos_bmp_y

            # Shift the map to center the vehicle position
            shifted_map = np.roll(modified_terrain, shift_y, axis=0)  # y shift affects rows (axis 0)
            shifted_map = np.roll(shifted_map, shift_x, axis=1)    # x shift affects columns (axis 1)
            
            # Rotate the map based on vehicle heading
            vehicle_heading_global = -self.m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z
            angle = np.degrees(vehicle_heading_global)
            rotated_map = np.array((F.rotate(torch.tensor(shifted_map).unsqueeze(0), angle)).squeeze().cpu(), dtype=np.uint8)

            # Extract the part under the vehicle
            # Vehicle's x-forward direction becomes negative Y in BMP space
            center_y, center_x = rotated_map.shape[0] // 2, rotated_map.shape[1] // 2
            under_vehicle_start_y = center_y - self.submap_shape_y // 2
            under_vehicle_end_y = center_y + self.submap_shape_y // 2
            under_vehicle_start_x = center_x - self.submap_shape_x // 2
            under_vehicle_end_x = center_x + self.submap_shape_x // 2
            
            # Handle boundary conditions for under_vehicle
            under_vehicle_start_x = max(0, under_vehicle_start_x)
            under_vehicle_end_x = min(rotated_map.shape[1], under_vehicle_end_x)
            under_vehicle_start_y = max(0, under_vehicle_start_y)
            under_vehicle_end_y = min(rotated_map.shape[0], under_vehicle_end_y)
            under_vehicle = rotated_map[
                under_vehicle_start_y:under_vehicle_end_y,
                under_vehicle_start_x:under_vehicle_end_x
            ]

            flattened_map = under_vehicle.flatten()
            flattened_map_normalized = (flattened_map / 255.0) * 2 - 1
            flattened_map_tensor = torch.tensor(flattened_map_normalized, dtype=torch.float32).unsqueeze(0).to(self.device)
            flattened_map_tensor = flattened_map_tensor.view(-1, 1, self.submap_shape_x, self.submap_shape_y)
            # -----------------------------------------
            # Here is the feature extraction for SWAE
            # -----------------------------------------
            _, _, z = self.swae(flattened_map_tensor) #64*64 -> 64*1
            
            # Normalize Observation
            z_normalized = 2 * (z - self.min_vector) / (self.max_vector - self.min_vector) - 1
            mapped_features_tensor = self.latent_space_mapper(z_normalized) #64*1 -> 16*1
            mapped_features_array = mapped_features_tensor.cpu().detach().numpy().flatten()
            
            # Heading difference
            self.local_goal = chrono.ChVectorD(self.local_goal[0], self.local_goal[1], 0)
            self.m_vector_to_goal = self.local_goal - self.m_vehicle_pos 
            goal_heading = np.arctan2(self.m_vector_to_goal.y, self.m_vector_to_goal.x)
            euler_angles = self.m_vehicle.GetVehicle().GetRot().Q_to_Euler123() #Global coordinate
            roll = euler_angles.x
            pitch = euler_angles.y
            vehicle_heading = euler_angles.z
            heading_error = (goal_heading - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
            normalized_heading_diff = heading_error / np.pi
            
            # Get vehicle speed and normalize
            vehicle_speed = self.m_chassis_body.GetPos_dt().Length()
            normalized_speed = vehicle_speed / self.max_speed
            normalized_speed = np.clip(normalized_speed, -1.0, 1.0)
            observation_array = np.array([normalized_heading_diff, normalized_speed])
            final_observation = np.concatenate((mapped_features_array, observation_array)).astype(np.float32)
            return final_observation
        
        except AssertionError as e:
            print(f"Assertion failed in get_observation: {str(e)}")
            raise
    
        except Exception as e:
            print(f"Error in get_observation: {str(e)}")
            raise

    def get_reward(self):
        # Compute the progress made
        progress_scale = 50 # coefficient for scaling progress reward
        
        distance = self.m_vector_to_goal.Length()
        # print(f"Distance: {distance}")
        # The progress made with the last action
        progress = self.m_old_distance - distance
        reward = progress_scale * progress

        # If we have not moved even by 1 cm in 0.1 seconds give a penalty
        if np.abs(progress) < 0.01:
            reward -= 10

        # Roll and pitch angles
        euler_angles = self.m_vehicle.GetVehicle().GetRot().Q_to_Euler123()
        roll = euler_angles.x
        pitch = euler_angles.y

        # Define roll and pitch thresholds
        roll_threshold = np.radians(30)  
        pitch_threshold = np.radians(30)

        # Scale for roll and pitch penalties
        roll_penalty_scale = 20 * np.abs(roll / roll_threshold) if np.abs(roll) > roll_threshold else 0
        pitch_penalty_scale = 20 * np.abs(pitch / pitch_threshold) if np.abs(pitch) > pitch_threshold else 0

        # Add penalties for excessive roll and pitch
        if abs(roll) > roll_threshold:
            reward -= roll_penalty_scale * (abs(roll) - roll_threshold)
        if abs(pitch) > pitch_threshold:
            reward -= pitch_penalty_scale * (abs(pitch) - pitch_threshold)

        self.m_old_distance = distance

        # # Debugging
        # print(f"Distance: {distance}")
        # print(f"Progress: {progress}")
        # print(f"Roll: {roll}, Pitch: {pitch}")
        # print(f"Roll Penalty: {roll_penalty_scale * (abs(roll) - roll_threshold)}")
        # print(f"Pitch Penalty: {pitch_penalty_scale * (abs(pitch) - pitch_threshold)}")
        # print(f"Reward: {reward}")

        return reward

    def _is_terminated(self):
        """
        Check if the environment is terminated
        """
        # If we are within a certain distance of the goal -> Terminate and give big reward
        m_vector_to_goal = self.m_goal - self.m_vehicle_pos
        if m_vector_to_goal.Length() < 10:
            print('--------------------------------------------------------------')
            print('Goal Reached')
            print('Initial position: ', self.m_initLoc)
            print('Goal position: ', self.m_goal)
            print('--------------------------------------------------------------')
            self.m_reward += 3000
            self.m_debug_reward += self.m_reward
            self.m_terminated = True
            self.m_success_count += 1
            self.m_episode_num += 1
            self.time_to_goal = self.m_system.GetChTime()

        # If we have exceeded the max time -> Terminate and give penalty for how far we are from the goal
        if self.m_system.GetChTime() > self.m_max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', self.m_initLoc)
            m_vector_to_goal = self.m_goal - self.m_vehicle_pos
            dist = m_vector_to_goal.Length()
            print('Final position of art: ', self.m_chassis_body.GetPos())
            print('Goal position: ', self.m_goal)
            print('Distance to goal: ', dist)
            # Give it a reward based on how close it reached the goal
            self.m_reward -= 100  # Fixed penalty for timeout
            self.m_reward -= 10 * dist

            self.m_debug_reward += self.m_reward
            print('Reward: ', self.m_reward)
            print('Accumulated Reward: ', self.m_debug_reward)
            print('--------------------------------------------------------------')
            self.m_terminated = True
            self.m_episode_num += 1
            self.m_timeout_count += 1

    def _is_truncated(self):
        """
        Check if we have crashed or fallen off terrain
        We will follow A* local goal path
        """
        # if self.obstacle_flag:
        #     collision = self.m_assets.CheckContact(self.m_chassis_body)
        #     if collision:
        #         self.m_reward -= 50
        #         print('--------------------------------------------------------------')
        #         print(f'Crashed')
        #         print('--------------------------------------------------------------')
        #         self.m_debug_reward += self.m_reward
        #         self.m_truncated = True
        #         self.m_episode_num += 1
        #         self.m_crash_count += 1
        
        if self.stuck_counter >= self.STUCK_TIME:
            m_vector_to_goal = self.m_goal - self.m_vehicle_pos 
            print('--------------------------------------------------------------')
            print('Vehicle stuck!')
            print(f'Stuck time: {self.stuck_counter:.2f} seconds')
            print(f'Initial position: {self.m_initLoc}')
            print(f'Current position: {self.m_vehicle_pos}')
            print(f'Goal position: {self.m_goal}')
            print(f'Distance to goal: {m_vector_to_goal.Length():.2f} m')
            print('--------------------------------------------------------------')

            # Penalize and terminate episode
            self.m_reward -= 300
            self.m_truncated = True
            self.m_episode_num += 1
            self.m_stuck_count += 1
    
        if (self._fallen_off_terrain()):
            self.m_reward -= 600
            print('--------------------------------------------------------------')
            print('Fallen off terrain')
            print('--------------------------------------------------------------')
            self.m_debug_reward += self.m_reward
            self.m_truncated = True
            self.m_episode_num += 1
            self.m_fallen_count += 1

    def _fallen_off_terrain(self):
        """
        Check if we have fallen off the terrain
        """
        terrain_length_tolerance = self.m_terrain_length
        terrain_width_tolerance = self.m_terrain_width

        vehicle_is_outside_terrain = abs(self.m_vehicle_pos.x) > terrain_length_tolerance or abs(
            self.m_vehicle_pos.y) > terrain_width_tolerance
        if (vehicle_is_outside_terrain):
            return True
        else:
            return False

    def deformable_params(self, terrain_type):
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
    
    def transform_to_bmp(self, chrono_positions):
        # Normalization factors
        s_norm_x = self.bmp_dim_x / (2 * self.m_terrain_length)
        s_norm_y = self.bmp_dim_y / (2 * self.m_terrain_width)
        
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
            pos_chrono = np.array([vehicle_x + self.m_terrain_length, vehicle_y + self.m_terrain_width, 1])

            # Transform to BMP coordinates
            pos_bmp = np.dot(T, pos_chrono)
            bmp_positions.append((pos_bmp[0], pos_bmp[1]))

        return bmp_positions

    def transform_to_chrono(self, bmp_positions):
        # Inverse normalization factors
        s_norm_x = self.bmp_dim_x / (2 * self.m_terrain_length)
        s_norm_y = self.bmp_dim_y / (2 * self.m_terrain_width)

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
            x = pos_chrono[0] - self.m_terrain_length
            y = -(pos_chrono[1] - self.m_terrain_width)
            chrono_positions.append((x, y))

        return chrono_positions
    
    def initialize_vw_pos(self, m_vehicle, start_pos, m_isFlat):
        if m_isFlat:
            start_height = start_pos[2]
        else:
            pos_bmp = self.transform_to_bmp([start_pos])[0]
            
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
            
            pos_bmp_x = np.clip(pos_bmp[0], 0, self.bmp_dim_x - 1)
            pos_bmp_y = np.clip(pos_bmp[1], 0, self.bmp_dim_y - 1)
            # Check if pos_bmp_x and pos_bmp_y are within bounds
            assert 0 <= pos_bmp_x < self.bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
            assert 0 <= pos_bmp_y < self.bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"
            
            height_ratio = get_interpolated_height(self.terrain_array, pos_bmp_x, pos_bmp_y, self.bmp_dim_x, self.bmp_dim_y)
            start_height = self.m_min_terrain_height + height_ratio * (self.m_max_terrain_height - self.m_min_terrain_height)
            
        start_pos = (start_pos[0], start_pos[1], start_height + start_pos[2])
        dx = self.goal_pos[0] - start_pos[0]
        dy = self.goal_pos[1] - start_pos[1]
        start_yaw = np.arctan2(dy, dx)
        m_initLoc = chrono.ChVectorD(*start_pos)
        m_initRot = chrono.Q_from_AngZ(start_yaw)
        m_vehicle.SetInitPosition(chrono.ChCoordsysD(m_initLoc, m_initRot))
        return m_initLoc, m_initRot, start_yaw

    def set_goal(self, m_system, goal_pos, m_isFlat):
        if m_isFlat:
            goal_height = goal_pos[2]
        else:
            pos_bmp = self.transform_to_bmp([goal_pos])[0]
            pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, self.bmp_dim_x - 1)))
            pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, self.bmp_dim_y - 1)))
            # Check if pos_bmp_x and pos_bmp_y are within bounds
            assert 0 <= pos_bmp_x < self.bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
            assert 0 <= pos_bmp_y < self.bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"
            
            pixel_value = self.terrain_array[pos_bmp_y, pos_bmp_x] / 255.0
            goal_height = self.m_min_terrain_height + pixel_value * (self.m_max_terrain_height - self.m_min_terrain_height)
            
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
    
    def get_cropped_map(self, m_vehicle, m_vehicle_pos, region_size, num_front_regions):
        pos_bmp = self.transform_to_bmp([m_vehicle_pos])[0]
        pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, self.bmp_dim_x - 1)))
        pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, self.bmp_dim_y - 1)))
        # Check if pos_bmp_x and pos_bmp_y are within bounds
        assert 0 <= pos_bmp_x < self.bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
        assert 0 <= pos_bmp_y < self.bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"

        center_x = self.bmp_dim_x // 2
        center_y = self.bmp_dim_y // 2
        shift_x = center_x - pos_bmp_x
        shift_y = center_y - pos_bmp_y

        # Shift the map to center the vehicle position
        shifted_map = np.roll(self.terrain_array, shift_y, axis=0)  # y shift affects rows (axis 0)
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
    
    def get_current_label(self, m_vehicle, m_vehicle_pos, region_size, terrain_labels):
        pos_bmp = self.transform_to_bmp([m_vehicle_pos])[0]
        pos_bmp_x = int(np.round(np.clip(pos_bmp[0], 0, self.bmp_dim_x - 1)))
        pos_bmp_y = int(np.round(np.clip(pos_bmp[1], 0, self.bmp_dim_y - 1)))
        # Check if pos_bmp_x and pos_bmp_y are within bounds
        assert 0 <= pos_bmp_x < self.bmp_dim_x, f"pos_bmp_x out of bounds: {pos_bmp_x}"
        assert 0 <= pos_bmp_y < self.bmp_dim_y, f"pos_bmp_y out of bounds: {pos_bmp_y}"

        center_x = self.bmp_dim_x // 2
        center_y = self.bmp_dim_y // 2
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

    def find_regular_shape(self, patch_size, max_dim):
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
    
    def best_shape_fit(self, shapes, patch_size, available_patches):
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
    
    def terrain_patch_bmp(self, terrain_array, start_y, end_y, start_x, end_x, idx):
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
        unique_id = uuid.uuid4().hex
        patch_file = f"terrain_patch_{idx}_{unique_id}.bmp"
        terrain_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../data/terrain_bitmaps/BenchMaps/sampled_maps/tmp")
        
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
    
    def combine_rigid(self, m_system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size, m_isFlat):
        rigid_sections = []
        max_dim = terrain_labels.shape[0]
        
        rigid_patches = defaultdict(set)
        for patch_file, i, j, center_pos in terrain_patches:
            label = terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]
            if not texture_options[label]['is_deformable']:
                rigid_patches[label].add((i, j, center_pos))
                
        processed_patches = set()
        shapes = self.find_regular_shape(patch_size, max_dim)
        
        for label, patches in rigid_patches.items():
            patch_coords = {(i, j) for i, j, _ in patches}
            
            while patch_coords:
                best_shape, selected_patches = self.best_shape_fit(shapes, patch_size, patch_coords)
                
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
                    
                    file = self.terrain_patch_bmp(self.terrain_array,
                                        start_i, end_i,
                                        start_j, end_j,
                                        len(rigid_sections))
                                        
                    patch = rigid_terrain.AddPatch(patch_mat,
                                                chrono.ChCoordsysD(section_pos, chrono.CSYSNORM.rot),
                                                file,
                                                width, height,
                                                self.m_min_terrain_height,
                                                self.m_max_terrain_height)
                
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
                                                self.m_min_terrain_height,
                                                self.m_max_terrain_height)
                                                
                patch.SetTexture(veh.GetDataFile(properties['texture_file']), patch_size, patch_size)
                rigid_terrain.Initialize()
                rigid_sections.append(rigid_terrain)
        
        return rigid_sections, property_dict, terrain_labels

    def combine_deformation(self, m_system, terrain_patches, property_dict, texture_options, m_isFlat):
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
        bmp_width, bmp_height = self.terrain_array.shape
        
        if num_textures == 1:
            terrain_type = terrain_types[0]
            center_x, center_y = bmp_width // 2, bmp_height // 2
            chrono_center_x, chrono_center_y = self.transform_to_chrono([(center_x, center_y)])[0]
            section_pos = chrono.ChVectorD(chrono_center_x, chrono_center_y, 0)
                
            # Create terrain section
            deform_terrain = veh.SCMTerrain(m_system)
            
            # Set SCM parameters
            terrain_params = self.deformable_params(terrain_type)
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
                deform_terrain.Initialize(2 * self.m_terrain_length, 2 * self.m_terrain_width, self.terrain_delta)
            else:
                deform_terrain.Initialize(
                    self.terrain_path,
                    2 * self.m_terrain_length,
                    2 * self.m_terrain_width,
                    self.m_min_terrain_height,
                    self.m_max_terrain_height,
                    self.terrain_delta
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
                    
                chrono_center_x, chrono_center_y = self.transform_to_chrono([(center_x, center_y)])[0]
                section_pos = chrono.ChVectorD(chrono_center_x, chrono_center_y, 0)
                
                # Create terrain section
                deform_terrain = veh.SCMTerrain(m_system)
                
                # Set SCM parameters
                terrain_params = self.deformable_params(terrain_type)
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
                        2 * self.m_terrain_length,
                        section_height * (2 * self.m_terrain_width / bmp_height),
                        self.terrain_delta
                    )
                else:
                    file = self.terrain_patch_bmp(self.terrain_array, start_y, end_y, 0, bmp_width, idx)
                    deform_terrain.Initialize(
                        file,
                        2 * self.m_terrain_length,
                        section_height * (2 * self.m_terrain_width / bmp_height),
                        self.m_min_terrain_height,
                        self.m_max_terrain_height,
                        self.terrain_delta
                    )
                
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
                
                chrono_center_x, chrono_center_y = self.transform_to_chrono([(center_x, center_y)])[0]
                section_pos = chrono.ChVectorD(chrono_center_x, chrono_center_y, 0)
                
                # Create terrain section
                deform_terrain = veh.SCMTerrain(m_system)
                
                # Set SCM parameters
                terrain_params = self.deformable_params(terrain_type)
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
                
                width = 2 * self.m_terrain_length
                height = section_height * (2 * self.m_terrain_width / bmp_height)

                if m_isFlat:
                    deform_terrain.Initialize(width, height, self.terrain_delta)
                else:
                    file = self.terrain_patch_bmp(self.terrain_array, start_y, end_y, 0, bmp_width, idx)
                    deform_terrain.Initialize(
                        file,
                        width,
                        height,
                        self.m_min_terrain_height,
                        self.m_max_terrain_height,
                        self.terrain_delta
                    )
                    
                label = type_to_label[terrain_type]
                texture_file = texture_options[label]['texture_file']
                deform_terrain.SetTexture(veh.GetDataFile(texture_file), bmp_width, bmp_height)
                deform_terrains.append(deform_terrain)
                
        return deform_terrains

    def mixed_terrain(self, m_system, terrain_patches, terrain_labels, property_dict, texture_options, patch_size, m_isFlat):
        deformable_sections = []
        max_dim = terrain_labels.shape[0]
        
        deformable_patches = defaultdict(set)
        for patch_file, i, j, center_pos in terrain_patches:
            label = terrain_labels[i * (patch_size - 1), j * (patch_size - 1)]
            if texture_options[label]['is_deformable']:
                deformable_patches[label].add((i, j, center_pos))
                
        processed_patches = set()
        shapes = self.find_regular_shape(patch_size, max_dim)
        
        for label, patches in deformable_patches.items():
            patch_coords = {(i, j) for i, j, _ in patches}
            best_shape, selected_patches = self.best_shape_fit(shapes, patch_size, patch_coords)
            
            if not best_shape or not selected_patches:
                continue

            width, height = best_shape
            patches_width = (width - 1) // (patch_size - 1)
            patches_height = (height - 1) // (patch_size - 1)
            
            # Create deformable terrain for this shape
            deform_terrain = veh.SCMTerrain(m_system)
            terrain_type = texture_options[label]['terrain_type']
            terrain_params = self.deformable_params(terrain_type)
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
                deform_terrain.Initialize(width, height, self.terrain_delta)
            else:
                start_i = min_i * (patch_size - 1)
                end_i = max_i * (patch_size - 1) + patch_size
                start_j = min_j * (patch_size - 1)
                end_j = max_j * (patch_size - 1) + patch_size
                file = self.terrain_patch_bmp(self.terrain_array, 
                                start_i, end_i,
                                start_j, end_j,
                                len(deformable_sections))
                deform_terrain.Initialize(
                    file,
                    width, height,
                    self.m_min_terrain_height,
                    self.m_max_terrain_height,
                    self.terrain_delta
                )
            
            # Set texture
            texture_file = texture_options[label]['texture_file']
            deform_terrain.SetTexture(veh.GetDataFile(texture_file), patches_width, patches_height)
            deformable_sections.append(deform_terrain)
            processed_patches.update(selected_patches)
                
        # Convert remaining deformable patches to first rigid texture
        first_rigid_label = min(label for label, info in texture_options.items() if not info['is_deformable'])
        first_rigid_info = next(info for info in self.textures if info['terrain_type'] == texture_options[first_rigid_label]['terrain_type'])
        
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

    def load_texture_config(self):
        property_dict = {}
        terrain_patches = []
        
        labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../data/terrain_bitmaps/BenchMaps/sampled_maps/configTest/Final", f"labels{self.world_id}_*.npy")
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
        for texture_info in self.textures:
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
    
    def add_obstacles(self, m_system, m_isFlat=False):
        self.m_assets = SimulationAssets(m_system, self.m_terrain_length * 1.8, self.m_terrain_width * 1.8,
                                    self.terrain_path, self.m_min_terrain_height, self.m_max_terrain_height, m_isFlat)
        
        # Add rocks
        for rock_info in self.config['obstacles']['rocks']:
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
        for tree_info in self.config['obstacles']['trees']:
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
        
        return self.m_assets
    
    def gen_modified_terrain(self, terrain_array, config, m_terrain_length, m_terrain_width):
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
            obstacle_bmp = self.transform_to_bmp([(obstacle_pos.x, obstacle_pos.y, obstacle_pos.z)])[0]
            obs_x = int(np.round(np.clip(obstacle_bmp[0], 0, bmp_dim_x - 1)))
            obs_y = int(np.round(np.clip(obstacle_bmp[1], 0, bmp_dim_y - 1)))
            
            if 'scale' in obstacle:  # Rocks
                # Rock bounding box is 4.2 x 4.2 x 3.8
                box_width = 4.2 * obstacle['scale']
                box_length = 4.2 * obstacle['scale']
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
            
            modified_terrain[y_min:y_max, x_min:x_max] = 119
        
        # Save modified terrain
        modified_terrain_image = Image.fromarray(modified_terrain.astype(np.uint8), mode='L')
        modified_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../data/terrain_bitmaps/BenchMaps/sampled_maps/configTest/Final", f"modified{self.world_id}_{self.difficulty}.bmp")
        modified_terrain_image.save(modified_path)
        
        return modified_path
    
    def astar_path(self, obs_path, m_terrain_length, m_terrain_width, start_pos, goal_pos, inflation_radius=5):
        """
        A* path on the obstacle map
        """
        # Load obstacle map
        obs_image = Image.open(obs_path)
        obs_map = np.array(obs_image.convert('L'))
        grid_map = np.where(obs_map == 119, 1, 0)
        structure = np.ones((2 * inflation_radius + 1, 2 * inflation_radius + 1))
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
        path = smooth_path_bezier(path)
        
        if path is None:
            print("No valid path found!")
            return None
        
        # # Plot figure
        # plt.figure(figsize=(8, 8), dpi=150)
        # # Set white background
        # ax = plt.gca()
        # ax.set_facecolor('white')
        # plt.gcf().set_facecolor('white')

        # # Plot the obstacle map
        # plt.imshow(grid_map, cmap='binary', alpha=1.0, vmin=0, vmax=1, extent=[0, 129, 129, 0])
        
        # # Plot the smooth path
        # path_y = [p[0] for p in path]
        # path_x = [p[1] for p in path]
        # plt.plot(path_x, path_y, color='red', linewidth=2, label='Planned Path', zorder=3)

        # # Plot start and goal
        # plt.scatter(start_grid[1], start_grid[0], color='green', s=150, label='Start', zorder=4, edgecolor='black')
        # plt.scatter(goal_grid[1], goal_grid[0], color='red', marker='*', s=250, label='Goal', zorder=4, edgecolor='black')

        # # Set axis limits
        # plt.xlim(0, 129)
        # plt.ylim(129, 0)
        
        # # Axis labels
        # plt.xlabel('X Position (m)', fontsize=12)
        # plt.ylabel('Y Position (m)', fontsize=12)
        # plt.title('Global Path Planning', fontsize=14, pad=15)
        
        # # Customize grid
        # plt.grid(True, linestyle='--', alpha=0.5, color='gray')

        # # # Legend customization
        # # legend = plt.legend(loc='upper right', framealpha=0.9, facecolor='white', fontsize=8, borderpad=1)
        # # for text in legend.get_texts():
        # #     text.set_fontweight('bold')

        # # Set aspect ratio
        # plt.axis('equal')
        # plt.tight_layout()
        # plt.show()
        
        return path

    def astar_replan(self, obs_path, m_terrain_length, m_terrain_width, current_pos, goal_pos, inflation_radius=5):
        """
        A* replan from current position to goal
        """
        # Load obstacle map
        obs_image = Image.open(obs_path)
        obs_map = np.array(obs_image.convert('L'))
        grid_map = np.where(obs_map == 119, 1, 0)
        structure = np.ones((2 * inflation_radius + 1, 2 * inflation_radius + 1))
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
        chrono_path = self.transform_to_chrono(bitmap_points)
        
        return chrono_path
    
    def find_local_goal(self, vehicle_pos, vehicle_heading, chrono_path, local_goal_idx, look_ahead_distance):
        if local_goal_idx >= len(chrono_path):
            return len(chrono_path) - 1, chrono_path[-1]
        
        for idx in range(local_goal_idx, len(chrono_path)):
            path_point = chrono_path[idx]
            distance = ((vehicle_pos[0] - path_point[0])**2 + (vehicle_pos[1] - path_point[1])**2)**0.5
            
            if distance >= look_ahead_distance:
                dx = path_point[0] - vehicle_pos[0]
                dy = path_point[1] - vehicle_pos[1]
                angle_to_point = np.arctan2(dy, dx)
                angle_diff = (angle_to_point - vehicle_heading + np.pi) % (2 * np.pi) - np.pi
                if abs(angle_diff) <= np.pi / 2:
                    return idx, path_point
                
        # If no point is far enough, return the last point
        return len(chrono_path) - 1, chrono_path[-1]
    
    def add_sensors(self, camera=True, gps=True, imu=True):
        """
        Add sensors to the simulation
        :param camera: Flag to add camera sensor
        :param gps: Flag to add gps sensor
        :param imu: Flag to add imu sensor
        """
        # -------------------------------
        # Add camera sensor
        # -------------------------------
        if camera:
            self.m_have_camera = True
            cam_loc = chrono.ChVectorD(0.1, 0, 0.08)
            cam_rot = chrono.Q_from_AngAxis(0, chrono.ChVectorD(0, 1, 0))
            cam_frame = chrono.ChFrameD(cam_loc, cam_rot)

            self.m_camera = sens.ChCameraSensor(
                self.m_chassis_body,  # body camera is attached to
                self.m_camera_frequency,  # update rate in Hz
                cam_frame,  # offset pose
                self.m_camera_width,  # image width
                self.m_camera_height,  # image height
                chrono.CH_C_PI / 3,  # FOV
                # supersampling factor (higher improves quality of the image)
                6
            )
            self.m_camera.SetName("Camera Sensor")
            self.m_camera.PushFilter(sens.ChFilterRGBA8Access())
            if (self.m_additional_render_mode == 'agent_pov'):
                self.m_camera.PushFilter(sens.ChFilterVisualize(
                    self.m_camera_width, self.m_camera_height, "Agent POV"))
            self.m_sens_manager.AddSensor(self.m_camera)
        if gps:
            self.m_have_gps = True
            std = 0.01  # GPS noise standard deviation - Good RTK GPS
            gps_noise = sens.ChNoiseNormal(chrono.ChVectorD(
                0, 0, 0), chrono.ChVectorD(std, std, std))
            gps_loc = chrono.ChVectorD(0, 0, 0)
            gps_rot = chrono.Q_from_AngAxis(0, chrono.ChVectorD(0, 1, 0))
            gps_frame = chrono.ChFrameD(gps_loc, gps_rot)
            self.m_gps_origin = chrono.ChVectorD(43.073268, -89.400636, 260.0)

            self.m_gps = sens.ChGPSSensor(
                self.m_chassis_body,
                self.m_gps_frequency,
                gps_frame,
                self.m_gps_origin,
                gps_noise
            )
            self.m_gps.SetName("GPS Sensor")
            self.m_gps.PushFilter(sens.ChFilterGPSAccess())
            self.m_sens_manager.AddSensor(self.m_gps)
        if imu:
            self.m_have_imu = True
            std = 0.01
            imu_noise = sens.ChNoiseNormal(chrono.ChVectorD(
                0, 0, 0), chrono.ChVectorD(std, std, std))
            imu_loc = chrono.ChVectorD(0, 0, 0)
            imu_rot = chrono.Q_from_AngAxis(0, chrono.ChVectorD(0, 1, 0))
            imu_frame = chrono.ChFrameD(imu_loc, imu_rot)
            self.m_imu_origin = chrono.ChVectorD(43.073268, -89.400636, 260.0)
            self.m_imu = sens.ChIMUSensor(
                self.m_chassis_body,
                self.m_imu_frequency,
                imu_frame,
                imu_noise,
                self.m_imu_origin
            )
            self.m_imu.SetName("IMU Sensor")
            self.m_imu.PushFilter(sens.ChFilterMagnetAccess())
            self.m_sens_manager.AddSensor(self.m_imu)

    def close(self):
        try:
            del self.m_vehicle
            if hasattr(self, 'm_sens_manager'):
                del self.m_sens_manager
            del self.m_system
            del self
        except Exception as e:
            print(f"Failed to close environment: {e}")
            raise e

    def __del__(self):
        try:
            if hasattr(self, 'm_sens_manager'):
                del self.m_sens_manager
            if hasattr(self, 'm_system'):
                del self.m_system
        except Exception as e:
            print(f"Failed to delete environment: {e}")
            raise e
