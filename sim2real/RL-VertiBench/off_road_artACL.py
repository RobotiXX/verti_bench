import gymnasium as gym
import numpy as np
import os
from verti_bench.envs.utils.terrain_utils import SCMParameters
from verti_bench.envs.utils.asset_utils import *
from verti_bench.envs.utils.utils import SetChronoDataDirectories
from verti_bench.envs.ChronoBase import ChronoBaseEnv
import pychrono.vehicle as veh # type: ignore
import pychrono as chrono # type: ignore
from typing import Any 
import logging
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import cv2
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.parallel as parallel
from verti_bench.train.custom_networks.ElevationCNN import ElevationExtractor
from verti_bench.train.custom_networks.vae_model import VAE
from verti_bench.train.custom_networks.swae_model import SWAE, LatentSpaceMapper

try:
    from pychrono import irrlicht as chronoirr # type: ignore
except:
    print('Could not import ChronoIrrlicht')
try:
    import pychrono.sensor as sens # type: ignore
except:
    print('Could not import Chrono Sensor')


class off_road_art(ChronoBaseEnv):

    # Supported render modes
    metadata = {'additional_render.modes': ['agent_pov', 'None']}

    def __init__(self, terrain_stage=0, additional_render_mode='None'):
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
            self.m_assets = None  # List of assets in the simulation
            self.m_system = None  # Chrono system
            self.m_vehicle = None  # Vehicle set in reset method
            self.m_vehicle_pos = None  # Vehicle position
            self.m_driver = None  # Driver set in reset method
            self.m_driver_input = None  # Driver input set in reset method
            self.m_chassis = None  # Chassis body of the vehicle
            self.m_chassis_body = None  # Chassis body of the vehicle
            self.m_chassis_collision_box = None  # Chassis collision box of the vehicle
            self.m_proper_collision = False
            # Initial location and rotation of the vehicle
            self.m_initLoc = None
            self.m_initRot = None
            self.m_contact_force = None  # Contact force on the vehicle

            # Simulation step sizes
            self.m_step_size = 5e-3 # seconds per step
            self.m_control_freq = 10 # control updates per second
            self.m_steps_per_control = round(1 / (self.m_step_size * self.m_control_freq))

            # Steer and speed controller
            self.m_speedController = None
            self.max_speed = 4.0

            # Terrain
            self.m_terrain = None  # Actual deformable terrain
            self.m_min_terrain_height = 0  # min terrain height
            self.m_max_terrain_height = 7  # max terrain height
            self.m_terrain_length = 21  # size in X direction
            self.m_terrain_width = 21  # size in Y direction
            self.m_assets = []
            self.m_positions = []
            self.bmp_dim_x = 0
            self.bmp_dim_y = 0
            self.submap_shape_x = 64 # Cropped image width by decoder size (pixels)
            self.submap_shape_y = 64 # Cropped image length by decoder size (pixels)
            self.map_offset_x  = 0
            self.map_offset_y  = 0
            self.terrain_file = [f"level_{i}.bmp" for i in range(1, 101)]
            self.current_terrain_stage = terrain_stage
            self.bitmap_file = None
            self.terrain_loaded = False  # Flag to check if the terrain loaded
            self.m_step = 0 # Stable step count 
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

            # Network params
            self.features_dim = 16
            self.input_size = self.submap_shape_x * self.submap_shape_y
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Load the SWAE model
            self.swae = SWAE(in_channels=1, latent_dim=64)
            self.swae.load_state_dict(torch.load(os.path.dirname(
                os.path.realpath(__file__)) + "/../utils/SWAE_ACL64.pth"))
            self.swae.freeze_encoder()
            self.swae.to(self.device)
            self.swae.eval()
            
            # Fully Connected Layer to map 64 to features_dim
            self.latent_space_mapper = LatentSpaceMapper(64, self.features_dim).to(self.device)
            
            # Load min/max vectors
            self.min_vectorACL = torch.tensor(np.load(os.path.dirname(
                os.path.realpath(__file__)) + "/../utils/min_vectorACL.npy")).to(self.device)
            self.max_vectorACL = torch.tensor(np.load(os.path.dirname(
                os.path.realpath(__file__)) + "/../utils/max_vectorACL.npy")).to(self.device)
            
            # ----------------------------------------------------
            # Observation space:
            #   1.Cropped array for elevation map: [-1, 1]
            #   2.Difference of Vehicle heading & Heading to goal: [-pi, pi] -> [-1, 1]
            #   3.Velocity of the vehicle [-max_speed, max_speed] -> [-1, 1]/[0, 1]
            # ----------------------------------------------------
            # # Observation space with elevation map => part-normalize
            # low_bound = np.concatenate(([-1] * self.features_dim, [-np.pi, -4.0]))
            # high_bound = np.concatenate(([1] * self.features_dim, [np.pi, 4.0]))
            # self.observation_space = gym.spaces.Box(
            #     low=low_bound, 
            #     high=high_bound, 
            #     shape=(self.features_dim + 2,), dtype=np.float32)
            
            # Observation space with elevation map => normalize
            # velocity is normalized to [-1, 1]
            low_bound = np.concatenate(([-1] * self.features_dim, [-1, -1]))
            high_bound = np.concatenate(([1] * self.features_dim, [1, 1]))
            self.observation_space = gym.spaces.Box(
                low=low_bound, 
                high=high_bound, 
                shape=(self.features_dim + 2,), dtype=np.float32)
            
            # ------------------------------------------------
            # Action space:
            # Steering is between -1 and 1
            # Linear velocity is: [-4, 4] => [-1, 1]/[0, 1]
            # ------------------------------------------------
            # # Continuous steering in action space => part-normalize
            # self.action_space = gym.spaces.Box(
            #     low=np.array([-1.0, -4.0]), high=np.array([1.0, 4.0]), 
            #     shape=(2,), dtype=np.float32)
            
            # Continuous steering in action space => normalize
            # velocity is normalized to [-1, 1]
            self.action_space = gym.spaces.Box(
                low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0]), 
                shape=(2,), dtype=np.float32)
            
            self.m_max_time = 20  # Max time for each episode
            self.m_reward = 0  # Reward for current step
            self.m_debug_reward = 0  # Reward for the whole episode
            # Position of goal as numpy array
            self.m_goal = None
            self.m_goal_vis = True
            # Distance to goal at previos time step
            self.m_vector_to_goal = None
            self.m_vector_to_goal_noNoise = None
            self.m_old_distance = None
            # Observation of the environment
            self.m_observation = None
            # Flag to determine if the environment has terminated -> In the event of timeOut or reach goal
            self.m_terminated = False
            # Flag to determine if the environment has truncated -> In the event of a crash
            self.m_truncated = False
            # Flag to check if the render setup has been done
            self.m_render_setup = False
            # Flag to count success while testing
            self.m_success = False
            # Flag to check if there is a plan to render or not
            self.m_play_mode = False
            self.m_additional_render_mode = additional_render_mode
            self.m_episode_num = 0
            self.m_success_count = 0
            self.m_crash_count = 0
            self.m_fallen_count = 0
            self.m_timeout_count = 0

            # Cropped image params
            self.run_count = 1
            self.base_dir = "./cropped_maps_train_noNorm"
            self.current_run_dir = os.path.join(self.base_dir, f"{self.run_count}")
            self.step_count = 1
            
            # Save the frames for video
            self.video_frames_dir = None
            self.render_frame = 0
        
        except Exception as e:
            print(f"Failed to initialize environment: {e}")
            raise e

    def reset(self, seed=None, options=None):
        """
        Reset the environment to its initial state -> Set up for standard gym API
        :param seed: Seed for the random number generator
        :param options: Options for the simulation (dictionary)
        """
        try:
            # -------------------------------
            # Reset Chrono system
            # -------------------------------
            self.m_system = chrono.ChSystemNSC()
            self.m_system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
            self.m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
            
            # -------------------------------
            # Reset the terrain
            # -------------------------------
            self.m_isFlat = False
            terrain_delta = 0.05
            self.m_isRigid = True
            self.m_step = 0

            texture_file_options = ["terrain/textures/grass.jpg", "terrain/textures/dirt.jpg", 
                                    "terrain/textures/Gravel034_1K-JPG/Gravel034_1K_Color.jpg", 
                                    "terrain/textures/concrete.jpg"]
            texture_file = texture_file_options[-1]
            
            if self.m_isRigid:
                self.m_terrain = veh.RigidTerrain(self.m_system)
                patch_mat = chrono.ChMaterialSurfaceNSC()
                patch_mat.SetFriction(0.9)
                patch_mat.SetRestitution(0.01)
                if self.m_isFlat:
                    patch = self.m_terrain.AddPatch(
                        patch_mat, chrono.CSYSNORM, self.m_terrain_length*1.5, self.m_terrain_width*1.5)
                    
                    half_length = self.m_terrain_length * 1.5 // 2
                    half_width = self.m_terrain_width * 1.5 // 2

                    top_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(0, half_width, 0), chrono.Q_ROTATE_Y_TO_Z)
                    top_patch = self.m_terrain.AddPatch(
                        patch_mat, top_patch_trans, self.m_terrain_length * 1.5, self.m_terrain_width / 3, 0.5)
                    bottom_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(0, -half_width, 0), chrono.Q_ROTATE_Y_TO_Z)
                    bottom_patch = self.m_terrain.AddPatch(
                        patch_mat, bottom_patch_trans, self.m_terrain_length * 1.5, self.m_terrain_width / 3, 0.5)
                    left_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(half_length, 0, 0), chrono.Q_ROTATE_X_TO_Z)
                    left_patch = self.m_terrain.AddPatch(
                        patch_mat, left_patch_trans, self.m_terrain_length / 3, self.m_terrain_width * 1.5, 0.5)
                    right_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(-half_length, 0, 0), chrono.Q_ROTATE_X_TO_Z)
                    right_patch = self.m_terrain.AddPatch(
                        patch_mat, right_patch_trans, self.m_terrain_length / 3, self.m_terrain_width * 1.5, 0.5)
                    
                else:
                    self.load_terrain()
                    # Some bitmap file backup (don't know why this is done in OG code)
                    bitmap_file_backup = os.path.dirname(os.path.realpath(
                        __file__)) + "/../data/terrain_bitmaps/height_map_backup.bmp"

                    rotation_quaternion = chrono.ChQuaternionD()
                    rotation_quaternion.Q_from_AngAxis(chrono.CH_C_PI, chrono.ChVectorD(0, 0, 1))
                    try:
                        patch = self.m_terrain.AddPatch(
                            patch_mat, chrono.ChCoordsysD(chrono.ChVectorD(0, 0, 0), rotation_quaternion), 
                            self.bitmap_file, self.m_terrain_length*1.5, 
                            self.m_terrain_width*1.5, self.m_min_terrain_height, self.m_max_terrain_height)
                        
                        half_length = self.m_terrain_length * 1.5 // 2
                        half_width = self.m_terrain_width * 1.5 // 2

                        top_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(0, half_width, 0), chrono.Q_ROTATE_Y_TO_Z)
                        top_patch = self.m_terrain.AddPatch(
                            patch_mat, top_patch_trans, self.m_terrain_length * 1.5, self.m_terrain_width / 3, 0.5)
                        bottom_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(0, -half_width, 0), chrono.Q_ROTATE_Y_TO_Z)
                        bottom_patch = self.m_terrain.AddPatch(
                            patch_mat, bottom_patch_trans, self.m_terrain_length * 1.5, self.m_terrain_width / 3, 0.5)
                        left_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(half_length, 0, 0), chrono.Q_ROTATE_X_TO_Z)
                        left_patch = self.m_terrain.AddPatch(
                            patch_mat, left_patch_trans, self.m_terrain_length / 3, self.m_terrain_width * 1.5, 0.5)
                        right_patch_trans = chrono.ChCoordsysD(chrono.ChVectorD(-half_length, 0, 0), chrono.Q_ROTATE_X_TO_Z)
                        right_patch = self.m_terrain.AddPatch(
                            patch_mat, right_patch_trans, self.m_terrain_length / 3, self.m_terrain_width * 1.5, 0.5)
                    except:
                        print('Corrupt Bitmap File')
                        patch = self.m_terrain.AddPatch(
                            patch_mat, chrono.CSYSNORM, bitmap_file_backup, self.m_terrain_length*1.5, 
                            self.m_terrain_width*1.5, self.m_min_terrain_height, self.m_max_terrain_height)

                patch.SetTexture(veh.GetDataFile(
                    texture_file), self.m_terrain_length*1.5, self.m_terrain_width*1.5)
                top_patch.SetTexture(veh.GetDataFile(
                    texture_file), self.m_terrain_length*1.5, self.m_terrain_width*1.5)
                bottom_patch.SetTexture(veh.GetDataFile(
                    texture_file), self.m_terrain_length*1.5, self.m_terrain_width*1.5)
                left_patch.SetTexture(veh.GetDataFile(
                    texture_file), self.m_terrain_length*1.5, self.m_terrain_width*1.5)
                right_patch.SetTexture(veh.GetDataFile(
                    texture_file), self.m_terrain_length*1.5, self.m_terrain_width*1.5)
                
                self.m_terrain.Initialize()

            else:
                # Real deformable terrain - Without mesh viz
                self.m_terrain = veh.SCMTerrain(self.m_system)
                # Set the SCM parameters
                terrain_params = SCMParameters()
                terrain_params.InitializeParametersAsMid()
                terrain_params.SetParameters(self.m_terrain)
                # Enable bulldozing effects
                self.m_terrain.EnableBulldozing(True)
                self.m_terrain.SetBulldozingParameters(
                    55,  # angle of friction for erosion of displaced material at the border of the rut
                    1,  # displaced material vs downward pressed material.
                    5,  # number of erosion refinements per timestep
                    10)  # number of concentric vertex selections subject to erosion
                self.m_terrain.SetPlane(chrono.ChCoordsysD(chrono.ChVectorD(0, -0.5, 0), chrono.Q_from_AngX(
                    -0)))

                if self.m_isFlat:
                    # Initialize the terrain using a flat patch
                    self.m_terrain.Initialize(
                        self.m_terrain_length * 1.5,  # Size in X direction
                        self.m_terrain_length * 1.5,  # Size in Y direction
                        terrain_delta)  # Mesh resolution
                else:
                    # Initialize the terrain using a bitmap for the height map
                    bitmap_file = os.path.dirname(os.path.realpath(
                        __file__)) + "/../data/terrain_bitmaps/height_map.bmp"

                    # Some bitmap file backup (don't know why this is done in OG code)
                    bitmap_file_backup = os.path.dirname(os.path.realpath(
                        __file__)) + "/../data/terrain_bitmaps/height_map_backup.bmp"

                    try:
                        self.m_terrain.Initialize(bitmap_file,  # heightmap file (.bmp)
                                                self.m_terrain_length * 1.5,  # sizeX
                                                self.m_terrain_width * 1.5,  # sizeY
                                                self.m_min_terrain_height,  # hMin
                                                self.m_max_terrain_height,  # hMax
                                                terrain_delta)  # mesh resolution
                    except Exception:
                        print('Corrupt Bitmap File')
                        self.m_terrain.Initialize(bitmap_file_backup,  # heightmap file (.bmp)
                                                self.m_terrain_length * 1.5,  # sizeX
                                                self.m_terrain_width * 1.5,  # sizeY
                                                self.m_min_terrain_height,  # hMin
                                                self.m_max_terrain_height,  # hMax
                                                terrain_delta)
            
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
            art_theta = self.initialize_vw_pos(seed)
            self.m_vehicle.Initialize()

            self.m_vehicle.LockAxleDifferential(0, True)    
            self.m_vehicle.LockAxleDifferential(1, True)
            self.m_vehicle.LockCentralDifferential(0, True)
            self.m_vehicle.LockCentralDifferential(1, True)
            self.m_vehicle.GetVehicle().EnableRealtime(False)

            # If we are visualizing, get mesh based
            if self.m_play_mode:
                self.m_vehicle.SetChassisVisualizationType(
                    veh.VisualizationType_MESH)
                self.m_vehicle.SetWheelVisualizationType(
                    veh.VisualizationType_MESH)
                self.m_vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
            else:
                self.m_vehicle.SetChassisVisualizationType(
                    veh.VisualizationType_PRIMITIVES)
                self.m_vehicle.SetWheelVisualizationType(
                    veh.VisualizationType_PRIMITIVES)
                self.m_vehicle.SetTireVisualizationType(
                    veh.VisualizationType_PRIMITIVES)
            self.m_vehicle.SetSuspensionVisualizationType(
                veh.VisualizationType_MESH)
            self.m_vehicle.SetSteeringVisualizationType(
                veh.VisualizationType_MESH)
            self.m_chassis_body = self.m_vehicle.GetChassisBody()

            # Set the driver
            self.m_driver = veh.ChDriver(self.m_vehicle.GetVehicle())
            self.m_driver_inputs = self.m_driver.GetInputs()

            # Set PID controller for speed
            self.m_speedController = veh.ChSpeedController()
            self.m_speedController.Reset(self.m_vehicle.GetRefFrame())
            self.m_speedController.SetGains(1, 0, 0)
        
            # ===============================
            # Add the moving terrain patches
            # ===============================
            if (self.m_isRigid == False):
                self.m_terrain.AddMovingPatch(self.m_chassis_body, chrono.ChVectorD(
                    0, 0, 0), chrono.ChVectorD(5, 3, 1))
                # Set a texture for the terrain
                self.m_terrain.SetTexture(veh.GetDataFile(
                    texture_file), self.m_terrain_length*2, self.m_terrain_width*2)

                # Set some vis
                self.m_terrain.SetPlotType(
                    veh.SCMTerrain.PLOT_PRESSURE, 0, 30000.2)

            # -------------------------------
            # Set the goal point
            # -------------------------------
            self.set_goal(art_theta, seed)

            # -----------------------------------
            # Prepare new directory for the run
            # -----------------------------------
            self.current_run_dir = os.path.join(self.base_dir, f"{self.run_count}")
            self.step_count = 1  # Reset step count for new run.
            self.run_count += 1  # Increment run count for the next run

            # -------------------------------
            # Initialize the sensors
            # -------------------------------
            if hasattr(self, 'm_sens_manager'):
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
            self.m_contact_force = 0
            self.m_debug_reward = 0
            self.m_reward = 0
            self.m_render_setup = False
            # Success count for eval
            self.m_success_count_eval = 0
            self.m_terminated = False
            self.m_truncated = False

            return self.m_observation, {}
        
        except Exception as e:
            logging.exception("Exception in reset method")
            print(f"Failed to reset environment: {e}")
            raise e

    def step(self, action):
        """
        One step of simulation. Get the driver input from simulation
            Steering: [-1, 1], -1 is right, 1 is left
            Speed: [-4, 4] => [-1, 1]/[0, 1]
        """
        try:
            # # part-normalize
            # steering = float(action[0])
            # speed = float(action[1])
            
            # normalize
            steering = float(action[0])
            normalized_speed = float(action[1])
            speed = normalized_speed * self.max_speed
            
            time = self.m_system.GetChTime()
            # Desired throttle/braking value
            out_throttle = self.m_speedController.Advance(
                self.m_vehicle.GetRefFrame(), speed, time, self.m_step_size)
            out_throttle = np.clip(out_throttle, -1, 1)
            
            if out_throttle >= 0:
                self.m_driver_inputs.m_braking = 0
                self.m_driver_inputs.m_throttle = out_throttle
            else:
                self.m_driver_inputs.m_braking = -out_throttle
                self.m_driver_inputs.m_throttle = 0

            # Apply the steering input directly without smoothing
            self.m_driver_inputs.m_steering = np.clip(steering, -1.0, 1.0)
            
            for _ in range(0, self.m_steps_per_control):
                time = self.m_system.GetChTime()
                # Update modules (process inputs from other modules)
                self.m_terrain.Synchronize(time)
                self.m_vehicle.Synchronize(time, self.m_driver_inputs, self.m_terrain)
                if (self.m_render_setup and self.render_mode == 'follow'):
                    self.vis.Synchronize(time, self.m_driver_inputs)

                # Advance simulation for one timestep for all modules
                self.m_driver.Advance(self.m_step_size)
                self.m_terrain.Advance(self.m_step_size)
                self.m_vehicle.Advance(self.m_step_size)
                if (self.m_render_setup and self.render_mode == 'follow'):
                    self.vis.Advance(self.m_step_size)

                self.m_system.DoStepDynamics(self.m_step_size)
                # Sensor update
                self.m_sens_manager.Update()

            # Get the observation
            self.m_observation = self.get_observation()
            self.m_reward = self.get_reward()
            self.m_debug_reward += self.m_reward
            
            # Check if the vehicle has on the terrain
            self.m_step += 1

            self._is_terminated()
            self._is_truncated()

            return self.m_observation, self.m_reward, self.m_terminated, self.m_truncated, {}

        except Exception as e:
            logging.exception("Exception in step method")
            print(f"Error during step execution: {e}")
            raise e

    def render(self, mode='follow', headless=False):
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
                self.vis.AddCamera(chrono.ChVectorD(
                    0, 0, 80), chrono.ChVectorD(0, 0, 1))
                self.vis.AddTypicalLights()
                self.vis.AddLightWithShadow(chrono.ChVectorD(
                    1.5, -2.5, 5.5), chrono.ChVectorD(0, 0, 0.5), 3, 4, 10, 40, 512)
                self.m_render_setup = True
        
            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()

        elif mode == 'follow':
            self.render_mode = 'follow'
            if self.m_render_setup == False:
                self.vis = veh.ChWheeledVehicleVisualSystemIrrlicht()
                self.vis.SetWindowTitle('vws in the wild')
                self.vis.SetWindowSize(2560, 1440)
                trackPoint = chrono.ChVectorD(-3, -3, 2) #-3, 0.0, 2
                self.vis.SetChaseCamera(trackPoint, 3, 1)
                self.vis.Initialize()
                self.vis.AddLightDirectional()
                self.vis.AddSkyBox()
                self.vis.AttachVehicle(self.m_vehicle.GetVehicle())
                self.m_render_setup = True

            self.vis.BeginScene()
            self.vis.Render()
            self.vis.EndScene()

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
            logging.debug("Starting get_observation function")
            self.m_vehicle_pos = self.m_chassis_body.GetPos()
        
            # Get GPS info
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

            self.m_vector_to_goal = self.m_goal - cur_gps_data
            self.m_vector_to_goal_noNoise = self.m_goal - self.m_vehicle_pos
            vector_to_goal_local = self.m_chassis_body.GetRot().RotateBack(self.m_vector_to_goal)

            vehicle_x = -self.m_vehicle_pos.x
            vehicle_y = self.m_vehicle_pos.y

            image = Image.open(self.bitmap_file)
            image_edge = cv2.imread(self.bitmap_file, cv2.IMREAD_GRAYSCALE)
            bitmap_array = np.array(image)
            
            if np.all(image_edge == 0):
                self.bmp_dim_x = self.m_terrain_width
                self.bmp_dim_y = self.m_terrain_length
            else:
                min_y_index = np.min(np.where(image_edge > 0)[1])  
                max_y_index = np.max(np.where(image_edge > 0)[1])  
                min_x_index = np.min(np.where(image_edge > 0)[0])  
                max_x_index = np.max(np.where(image_edge > 0)[0])

                self.bmp_dim_x = max_x_index - min_x_index #width
                self.bmp_dim_y = max_y_index - min_y_index #length
                
            terrain_length_tolerance = self.m_terrain_length * 1.5
            terrain_width_tolerance = self.m_terrain_width * 1.5

            self.map_offset_x = terrain_width_tolerance / 2
            self.map_offset_y = terrain_length_tolerance / 2

            # Normalization scaling factors
            s_norm_x = 1 / terrain_width_tolerance
            s_norm_y = 1 / terrain_length_tolerance
            s_x = s_norm_x * self.bmp_dim_x
            s_y = s_norm_y * self.bmp_dim_y

            # Transformation matrix 
            T = np.array([
                [s_y, 0, 0],
                [0, s_x, 0],
                [0, 0, 1]
            ])

            pos_chrono = np.array([vehicle_x + self.map_offset_y, 
                                vehicle_y, 1])

            pos_bmp = np.dot(T, pos_chrono)
            pos_bmp_x, pos_bmp_y = bitmap_array.shape[1] // 2 + int(pos_bmp[1]), int(pos_bmp[0])

            # Check if pos_bmp_x and pos_bmp_y are within bounds
            assert 0 <= pos_bmp_x < bitmap_array.shape[0], f"pos_bmp_x out of bounds: {pos_bmp_x}"
            assert 0 <= pos_bmp_y < bitmap_array.shape[1], f"pos_bmp_y out of bounds: {pos_bmp_y}"

            # x-axis: back, y-axis: right, z-axis: up
            vehicle_heading_global = self.m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z

            center_x = bitmap_array.shape[0] // 2
            center_y = bitmap_array.shape[1] // 2
            shift_x = center_x - pos_bmp_x
            shift_y = center_y - pos_bmp_y

            shifted_map = np.roll(bitmap_array, shift_x, axis=0)
            shifted_map = np.roll(shifted_map, shift_y, axis=1)
            shifted_map = np.expand_dims(shifted_map, axis=0)
            r_map = torch.tensor(shifted_map)
            
            if(np.degrees(vehicle_heading_global) < 0):
                angle = np.degrees(np.pi + vehicle_heading_global)
            elif(np.degrees(vehicle_heading_global) >= 0):
                angle = np.degrees(-np.pi + vehicle_heading_global)
            rotated_map = np.array((F.rotate(r_map, angle)).squeeze().cpu(), dtype=np.uint8)

            half_size_x = self.submap_shape_x // 2
            half_size_y = self.submap_shape_y // 2

            # Symmetric Observation
            start_y = center_y - half_size_y
            end_y = center_y + half_size_y
            start_x = center_x - half_size_x
            end_x = center_x + half_size_x

            # Calculate start and end indices for the sub-array
            start_y = max(start_y, 0)
            end_y = min(end_y, rotated_map.shape[0])
            start_x = max(start_x, 0)
            end_x = min(end_x, rotated_map.shape[1])

            if end_y - start_y < self.submap_shape_y:
                if start_y == 0:
                    # If we're at the lower boundary, adjust end_y up
                    end_y = min(start_y + self.submap_shape_y, rotated_map.shape[0])
                elif end_y == rotated_map.shape[0]:
                    # If we're at the upper boundary, adjust start_y down
                    start_y = max(end_y - self.submap_shape_y, 0)

            if end_x - start_x < self.submap_shape_x:
                if start_x == 0:
                    # If we're at the left boundary, adjust end_x right
                    end_x = min(start_x + self.submap_shape_x, rotated_map.shape[1])
                elif end_x == rotated_map.shape[1]:
                    # If we're at the right boundary, adjust start_x left
                    start_x = max(end_x - self.submap_shape_x, 0)

            # Extract the sub-array
            sub_array = rotated_map[start_x:end_x, start_y:end_y]
            
            # # Collect data for SWAE model
            # self.collect_data(sub_array)

            # Normalize the sub-array to [-1, 1] before passing to the network
            min_val = np.min(sub_array)
            max_val = np.max(sub_array)
            if max_val > min_val: 
                sub_array = 2 * ((sub_array - min_val) / (max_val - min_val)) - 1
            else:
                sub_array = np.zeros_like(sub_array) 

            flattened_map = sub_array.flatten()
            flattened_map_tensor = torch.tensor(flattened_map, dtype=torch.float32).unsqueeze(0).to(self.device)
            flattened_map_tensor = flattened_map_tensor.view(-1, 1, self.submap_shape_x, self.submap_shape_y)

            # -----------------------------------------
            # Here is the feature extraction for SWAE
            # -----------------------------------------
            _, _, z = self.swae(flattened_map_tensor) #64*64 -> 64*1
            
            # # Part-normalize Observation
            # z_normalized = 2 * (z - self.min_vectorMCL) / (self.max_vectorMCL - self.min_vectorMCL) - 1
            # mapped_features_tensor = self.latent_space_mapper(z_normalized)
            # mapped_features_array = mapped_features_tensor.cpu().detach().numpy().flatten()
                  
            # vector_to_goal_local = self.m_chassis_body.GetRot().RotateBack(self.m_vector_to_goal)
            # target_heading_local = np.arctan2(vector_to_goal_local.y, vector_to_goal_local.x)
            # heading_diff_local = (target_heading_local - 0 + np.pi) % (2 * np.pi) - np.pi
            # vehicle_speed = self.m_chassis_body.GetPos_dt().Length()
            
            # observation_array = np.array([heading_diff_local, vehicle_speed])
            # final_observation = np.concatenate((mapped_features_array, observation_array)).astype(np.float32)
            # return final_observation
            
            # Normalize Observation
            z_normalized = 2 * (z - self.min_vectorACL) / (self.max_vectorACL - self.min_vectorACL) - 1
            mapped_features_tensor = self.latent_space_mapper(z_normalized)
            mapped_features_array = mapped_features_tensor.cpu().detach().numpy().flatten()
                  
            vector_to_goal_local = self.m_chassis_body.GetRot().RotateBack(self.m_vector_to_goal)
            target_heading_local = np.arctan2(vector_to_goal_local.y, vector_to_goal_local.x)
            heading_diff_local = (target_heading_local - 0 + np.pi) % (2 * np.pi) - np.pi
            normalized_heading_diff = heading_diff_local / np.pi
            
            vehicle_speed = self.m_chassis_body.GetPos_dt().Length()
            normalized_speed = vehicle_speed / self.max_speed
            # Normalize speed to [-1, 1]
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
        
        distance = self.m_vector_to_goal_noNoise.Length()
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
        if np.linalg.norm(self.m_vector_to_goal_noNoise.Length()) < 4:
            print('--------------------------------------------------------------')
            print('Goal Reached')
            print('Initial position: ', self.m_initLoc)
            print('Goal position: ', self.m_goal)
            print('--------------------------------------------------------------')
            self.m_reward += 3000
            self.m_debug_reward += self.m_reward
            self.m_terminated = True
            self.m_success_count += 1
            self.m_success_count_eval += 1
            self.m_episode_num += 1

        # If we have exceeded the max time -> Terminate and give penalty for how far we are from the goal
        if self.m_system.GetChTime() > self.m_max_time:
            print('--------------------------------------------------------------')
            print('Time out')
            print('Initial position: ', self.m_initLoc)
            dist = self.m_vector_to_goal_noNoise.Length()
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
        """
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
        For now just checks if the CG of the vehicle is within the rectangle bounds with some tolerance
        """
        terrain_length_tolerance = self.m_terrain_length
        terrain_width_tolerance = self.m_terrain_width

        vehicle_is_outside_terrain = abs(self.m_vehicle_pos.x) > terrain_length_tolerance or abs(
            self.m_vehicle_pos.y) > terrain_width_tolerance
        if (vehicle_is_outside_terrain):
            return True
        else:
            return False

    def initialize_vw_pos(self, seed):
        """
        Initialize the robot position
        :param seed: Seed for the random number generator
        :return: Random angle between 0 and 2pi along which art is oriented
         """
        # Random angle between 0 and 2pi
        theta = random.random() * 2 * np.pi
        x, y = self.m_terrain_length * 0.5 * \
            np.cos(theta) + 2, self.m_terrain_width * 0.5 * np.sin(theta)
        z = self.m_terrain.GetHeight(chrono.ChVectorD(x, y, 0)) + 3
        ang = np.pi + theta
        self.m_initLoc = chrono.ChVectorD(x, y, z)
        self.m_initRot = chrono.Q_from_AngZ(ang)
        self.m_vehicle.SetInitPosition(
            chrono.ChCoordsysD(self.m_initLoc, self.m_initRot))
        return theta
        

    def set_goal(self, art_theta, seed):
        """
        Set the goal point
        :param seed: Seed for the random number generator
        """
        # Random angle between -pi/2 and pi/2
        delta_theta = (random.random() - 0.5) * 1.0 * np.pi
        # Goal is always an angle between -pi/2 and pi/2 from the art
        gx, gy = self.m_terrain_length * 0.5 * np.cos(art_theta + np.pi + delta_theta), self.m_terrain_width * 0.5 * np.sin(
            art_theta + np.pi + delta_theta)
        self.m_goal = chrono.ChVectorD(
            gx, gy, self.m_terrain.GetHeight(chrono.ChVectorD(gx, gy, 0)) + 2.5)
    
        i = 0
        while (self.m_goal - self.m_initLoc).Length() < 20:
            gx = random.random() * self.m_terrain_length - self.m_terrain_length / 2
            gy = random.random() * self.m_terrain_width - self.m_terrain_width / 2
            self.m_goal = chrono.ChVectorD(
                gx, gy, self.m_terrain.GetHeight(chrono.ChVectorD(gx, gy, 0)) + 3)
            if i > 100:
                print('Failed setting goal randomly, using default')
                gx = self.m_terrain_length * 0.625 * \
                    np.cos(art_theta + np.pi + delta_theta)
                gy = self.m_terrain_width * 0.625 * \
                    np.sin(art_theta + np.pi + delta_theta)
                break
            i += 1

        # Set the goal visualization
        if (self.m_goal_vis):
            goal_contact_material = chrono.ChMaterialSurfaceNSC()
            goal_mat = chrono.ChVisualMaterial()
            goal_mat.SetAmbientColor(chrono.ChColor(1., 0., 0.))
            goal_mat.SetDiffuseColor(chrono.ChColor(1., 0., 0.))

            goal_body = chrono.ChBodyEasySphere(
                0.35, 1000, True, False, goal_contact_material)

            goal_body.SetPos(self.m_goal)
            goal_body.SetBodyFixed(True)
            goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)

            self.m_system.Add(goal_body)

    def collect_data(self, sub_array):
        """Save the sub-array to a BMP file under the current run directory."""
        if not os.path.exists(self.base_dir):
            os.makedirs(self.base_dir)
        if not os.path.exists(self.current_run_dir):
            os.makedirs(self.current_run_dir)
        filename = f"step_{self.step_count}.bmp"
        filepath = os.path.join(self.current_run_dir, filename)
        cv2.imwrite(filepath, sub_array)
        self.step_count += 1

    def load_terrain(self):
        # Initialize the terrain using a bitmap for the height map
        terrain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                "../data/terrain_bitmaps/Automatic-CL/TrainLevels",
                                self.terrain_file[self.current_terrain_stage])
        if not os.path.exists(terrain_path):
            print("Terrain file does not exist:", terrain_path)
            return
    
        self.bitmap_file = terrain_path
        if not self.terrain_loaded:
            print(f"Terrain loaded from {terrain_path}")
            self.terrain_loaded = True
        
    def update_terrain_stage(self, level_index):
        self.current_terrain_stage = level_index
        self.terrain_loaded = False
        self.load_terrain()
        
    def get_vehicle_position_on_bmp(self):
        self.m_vehicle_pos = self.m_chassis_body.GetPos()
        vehicle_x = -self.m_vehicle_pos.x
        vehicle_y = self.m_vehicle_pos.y

        terrain_length_tolerance = self.m_terrain_length * 1.5
        terrain_width_tolerance = self.m_terrain_width * 1.5

        # Normalization scaling factors
        s_norm_x = 1 / terrain_width_tolerance
        s_norm_y = 1 / terrain_length_tolerance
        s_x = s_norm_x * self.bmp_dim_x
        s_y = s_norm_y * self.bmp_dim_y

        # Transformation matrix 
        T = np.array([
            [s_y, 0, 0],
            [0, s_x, 0],
            [0, 0, 1]
        ])

        pos_chrono = np.array([vehicle_x + self.map_offset_y, 
                            vehicle_y + self.map_offset_x, 1])

        pos_bmp = np.dot(T, pos_chrono)
        pos_bmp_x, pos_bmp_y = int(pos_bmp[1]), int(pos_bmp[0])

        return pos_bmp_x, pos_bmp_y
        
    def seed(self, seed=None):
        """
        Seed the environment's random number generator
        :param seed: Seed for the random number generator
        """
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        return [seed]
        
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

    def set_nice_vehicle_mesh(self):
        self.m_play_mode = True
        
    def capture_frame(self):
        filename = os.path.join(self.video_frames_dir, 'img_' + str(self.render_frame).zfill(4) + '.jpg')
        self.vis.WriteImageToFile(filename)
        self.render_frame += 1

    def close(self):
        try:
            del self.m_vehicle
            if hasattr(self, 'm_sens_manager'):
                del self.m_sens_manager
            if hasattr(self, 'm_system'):
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
