import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os
import math
import random
import json
import heapq
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from PIL import Image

from verti_bench.envs.utils.utils import SetChronoDataDirectories
import pychrono.sensor as sens

import numpy as np
import math
import cv2
from datetime import datetime
import traceback
from matplotlib.cm import get_cmap
from PIL import Image
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
from multiprocessing import Pool
import copy
import cv2

import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.parallel as parallel
import csv

class MPPIController:
    def __init__(self, num_samples, horizon, dt, lambda_, wheelbase, goal, terrain_file, terrain_length, terrain_width, min_height, max_height):
        self.num_samples = num_samples
        self.horizon = horizon
        self.dt = dt
        self.lambda_ = lambda_
        self.wheelbase = wheelbase
        self.goal = goal
        self.max_steer_angle = 0.5  # ~30 degrees max steering angle
        self.max_speed = 4.0  # Maximum speed in m/s
        self.min_speed = 0.0   # Minimum speed in m/s
        self.crop_size = 5  # Size of elevation map crop (5x5 meters)
        
        # Load and process elevation map
        self.elevation_map = self.load_elevation_map(terrain_file)
        self.terrain_length = terrain_length
        self.terrain_width = terrain_width
        self.min_height = min_height
        self.max_height = max_height
        
        # Calculate conversion factors from world coordinates to pixel coordinates
        self.pixels_per_meter_x = self.elevation_map.shape[1] / terrain_length
        self.pixels_per_meter_y = self.elevation_map.shape[0] / terrain_width
        
        # Calculate terrain center offset
        self.terrain_center_x = terrain_length / 2
        self.terrain_center_y = terrain_width / 2

    def load_elevation_map(self, terrain_file):
        """Load and process the BMP elevation map"""
        # Read the BMP file in grayscale
        elevation_map = cv2.imread(terrain_file, cv2.IMREAD_GRAYSCALE)
        if elevation_map is None:
            raise ValueError(f"Failed to load terrain file: {terrain_file}")
        return elevation_map

    def world_to_pixel_coordinates(self, world_x, world_y):
        """Convert world coordinates to pixel coordinates"""
        # Adjust coordinates relative to terrain center
        rel_x = world_x + self.terrain_center_x
        rel_y = world_y + self.terrain_center_y
        
        # Convert to pixel coordinates
        pixel_x = int(rel_x * self.pixels_per_meter_x)
        pixel_y = int(rel_y * self.pixels_per_meter_y)
        
        # Ensure coordinates are within bounds
        pixel_x = np.clip(pixel_x, 0, self.elevation_map.shape[1] - 1)
        pixel_y = np.clip(pixel_y, 0, self.elevation_map.shape[0] - 1)
        
        return pixel_x, pixel_y

    def get_elevation_crop(self, position):
        """Get elevation values for a cropped area around the position from BMP"""
        crop_half = self.crop_size / 2
        elevations = []
        
        # Sample points in a grid around the position
        for dx in np.linspace(-crop_half, crop_half, 5):  # 5x5 sampling grid
            for dy in np.linspace(-crop_half, crop_half, 5):
                x = position[0] + dx
                y = position[1] + dy
                
                # Convert world coordinates to pixel coordinates
                pixel_x, pixel_y = self.world_to_pixel_coordinates(x, y)
                
                # Get normalized elevation value from the elevation map
                elevation_normalized = self.elevation_map[pixel_y, pixel_x] / 255.0
                
                # Convert to actual elevation
                elevation = self.min_height + elevation_normalized * (self.max_height - self.min_height)
                elevations.append(elevation)
                
        return np.mean(elevations)

    def compute_control(self, current_pos, current_yaw, m_vehicle, m_terrain):
        controls = []
        costs = []
        
        # Get current speed and elevation
        current_speed = m_vehicle.GetVehicle().GetSpeed()
        current_elevation = self.get_elevation_crop([current_pos.x, current_pos.y])
        
        # Calculate angle to goal
        goal_angle = np.arctan2(self.goal.y - current_pos.y, self.goal.x - current_pos.x)
        angle_diff = self.normalize_angle(goal_angle - current_yaw)
        
        for _ in range(self.num_samples):
            trajectory, roll_values, pitch_values, elevation_values, control_inputs = self.rollout_trajectory(
                current_pos, current_yaw, current_speed, current_elevation, angle_diff, m_vehicle
            )
            cost = self.calculate_cost(trajectory, roll_values, pitch_values, elevation_values, control_inputs)
            controls.append(control_inputs)
            costs.append(cost)

        # Apply softmax weighting
        costs = np.array(costs)
        min_cost = np.min(costs)  # Numerical stability
        weights = np.exp(-(costs - min_cost) / self.lambda_)
        weights /= np.sum(weights)

        # Weighted average of controls
        optimal_control = np.average(np.array(controls), axis=0, weights=weights)
        
        # Return first control input
        return optimal_control[0]

    def rollout_trajectory(self, current_pos, current_yaw, current_speed, current_elevation, angle_diff, m_vehicle):
        trajectory = []
        roll_values = []
        pitch_values = []
        elevation_values = []  # Store elevation differences
        control_inputs = []

        pos = np.array([current_pos.x, current_pos.y])
        yaw = current_yaw
        speed = current_speed
        prev_elevation = current_elevation

        # Base steering angle on angle to goal
        base_steer = np.clip(angle_diff, -self.max_steer_angle, self.max_steer_angle)

        for _ in range(self.horizon):
            # Sample control inputs with bias towards goal direction
            delta_steer = np.clip(
                base_steer + np.random.normal(0, 0.3),
                -self.max_steer_angle,
                self.max_steer_angle
            )
            
            # Adjust speed based on steering angle
            target_speed = self.max_speed * (1 - abs(delta_steer) / self.max_steer_angle)
            target_speed = max(target_speed, self.min_speed)
            
            # Calculate required acceleration
            acceleration = np.clip((target_speed - speed) / self.dt, 0, 1)

            # Update state
            speed = speed + acceleration * self.dt
            pos[0] += speed * np.cos(yaw) * self.dt
            pos[1] += speed * np.sin(yaw) * self.dt
            yaw += (speed / self.wheelbase) * np.tan(delta_steer) * self.dt
            
            # Get vehicle orientation
            euler_angles = m_vehicle.GetVehicle().GetRot().Q_to_Euler123()
            roll, pitch = euler_angles.y, euler_angles.x
            
            # Get elevation at new position from BMP
            new_elevation = self.get_elevation_crop(pos)
            elevation_diff = abs(new_elevation - prev_elevation)
            prev_elevation = new_elevation

            trajectory.append((pos[0], pos[1]))
            roll_values.append(roll)
            pitch_values.append(pitch)
            elevation_values.append(elevation_diff)
            control_inputs.append((delta_steer, acceleration))

        return trajectory, roll_values, pitch_values, elevation_values, control_inputs

    def calculate_cost(self, trajectory, roll_values, pitch_values, elevation_values, control_inputs):
        cost = 0
        goal_np = np.array([self.goal.x, self.goal.y])

        prev_steering = 0
        for i, (traj_point, control, elevation_diff) in enumerate(zip(trajectory, control_inputs, elevation_values)):
            traj_np = np.array(traj_point)
            steering, acceleration = control

            # Distance to goal cost
            distance_to_goal = np.linalg.norm(traj_np - goal_np)
            cost += 100 * distance_to_goal ** 2

            # Penalize steering changes (smooth steering)
            steering_change = abs(steering - prev_steering)
            cost += 50 * steering_change
            prev_steering = steering

            # Stability costs
            roll = roll_values[i]
            pitch = pitch_values[i]
            cost += 100 * (abs(roll) ** 2)  # Heavily penalize roll
            cost += 100 * (abs(pitch) ** 2)  # Heavily penalize pitch
            
            # Elevation difference cost
            cost += 200 * elevation_diff ** 2  # Penalize large elevation changes

        return cost

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

class RigidProperties:
    def __init__(self):
        # Friction coefficients: (mean, standard_deviation)
        self.friction_properties = {
            'clay': (0.4, 0.05),    
            'concrete': (0.7, 0.08), 
            'dirt': (0.6, 0.1),     
            'grass': (0.35, 0.07),  
            'gravel': (0.6, 0.1),   
            'rock': (0.65, 0.08),   
            'wood': (0.45, 0.06)    
        }
        
        # Restitution coefficients: (mean, standard_deviation)
        self.restitution_properties = {
            'clay': (0.1, 0.02),    
            'concrete': (0.85, 0.05), 
            'dirt': (0.3, 0.05),     
            'grass': (0.5, 0.08),    
            'gravel': (0.4, 0.07),   
            'rock': (0.8, 0.05),     
            'wood': (0.5, 0.06)      
        }
        
    def sample_properties(self, terrain_type):
        """
        Sample friction and restitution values from Gaussian distributions
        for the given terrain type.
        """
        # Sample friction coefficient
        friction_mean, friction_std = self.friction_properties[terrain_type]
        friction = np.random.normal(friction_mean, friction_std)
        friction = np.clip(friction, 0.1, 1.0)
        
        # Sample restitution coefficient
        restitution_mean, restitution_std = self.restitution_properties[terrain_type]
        restitution = np.random.normal(restitution_mean, restitution_std)
        restitution = np.clip(restitution, 0.0, 1.0)
        
        return friction, restitution


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
    def InitializeParametersAsSoft(self):
        self.Bekker_Kphi = 0.2e6
        self.Bekker_Kc = 0
        self.Bekker_n = 1.1
        self.Mohr_cohesion = 0
        self.Mohr_friction = 30
        self.Janosi_shear = 0.01
        self.elastic_K = 4e7
        self.damping_R = 3e4


    # Middle default parameters
    def InitializeParametersAsMid(self):
        self.Bekker_Kphi = 2e6
        self.Bekker_Kc = 0
        self.Bekker_n = 1.1
        self.Mohr_cohesion = 0
        self.Mohr_friction = 30
        self.Janosi_shear = 0.01
        self.elastic_K = 2e8
        self.damping_R = 3e4


    # Hard default parameters
    def InitializeParametersAsHard(self):
        self.Bekker_Kphi = 5301e3
        self.Bekker_Kc = 102e3
        self.Bekker_n = 0.793
        self.Mohr_cohesion = 1.3e3
        self.Mohr_friction = 31.1
        self.Janosi_shear = 1.2e-2
        self.elastic_K = 4e8
        self.damping_R = 3e4

def visualize_path(self, path):
    plt.figure(figsize=(8, 8))
    plt.imshow(self.grid_map.T, cmap="gray", origin="lower")
    plt.scatter(self.start[0], self.start[1], color="green", label="Start")
    plt.scatter(self.goal[0], self.goal[1], color="red", label="Goal")

    if path:
        x, y = zip(*path)
        plt.plot(x, y, color="blue", linewidth=2, label="Path")
    else:
        print("No valid path found!")

    plt.legend()
    plt.title("A* Path Planning")
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.grid(True)
    plt.show()

def get_random_start_pos_and_yaw(xlim, ylim, map_center, buffer, z_height):
    # Adjust x and y limits to incorporate the buffer
    new_xlim = [xlim[0] + buffer, xlim[1] - buffer]
    new_ylim = [ylim[0] + buffer, ylim[1] - buffer]

    # Select a random side to spawn on
    side = np.random.randint(0, 4)
    pos = [0, 0, z_height]
    if side == 0:
        # Spawn on the top side
        pos[0] = np.random.uniform(*new_xlim)
        pos[1] = new_ylim[1]
    elif side == 1:
        # Spawn on the left side
        pos[0] = new_xlim[0]
        pos[1] = np.random.uniform(*new_ylim)
    elif side == 2:
        # Spawn on the bottom side
        pos[0] = np.random.uniform(*new_xlim)
        pos[1] = new_ylim[0]
    else:
        # Spawn on the right side
        pos[0] = new_xlim[1]
        pos[1] = np.random.uniform(*new_ylim)
    
    # Calculate the yaw angle so it points towards the map center
    vec_start_to_center = np.array(map_center[:2]) - np.array(pos[:2])
    yaw = np.arctan2(vec_start_to_center[1], vec_start_to_center[0])

    return pos, yaw

def initialize_vw_pos(m_vehicle, m_terrain, m_terrain_length, m_terrain_width, m_terrain_center, seed=None):
    map_xlim = np.array([m_terrain_center[0] - m_terrain_length/2, m_terrain_center[0] + m_terrain_length/2])
    map_ylim = np.array([m_terrain_center[1] - m_terrain_width/2, m_terrain_center[1] + m_terrain_width/2])
    m_initLoc, m_initYaw = get_random_start_pos_and_yaw(map_xlim, map_ylim, m_terrain_center, 2.0, z_height=0.0)
    
    m_initLoc=[-35, 40, -1]
    m_initYaw = -np.pi / 6 
    m_initLoc = chrono.ChVectorD(*m_initLoc)
    m_initRot = chrono.Q_from_AngZ(m_initYaw)
    m_vehicle.SetInitPosition(chrono.ChCoordsysD(m_initLoc, m_initRot))
    return m_initLoc, m_initRot, m_initYaw

def set_goal(m_terrain, m_system, m_terrain_length, m_terrain_width, m_initLoc, m_terrain_center, buffer=4.0, z_offset=1.0):
    map_xlim = [m_terrain_center[0] - m_terrain_length / 2 + buffer, m_terrain_center[0] + m_terrain_length / 2 - buffer]
    map_ylim = [m_terrain_center[1] - m_terrain_width / 2 + buffer, m_terrain_center[1] + m_terrain_width / 2 - buffer]
    
    while True:
        gx = np.random.uniform(*map_xlim)
        gy = np.random.uniform(*map_ylim)
        gz = m_terrain.GetHeight(chrono.ChVectorD(gx, gy, 0)) + z_offset
        
        gx, gy, gz = 35, -20, -1
        m_goal = chrono.ChVectorD(gx, gy, gz)
        
        if (m_goal - m_initLoc).Length() > 30:
            break

    # Create goal sphere with visualization settings
    goal_contact_material = chrono.ChMaterialSurfaceNSC()
    goal_body = chrono.ChBodyEasySphere(0.35, 1000, True, False, goal_contact_material)
    goal_body.SetPos(m_goal)
    goal_body.SetBodyFixed(True)
    
    # Apply red visualization material
    goal_mat = chrono.ChVisualMaterial()
    goal_mat.SetAmbientColor(chrono.ChColor(1, 0, 0))  # Bright red color for visibility
    goal_mat.SetDiffuseColor(chrono.ChColor(1, 0, 0))
    goal_body.GetVisualShape(0).SetMaterial(0, goal_mat)
    
    # Add the goal body to the system
    m_system.Add(goal_body)

    return m_goal

def get_patch_center_position(i, j, patch_size, terrain_length, terrain_width):
    """
    Calculate the center position of each small patch based on its index (i, j).
    """
    x_start = j * patch_size
    y_start = i * patch_size
    x_center = -(terrain_length + 1) / 2 + (x_start + (patch_size / 2))
    y_center = (terrain_width + 1) / 2 - (y_start + (patch_size / 2))
    return x_center, y_center, 0

def divide_terrain_image(terrain_path, output_folder, patch_size=65):
    """
    Divide the terrain image into smaller patches and save them.
    """
    if os.path.exists(output_folder):
        shutil.rmtree(output_folder)
        
    os.makedirs(output_folder, exist_ok=True)
    terrain_image = Image.open(terrain_path)
    terrain_array = np.array(terrain_image)

    num_patches_x = (terrain_array.shape[0] - 1) // (patch_size - 1)
    num_patches_y = (terrain_array.shape[1] - 1) // (patch_size - 1)

    patches = []
    for i in range(num_patches_x):
        for j in range(num_patches_y):
            # Calculate patch boundaries with one pixel overlap
            x_start = i * (patch_size - 1)
            y_start = j * (patch_size - 1)
            x_end = min(x_start + patch_size, terrain_array.shape[0])
            y_end = min(y_start + patch_size, terrain_array.shape[1])

            # Extract patch with overlapping boundaries
            patch = terrain_array[x_start:x_end, y_start:y_end]
            
            patch_image = Image.fromarray(patch)
            patch_filename = os.path.join(output_folder, f"patch_{i}_{j}.bmp")
            patch_image.save(patch_filename)

            # Calculate center position
            x_center, y_center, z_center = get_patch_center_position(i, j, patch_size, 129, 129)

            patches.append((patch_filename, i, j, (x_center, y_center, z_center)))

    return patches


# Texture classification
def assign_textures_to_patches(num_textures, patches, num_clusters=10, terrain_path=None, patch_size=65):
    """
    Assign textures to patches ensuring that same texture types have contiguous boundaries.
    """
    assert num_textures >= num_clusters, "Number of textures must be >= number of clusters."
    assert terrain_path is not None, "Terrain path must be provided."
    
    # Extract patch indices (spatial coordinates)
    patch_indices = [(i, j) for _, i, j, _ in patches]
    patch_coords = np.array(patch_indices)
    mean_i = np.mean(patch_coords[:, 0])
    mean_j = np.mean(patch_coords[:, 1])
    std_i = np.std(patch_coords[:, 0])
    std_j = np.std(patch_coords[:, 1])
    
    cluster_centers = []
    while len(cluster_centers) < num_clusters:
        i = int(np.random.normal(mean_i, std_i))
        j = int(np.random.normal(mean_j, std_j))
        
        # Check if the sampled center is within bounds
        if (i, j) in patch_indices and (i, j) not in cluster_centers:
            cluster_centers.append((i, j))
    
    print(f"Sampled cluster centers: {cluster_centers}")
    
    label_assignments = {}
    patch_texture_map = {}
    
    # selected_terrain_labels = random.sample(range(num_textures), num_clusters)
    print(f"Selected terrain labels: {selected_terrain_labels}")
    # Map each cluster to a terrain label
    cluster_to_label = {cluster: label for cluster, label in enumerate(selected_terrain_labels)}
    
    # Assign textures and labels to patches
    for patch_file, i, j, _ in patches:
        distances = [np.sqrt((i - center[0])**2 + (j - center[1])**2) for center in cluster_centers]
        closest_cluster = np.argmin(distances)
        
        terrain_label = cluster_to_label[closest_cluster]
        label_assignments[(i, j)] = terrain_label
        patch_texture_map[patch_file] = terrain_label
    
    terrain_image = Image.open(terrain_path)
    terrain_array = np.array(terrain_image)
    terrain_labels = np.full((129, 129), -1, dtype=np.int32)
    
    for (i, j), label in label_assignments.items():
        x_start = i * (patch_size - 1)
        y_start = j * (patch_size - 1)
        x_end = min(x_start + patch_size, terrain_array.shape[0])
        y_end = min(y_start + patch_size, terrain_array.shape[1])
        terrain_labels[x_start:x_end, y_start:y_end] = label
   
    return patch_texture_map, terrain_labels

def create_texture_dictionary(base_path):
    """
    Create a dictionary of terrain types and their corresponding file paths.
    Each terrain type points to a randomly selected image file from its folder.
    """
    terrain_types = ['clay', 'concrete', 'dirt', 'grass', 'gravel', 'rock', 'wood']
    
    texture_dict = {}
    label_dict = {} 
    for idx, terrain in enumerate(terrain_types):
        folder_path = os.path.join(base_path, terrain)
        if os.path.exists(folder_path):
            files = [f for f in os.listdir(folder_path) if f.endswith(('.jpg', '.png', '.jpeg'))]
            if files:
                # Randomly select one file from the folder
                selected_texture = os.path.join(folder_path, random.choice(files))
                relative_texture_path = os.path.relpath(selected_texture, base_path)
                texture_dict[terrain] = "terrain/textures/rigid/" + relative_texture_path
                label_dict[terrain] = idx
            else:
                print(f"No valid images found in {folder_path}")
        else:
            print(f"Folder not found: {folder_path}")

    return texture_dict, label_dict

def sample_rigid_patch_properties(terrain_patches, patch_texture_map, texture_file_options):
    """
    Set material properties for a terrain patch based on its type.
    """
    property_dict = {}
    rigid_properties = RigidProperties()
    
    for patch_file, i, j, _ in terrain_patches:
        texture_index = patch_texture_map[patch_file]
        texture_types = list(texture_file_options.keys())
        texture_type = texture_types[texture_index]
        
        friction, restitution = rigid_properties.sample_properties(texture_type)
        property_dict[(i, j)] = {
            'friction': friction,
            'restitution': restitution,
            'texture_type': texture_type
        }
    
    return property_dict

def save_terrain_labels(terrain_labels, output_path):
    """
    Save terrain labels as a single numpy array matching the original terrain dimensions.
    """
    # Save the complete label array
    np.save(output_path, terrain_labels)
    print(f"Saved terrain labels to {output_path}")
    
def save_property_maps(property_dict, terrain_patches, terrain_path, patch_size, output_dir):
    """
    Save the stored properties as npy files.
    """
    # Initialize arrays
    terrain_image = Image.open(terrain_path)
    terrain_array = np.array(terrain_image)
    terrain_shape = terrain_array.shape
    
    friction_map = np.full(terrain_shape, -1.0, dtype=np.float32)
    restitution_map = np.full(terrain_shape, -1.0, dtype=np.float32)
    
    # Fill arrays using stored properties
    for patch_file, i, j, _ in terrain_patches:
        properties = property_dict[(i, j)]
        
        x_start = i * (patch_size - 1)
        y_start = j * (patch_size - 1)
        x_end = min(x_start + patch_size, terrain_array.shape[0])
        y_end = min(y_start + patch_size, terrain_array.shape[1])
        
        friction_map[x_start:x_end, y_start:y_end] = properties['friction']
        restitution_map[x_start:x_end, y_start:y_end] = properties['restitution']
    
    # Save arrays
    os.makedirs(output_dir, exist_ok=True)
    friction_path = os.path.join(output_dir, 'friction.npy')
    restitution_path = os.path.join(output_dir, 'restitution.npy')
    
    np.save(friction_path, friction_map)
    np.save(restitution_path, restitution_map)
    
    # Save text versions for visualization
    np.savetxt(friction_path.replace('.npy', '.txt'), friction_map, fmt='%.4f', delimiter=',')
    np.savetxt(restitution_path.replace('.npy', '.txt'), restitution_map, fmt='%.4f', delimiter=',')
    
    print(f"Saved friction coefficients to: {friction_path}")
    print(f"Saved restitution coefficients to: {restitution_path}")
    

def run_simulation(render=False, use_gui=False, m_isRigid=False, m_isFlat=False):
    # System and Terrain Setup
    m_system = chrono.ChSystemNSC()
    m_system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
    m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    
    patches_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                  "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps/patches")
    patch_size = 17
    terrain_patches = divide_terrain_image(terrain_path, patches_folder, patch_size)
    
    num_clusters = 1
    patch_texture_map, terrain_labels = assign_textures_to_patches(
        num_textures=len(texture_file_options),
        patches=terrain_patches,
        num_clusters=num_clusters,
        terrain_path=terrain_path,
        patch_size=patch_size
    )
    
    labels_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps", "terrain_labels.npy")
    save_terrain_labels(terrain_labels, labels_path)
    
    #Check correctness of terrain labels
    terrain_labels = np.load(labels_path)
    txt_path = labels_path.replace('.npy', '.txt')
    np.savetxt(txt_path, terrain_labels, fmt='%d', delimiter=',')
    
    # Terrain setup
    if m_isRigid:
        m_terrain = veh.RigidTerrain(m_system)
    
        # Rotation and patch creation
        rotation_quaternion = chrono.ChQuaternionD()
        rotation_quaternion.Q_from_AngAxis(0, chrono.ChVectorD(0, 0, 1))
        property_dict = sample_rigid_patch_properties(terrain_patches, patch_texture_map, texture_file_options)
        
        if m_isFlat:
            for patch_file, i, j, center_pos in terrain_patches:
                properties = property_dict[(i, j)]
                
                patch_mat = chrono.ChMaterialSurfaceNSC()
                patch_mat.SetFriction(properties['friction'])
                patch_mat.SetRestitution(properties['restitution'])
                
                patch_pos = chrono.ChVectorD(*center_pos)
                patch = m_terrain.AddPatch(patch_mat, chrono.ChCoordsysD(patch_pos, rotation_quaternion), 
                                           patch_size, patch_size) 
                
                texture_file = texture_file_options[properties['texture_type']]
                patch.SetTexture(veh.GetDataFile(texture_file), patch_size, patch_size)
                
            properties_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps")
            save_property_maps(property_dict, terrain_patches, terrain_path, patch_size, properties_path)
        
        else:    
            # Load small patches
            for patch_file, i, j, center_pos in terrain_patches:
                properties = property_dict[(i, j)]
                
                patch_mat = chrono.ChMaterialSurfaceNSC()
                patch_mat.SetFriction(properties['friction'])
                patch_mat.SetRestitution(properties['restitution'])
                
                patch_pos = chrono.ChVectorD(*center_pos)
                patch = m_terrain.AddPatch(patch_mat, chrono.ChCoordsysD(patch_pos, rotation_quaternion), 
                                           patch_file, patch_size, patch_size, 
                                           m_min_terrain_height, m_max_terrain_height) 
                
                texture_file = texture_file_options[properties['texture_type']]
                patch.SetTexture(veh.GetDataFile(texture_file), patch_size, patch_size)
            
            properties_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps")
            save_property_maps(property_dict, terrain_patches, terrain_path, patch_size, properties_path)
                
        m_terrain.Initialize()
    
    else:
        m_terrain = veh.SCMTerrain(m_system)
        # Set the SCM parameters
        terrain_params = SCMParameters()
        terrain_params.InitializeParametersAsHard()
        terrain_params.SetParameters(m_terrain)
        m_terrain.EnableBulldozing(True)
        m_terrain.SetBulldozingParameters(
                55,  # angle of friction for erosion of displaced material at the border of the rut
                1,  # displaced material vs downward pressed material.
                5,  # number of erosion refinements per timestep
                10)  # number of concentric vertex selections subject to erosion
        m_terrain.SetMeshWireframe(False)
        m_terrain.Initialize(terrain_path, m_terrain_length, m_terrain_width, 
                             m_min_terrain_height, m_max_terrain_height, terrain_delta)  
    
    # Vehicle setup
    m_vehicle = veh.HMMWV_Reduced(m_system)
    m_vehicle.SetContactMethod(chrono.ChContactMethod_NSC)
    m_vehicle.SetChassisCollisionType(veh.CollisionType_PRIMITIVES)
    m_vehicle.SetChassisFixed(False)
    m_vehicle.SetEngineType(veh.EngineModelType_SIMPLE_MAP)
    m_vehicle.SetTransmissionType(veh.TransmissionModelType_AUTOMATIC_SIMPLE_MAP)
    m_vehicle.SetDriveType(veh.DrivelineTypeWV_AWD)
    m_vehicle.SetTireType(veh.TireModelType_RIGID)
    m_vehicle.SetTireStepSize(m_step_size)
    
    m_terrain_center = [0, 0, 0]
    m_initLoc, m_initRot, m_initYaw = initialize_vw_pos(m_vehicle, m_terrain, m_terrain_length, m_terrain_width, m_terrain_center, seed=None)
    m_goal = set_goal(m_terrain, m_system, m_terrain_length, m_terrain_width, m_initLoc, m_terrain_center, buffer=4.0, z_offset=2.0)
    
    m_vehicle.Initialize()

    m_vehicle.LockAxleDifferential(0, True)    
    m_vehicle.LockAxleDifferential(1, True)
    m_vehicle.LockCentralDifferential(0, True)
    m_vehicle.LockCentralDifferential(1, True)
    m_vehicle.GetVehicle().EnableRealtime(False)
    
    m_vehicle.SetChassisVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetWheelVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
    
    m_chassis_body = m_vehicle.GetChassisBody()
    
    if m_isRigid == False:
        # Add texture to SCM terrain
        m_terrain.AddMovingPatch(m_chassis_body, chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(5, 3, 1))
        texture_file = texture_file_options[-1]
        print(terrain_file)
        m_terrain.SetTexture(veh.GetDataFile(texture_file), m_terrain_length, m_terrain_width)
        m_terrain.SetPlotType(veh.SCMTerrain.PLOT_NONE, m_min_terrain_height, m_max_terrain_height)

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

    # Initialize MPPI Controller
    # In your main simulation code
    mppi = MPPIController(
        num_samples=100,
        horizon=10,
        dt=0.01,
        lambda_=15,
        wheelbase=2.85,
        goal=m_goal,
        terrain_file=terrain_path,
        terrain_length=m_terrain_length,
        terrain_width=m_terrain_width,
        min_height=m_min_terrain_height,
        max_height=m_max_terrain_height
    )

    # Track roll and pitch values
    start_time = m_system.GetChTime()
    roll_angles = []
    pitch_angles = []
    actual_path = []

    while vis.Run() if render else True:
        current_time = m_system.GetChTime()
        if len(actual_path) == 0:  # If this is the first iteration
            start_time = current_time

        if render:
            vis.BeginScene()
            vis.Render()
            vis.EndScene()

        euler_angles = m_vehicle.GetVehicle().GetRot().Q_to_Euler123()
        current_pos = m_vehicle.GetVehicle().GetPos()
        current_speed = m_vehicle.GetVehicle().GetSpeed()
        current_yaw = m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z
        m_vector_to_goal_noNoise = current_pos - m_goal
        
        roll = euler_angles.x
        pitch = euler_angles.y

        roll_angles.append(np.degrees(abs(roll)))
        pitch_angles.append(np.degrees(abs(pitch)))

        # Update position using control input (e.g., bicycle model dynamics)
        control_input = mppi.compute_control(current_pos, current_yaw, m_vehicle, m_terrain)
        delta_steer, acceleration = control_input

        actual_path.append((current_pos.x, current_pos.y, current_pos.z))
    
        # Apply control inputs
        m_driver_inputs = m_driver.GetInputs()
        m_driver_inputs.m_steering = np.clip(delta_steer, -1, 1)
        m_driver_inputs.m_throttle = np.clip(acceleration, 0, 1)
        
        if m_vector_to_goal_noNoise.Length() < 4:
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
            dist = m_vector_to_goal_noNoise.Length()
            print('Final position of art: ', m_chassis_body.GetPos())
            print('Goal position: ', m_goal)
            print('Distance to goal: ', dist)
            print('--------------------------------------------------------------')
            if render:
                vis.Quit()
                
            avg_roll = np.mean(roll_angles) if roll_angles else 0 
            avg_pitch = np.mean(pitch_angles) if pitch_angles else 0
            return time - start_time, False, avg_roll, avg_pitch

        for _ in range(0, m_steps_per_control):
            time = m_system.GetChTime()
            # Update modules (process inputs from other modules)
            m_terrain.Synchronize(time)
            m_vehicle.Synchronize(time, m_driver_inputs, m_terrain)
            if render:
                vis.Synchronize(time, m_driver_inputs)

            # Advance simulation for one timestep for all modules
            m_driver.Advance(m_step_size)
            m_terrain.Advance(m_step_size)
            m_vehicle.Advance(m_step_size)
            if render:
                vis.Advance(m_step_size)
            
            m_system.DoStepDynamics(m_step_size)

    return None, False, 0, 0

if __name__ == '__main__':
    # Terrain parameters
    SetChronoDataDirectories()
    CHRONO_DATA_DIR = chrono.GetChronoDataPath()
    base_texture_path = os.path.join(CHRONO_DATA_DIR, "vehicle/terrain/textures/rigid")
    texture_file_options, texture_labels = create_texture_dictionary(base_texture_path)
    
    print("Labels of texture:")
    print(texture_labels)
    
    terrain_file_options = ["1.bmp"]
    terrain_file = terrain_file_options[0]
    terrain_path = os.path.join(os.path.dirname(os.path.realpath(__file__)),
                                "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps", terrain_file)

    # Parameters                                                                                                                                                                                                                 
    m_max_time = 40
    m_min_terrain_height = -7  # min terrain height
    m_max_terrain_height = 7 # max terrain height
    m_terrain_length = 80  # size in X direction
    m_terrain_width = 80  # size in Y direction
    terrain_delta = 0.05 # mesh resolution for SCM terrain

    # Simulation step sizes
    m_step_size = 5e-3 # simulation update every num milliseconds
    m_control_freq = 10 # control inputs frequency
    m_steps_per_control = round(1 / (m_step_size * m_control_freq))
    
    # Define the terrain labels to iterate over
    terrain_labels_to_test = [1, 2, 3, 4, 5]
    overall_results = {}  # To store results for all terrain labels

    for terrain_label in terrain_labels_to_test:
        # Modify the selected terrain label dynamically
        selected_terrain_labels = [terrain_label]

        # Run multiple experiments for the current terrain label
        num_experiments = 5
        results = []  # Store results for each experiment

        for i in range(num_experiments):
            print(f"Running experiment {i + 1}/{num_experiments} for terrain label {terrain_label}")
            time_to_goal, success, avg_roll, avg_pitch = run_simulation(render=False, use_gui=False, m_isRigid=True, m_isFlat=False)
            results.append({
                'time_to_goal': time_to_goal if success else None,
                'success': success,
                'avg_roll': avg_roll,
                'avg_pitch': avg_pitch
            })

        # Process results for the current terrain label
        success_count = sum(1 for r in results if r['success'])
        successful_times = [r['time_to_goal'] for r in results if r['time_to_goal'] is not None]
        avg_rolls = [r['avg_roll'] for r in results if r['success']]
        avg_pitchs = [r['avg_pitch'] for r in results if r['success']]

        mean_traversal_time = np.mean(successful_times) if successful_times else None
        roll_mean = np.mean(avg_rolls) if avg_rolls else None
        roll_variance = np.var(avg_rolls) if avg_rolls else None
        pitch_mean = np.mean(avg_pitchs) if avg_pitchs else None
        pitch_variance = np.var(avg_pitchs) if avg_pitchs else None

        # Save results for the current terrain label
        overall_results[terrain_label] = {
            'success_count': success_count,
            'mean_traversal_time': mean_traversal_time,
            'roll_mean': roll_mean,
            'roll_variance': roll_variance,
            'pitch_mean': pitch_mean,
            'pitch_variance': pitch_variance
        }

        # Print results for the current terrain label
        print("--------------------------------------------------------------")
        print(f"Results for terrain label {terrain_label}:")
        print(f"Success rate: {success_count}/{num_experiments}")
        if success_count > 0:
            print(f"Mean traversal time (successful trials): {mean_traversal_time:.2f} seconds")
            print(f"Average roll angle: {roll_mean:.2f} degrees, Variance: {roll_variance:.2f}")
            print(f"Average pitch angle: {pitch_mean:.2f} degrees, Variance: {pitch_variance:.2f}")
        else:
            print("No successful trials")
        print("--------------------------------------------------------------")

    # Print summary of all results
    print("\n==================== Overall Results ====================")
    for terrain_label, result in overall_results.items():
        print(f"Terrain Label {terrain_label}:")
        print(f"  Success rate: {result['success_count']}/{num_experiments}")
        if result['success_count'] > 0:
            print(f"  Mean traversal time: {result['mean_traversal_time']:.2f} seconds")
            print(f"  Average roll: {result['roll_mean']:.2f} degrees (Variance: {result['roll_variance']:.2f})")
            print(f"  Average pitch: {result['pitch_mean']:.2f} degrees (Variance: {result['pitch_variance']:.2f})")
        else:
            print("  No successful trials")
        print("--------------------------------------------------------")