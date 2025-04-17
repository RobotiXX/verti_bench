import pychrono as chrono
import pychrono.vehicle as veh
import pychrono.irrlicht as chronoirr
import numpy as np
import os
import random
import json
import math

from verti_bench.envs.utils.utils import SetChronoDataDirectories
import pychrono.sensor as sens

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
import heapq

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

class PurePursuitController:
    def __init__(self, lookahead_distance, min_lookahead=5.0, max_lookahead=15.0):
        self.base_lookahead = lookahead_distance
        self.min_lookahead = min_lookahead
        self.max_lookahead = max_lookahead
        self.last_closest_point_index = 0

    def compute_steering(self, current_pos, current_yaw, path, current_speed):
        """
        Compute steering angle using pure pursuit algorithm
        
        Args:
            current_pos: ChVectorD with vehicle position
            current_yaw: Current vehicle heading angle
            path: List of (x,y) waypoints to follow
            current_speed: Current vehicle speed in m/s
        """
        # Update lookahead distance based on speed
        self.lookahead_distance = max(self.min_lookahead, 
                                    min(current_speed * 1.5, self.max_lookahead))
        
        # Find first point on path that's at least lookahead distance away
        lookahead_point = None
        for i in range(self.last_closest_point_index, len(path)):
            distance = np.sqrt((path[i][0] - current_pos.x) ** 2 + 
                             (path[i][1] - current_pos.y) ** 2)
            
            if distance >= self.lookahead_distance:
                lookahead_point = path[i]
                self.last_closest_point_index = max(0, i-1)
                break

        # If no point found, use the last point
        if lookahead_point is None:
            lookahead_point = path[-1]

        # Transform goal point to vehicle's local frame
        dx = lookahead_point[0] - current_pos.x 
        dy = lookahead_point[1] - current_pos.y

        # Transform to vehicle's local frame
        lookahead_local_x = dx * math.cos(-current_yaw) - dy * math.sin(-current_yaw)
        lookahead_local_y = dx * math.sin(-current_yaw) + dy * math.cos(-current_yaw)

        # Prevent division by zero
        if abs(lookahead_local_x) < 0.001:
            lookahead_local_x = 0.001

        # Compute curvature (k = 2y/L^2)
        curvature = 2.0 * lookahead_local_y / (self.lookahead_distance ** 2)
        
        # Convert curvature to steering angle
        wheelbase = 2.5  # approximate wheelbase length
        steering_angle = math.atan(wheelbase * curvature)
        
        # Return steering angle and lookahead point for visualization
        return steering_angle, lookahead_point

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
    
    def resample_path(path, resolution=1.0):
        resampled_path = [path[0]]
        for i in range(1, len(path)):
            x0, y0 = path[i - 1]
            x1, y1 = path[i]
            distance = np.sqrt((x1 - x0) ** 2 + (y1 - y0) ** 2)
            num_points = max(2, int(distance / resolution))
            for j in range(1, num_points):
                t = j / num_points
                resampled_path.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
            resampled_path.append((x1, y1))
        return resampled_path

def compute_throttle(current_speed, target_speed=4.0, distance_to_goal=float('inf')):
    """Compute throttle input based on speed error"""
    # Reduce target speed when approaching goal
    if distance_to_goal < 10.0:
        target_speed *= (distance_to_goal / 10.0)
        
    # Simple proportional control
    speed_error = target_speed - current_speed
    kp = 0.5
    
    throttle = np.clip(kp * speed_error, 0.0, 0.8)
    return throttle

def compute_brake(current_speed, target_speed=8.0):
    """Compute brake input based on speed error"""
    speed_error = current_speed - target_speed
    if speed_error > 0:
        kp = 0.5
        return np.clip(kp * speed_error, 0.0, 1.0)
    return 0.0

def grid_to_world(grid_point, grid_resolution, terrain_length, terrain_width):
    """Convert grid map coordinates to world coordinates."""
    world_x = grid_point[0] * grid_resolution - terrain_length / 2
    world_y = grid_point[1] * grid_resolution - terrain_width / 2
    return (world_x, world_y)

def world_to_grid(world_point, grid_resolution, terrain_length, terrain_width):
    """Convert world coordinates to grid map coordinates."""
    grid_x = int((world_point[0] + terrain_length / 2) / grid_resolution)
    grid_y = int((world_point[1] + terrain_width / 2) / grid_resolution)
    return (grid_x, grid_y)

class MPPIController:
    def __init__(self, num_samples, horizon, dt, lambda_, wheelbase, goal):
        self.num_samples = num_samples
        self.horizon = horizon
        self.dt = dt
        self.lambda_ = lambda_
        self.wheelbase = wheelbase
        self.goal = goal
        self.max_steer_angle = 0.5  # ~30 degrees max steering angle
        self.max_speed = 4.0  # Maximum speed in m/s
        self.min_speed = 0.0   # Minimum speed in m/s
    
    def compute_control(self, current_pos, current_yaw, m_vehicle, m_terrain):
        controls = []
        costs = []
        
        # Get current speed
        current_speed = m_vehicle.GetVehicle().GetSpeed()
        
        # Calculate angle to goal
        goal_angle = np.arctan2(self.goal.y - current_pos.y, self.goal.x - current_pos.x)
        angle_diff = self.normalize_angle(goal_angle - current_yaw)
        
        for _ in range(self.num_samples):
            trajectory, roll_values, pitch_values, control_inputs = self.rollout_trajectory(
                current_pos, current_yaw, current_speed, angle_diff, m_vehicle, m_terrain
            )
            cost = self.calculate_cost(trajectory, roll_values, pitch_values, control_inputs)
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

    def rollout_trajectory(self, current_pos, current_yaw, current_speed, angle_diff, m_vehicle, m_terrain):
        trajectory = []
        roll_values = []
        pitch_values = []
        control_inputs = []

        pos = np.array([current_pos.x, current_pos.y])
        yaw = current_yaw
        speed = current_speed

        base_steer = np.clip(angle_diff, -self.max_steer_angle, self.max_steer_angle)
        trajectory.append((pos[0], pos[1]))

        for step in range(self.horizon):
            # Sample control inputs with bias towards goal direction
            delta_steer = np.clip(
                base_steer + np.random.normal(0, 0.3),
                -self.max_steer_angle,
                self.max_steer_angle
            )
            target_speed = self.max_speed * (1 - abs(delta_steer) / self.max_steer_angle)
            target_speed = max(target_speed, self.min_speed)

            # Add additional noise for diversity
            delta_steer += np.random.uniform(-0.1, 0.1)
            target_speed += np.random.uniform(-0.5, 0.5)

            # Calculate acceleration
            acceleration = np.clip((target_speed - speed) / self.dt, -1, 1)

            # Update state
            speed = speed + acceleration * self.dt
            pos[0] += speed * np.cos(yaw) * self.dt
            pos[1] += speed * np.sin(yaw) * self.dt
            yaw += (speed / self.wheelbase) * np.tan(delta_steer) * self.dt

            # Append updated states to trajectory
            trajectory.append((pos[0], pos[1]))
            control_inputs.append((delta_steer, acceleration))

        return trajectory, roll_values, pitch_values, control_inputs

    def calculate_cost(self, trajectory, roll_values, pitch_values, control_inputs):
        cost = 0
        goal_np = np.array([self.goal.x, self.goal.y])

        prev_steering = 0
        for i, (traj_point, control) in enumerate(zip(trajectory, control_inputs)):
            traj_np = np.array(traj_point)
            steering, acceleration = control

            # Distance to goal cost
            distance_to_goal = np.linalg.norm(traj_np - goal_np)
            cost += 100 * distance_to_goal ** 2

            # Penalize steering changes
            steering_change = abs(steering - prev_steering)
            cost += 50 * steering_change
            prev_steering = steering

            # Stability costs
            roll = roll_values[i] if i < len(roll_values) else 0
            pitch = pitch_values[i] if i < len(pitch_values) else 0
            cost += 100 * (abs(roll) ** 2)  # Penalize roll
            cost += 100 * (abs(pitch) ** 2)  # Penalize pitch

        return cost

    @staticmethod
    def normalize_angle(angle):
        """Normalize angle to [-pi, pi]"""
        while angle > np.pi:
            angle -= 2 * np.pi
        while angle < -np.pi:
            angle += 2 * np.pi
        return angle

class MPPIElevationController:
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
    
    m_initLoc=[-35, 40, -2]
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

def get_cropped_map(bitmap_file, m_vehicle, m_vehicle_pos, submap_shape_x, submap_shape_y):
    vehicle_x = -m_vehicle_pos.x
    vehicle_y = m_vehicle_pos.y

    image = Image.open(bitmap_file)
    image_edge = cv2.imread(bitmap_file, cv2.IMREAD_GRAYSCALE)
    bitmap_array = np.array(image)

    min_y_index = np.min(np.where(image_edge > 0)[1])  
    max_y_index = np.max(np.where(image_edge > 0)[1])  
    min_x_index = np.min(np.where(image_edge > 0)[0])  
    max_x_index = np.max(np.where(image_edge > 0)[0])

    bmp_dim_x = max_x_index - min_x_index #width
    bmp_dim_y = max_y_index - min_y_index #length
    
    terrain_length_tolerance = m_terrain_length
    terrain_width_tolerance = m_terrain_width

    map_offset_x = terrain_width_tolerance / 2
    map_offset_y = terrain_length_tolerance / 2

    # Normalization scaling factors
    s_norm_x = 1 / terrain_width_tolerance
    s_norm_y = 1 / terrain_length_tolerance
    s_x = s_norm_x * bmp_dim_x
    s_y = s_norm_y * bmp_dim_y

    # Transformation matrix 
    T = np.array([
        [s_y, 0, 0],
        [0, s_x, 0],
        [0, 0, 1]
    ])

    pos_chrono = np.array([vehicle_x + map_offset_y, 
                        vehicle_y, 1])

    pos_bmp = np.dot(T, pos_chrono)
    pos_bmp_x, pos_bmp_y = bitmap_array.shape[1] // 2 + int(pos_bmp[1]), int(pos_bmp[0])

    # Check if pos_bmp_x and pos_bmp_y are within bounds
    assert 0 <= pos_bmp_x < bitmap_array.shape[0], f"pos_bmp_x out of bounds: {pos_bmp_x}"
    assert 0 <= pos_bmp_y < bitmap_array.shape[1], f"pos_bmp_y out of bounds: {pos_bmp_y}"

    # x-axis: back, y-axis: right, z-axis: up
    vehicle_heading_global = m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z

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

    # cv2.imshow('Rotated Bitmap', rotated_map)
    # cv2.waitKey(0)

    half_size_x = submap_shape_x // 2
    half_size_y = submap_shape_y // 2

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

    if end_y - start_y < submap_shape_y:
        if start_y == 0:
            # If we're at the lower boundary, adjust end_y up
            end_y = min(start_y + submap_shape_y, rotated_map.shape[0])
        elif end_y == rotated_map.shape[0]:
            # If we're at the upper boundary, adjust start_y down
            start_y = max(end_y - submap_shape_y, 0)

    if end_x - start_x < submap_shape_x:
        if start_x == 0:
            # If we're at the left boundary, adjust end_x right
            end_x = min(start_x + submap_shape_x, rotated_map.shape[1])
        elif end_x == rotated_map.shape[1]:
            # If we're at the right boundary, adjust start_x left
            start_x = max(end_x - submap_shape_x, 0)

    # Extract the sub-array
    sub_array = rotated_map[start_x:end_x, start_y:end_y]
    return sub_array

def crop_front_part(map):
    # Crop the upper half of the map
    center_y = map.shape[0] // 2
    front_part = map[:center_y, :]
    return front_part

def find_lowest_rock_position(front_part, vehicle_x, vehicle_y):
    current_mean = np.mean(front_part)
    current_var = np.var(front_part)
    
    path_width = vehicle_y // 5
    forward_paths = []
    path_stats = []
    
    for i in range(5):
        start_col = i * path_width
        end_col = (i + 1) * path_width
        forward_path = front_part[:vehicle_y, start_col:end_col]
        
        path_mean = np.mean(forward_path)
        path_var = np.var(forward_path)
        
        mean_diff = abs(path_mean - current_mean)
        var_diff = abs(path_var - current_var)
        similarity_score = mean_diff + 0.5 * var_diff 
        
        forward_paths.append(forward_path)
        path_stats.append((i, similarity_score, path_mean, path_var))
        
    best_path_idx = min(path_stats, key=lambda x: (x[1], x[3]))[0]
    # Calculate relative position for steering
    target_x = (best_path_idx + 0.5) * path_width 
    relative_x = target_x - vehicle_x
    relative_y = -vehicle_y
    
    return relative_x, relative_y

def calculate_target_heading(relative_x, relative_y):
    target_heading_local = np.arctan2(relative_y, relative_x)
    return target_heading_local

def compute_steering_command(m_steeringController, target_heading_local, current_heading_local, dt):
    heading_diff_local = (target_heading_local - current_heading_local + np.pi) % (2 * np.pi) - np.pi
    steering_command = -m_steeringController.compute(heading_diff_local, dt)
    return steering_command

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


def pid(speed_controller, steering_controller, heading_diff_local, m_step_size, m_vehicle, speed, time, m_driver_inputs):
    # Compute steering using PID
    steering = -steering_controller.compute(heading_diff_local, m_step_size)
    # Apply steering limits for smoothness
    m_driver_inputs.m_steering = np.clip(steering, m_driver_inputs.m_steering - 0.05, 
                                         m_driver_inputs.m_steering + 0.05)
    # Desired throttle/braking value
    out_throttle = speed_controller.Advance(m_vehicle.GetRefFrame(), speed, time, m_step_size)
    out_throttle = np.clip(out_throttle, -1, 1)
    if out_throttle > 0:
        m_driver_inputs.m_braking = 0
        m_driver_inputs.m_throttle = out_throttle
    else:
        m_driver_inputs.m_braking = -out_throttle
        m_driver_inputs.m_throttle = 0
    return m_driver_inputs

def pid_elevation(speed_controller, steering_controller, heading_diff_local, current_heading_local, m_vehicle, 
                  speed, time, terrain_path, m_step_size, m_driver_inputs):
    m_vehicle_pos = m_vehicle.GetVehicle().GetPos()
    sub_array = get_cropped_map(terrain_path, m_vehicle, m_vehicle_pos, submap_shape_x=64, submap_shape_y=64)
    front_part = crop_front_part(sub_array)
    
    vehicle_center_x = front_part.shape[1] // 2
    vehicle_center_y = front_part.shape[0]
    relative_x, relative_y = find_lowest_rock_position(front_part, vehicle_center_x, vehicle_center_y)
    target_heading_local = calculate_target_heading(relative_x, relative_y)
    
    target_heading_weight = 0.25
    goal_heading_weight = 0.75

    combined_heading = target_heading_weight * target_heading_local + goal_heading_weight * heading_diff_local
    steering_command = compute_steering_command(steering_controller, combined_heading, current_heading_local, m_step_size)
    m_driver_inputs.m_steering = np.clip(steering_command, -1, 1)
    
    # Desired throttle/braking value
    out_throttle = speed_controller.Advance(m_vehicle.GetRefFrame(), speed, time, m_step_size)
    out_throttle = np.clip(out_throttle, -1, 1)
    if out_throttle > 0:
        m_driver_inputs.m_braking = 0
        m_driver_inputs.m_throttle = out_throttle
    else:
        m_driver_inputs.m_braking = -out_throttle
        m_driver_inputs.m_throttle = 0
    return m_driver_inputs

def astar_pure(grid_map, m_initLoc, m_goal, m_terrain_length, m_terrain_width, m_vehicle, m_driver_inputs):
    grid_size = grid_map.shape[0]
    grid_resolution = m_terrain_length / grid_size
    start = world_to_grid((m_initLoc.x, m_initLoc.y), grid_resolution, m_terrain_length, m_terrain_width)
    goal = world_to_grid((m_goal.x, m_goal.y), grid_resolution, m_terrain_length, m_terrain_width)

    # Plan path with A*
    a_star = AStarPlanner(grid_map, start, goal)
    a_star_path = a_star.plan()
    if a_star_path is None:
        print("No path found by A* planner")
        return m_driver_inputs

    # Convert path to world coordinates
    world_a_star_path = [grid_to_world(point, grid_resolution, m_terrain_length, m_terrain_width) for point in a_star_path]

    # Initialize Pure Pursuit Controller
    pure_pursuit_controller = PurePursuitController(lookahead_distance=10.0)
    
    # Get current vehicle state
    current_pos = m_vehicle.GetVehicle().GetPos()
    current_yaw = m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z
    current_speed = m_vehicle.GetVehicle().GetSpeed()
    distance_to_goal = np.sqrt((current_pos.x - m_goal.x)**2 + (current_pos.y - m_goal.y)**2)

    # Compute steering
    steering_angle, lookahead_point = pure_pursuit_controller.compute_steering(
        current_pos, current_yaw, world_a_star_path, current_speed)

    # Compute throttle and brake
    throttle = compute_throttle(current_speed, target_speed=4.0, distance_to_goal=distance_to_goal)
    brake = compute_brake(current_speed, target_speed=4.0)

    # Apply control inputs
    m_driver_inputs.m_steering = np.clip(steering_angle, -1, 1)
    m_driver_inputs.m_throttle = throttle
    m_driver_inputs.m_braking = brake
    
    return m_driver_inputs

def mppi(mppi_controller, m_vehicle, m_terrain, m_driver_inputs):
    current_pos = m_vehicle.GetVehicle().GetPos()
    current_yaw = m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z
    current_speed = m_vehicle.GetVehicle().GetSpeed()

    # Compute control input using MPPI
    control_input = mppi_controller.compute_control(current_pos, current_yaw, m_vehicle, m_terrain)
    delta_steer, acceleration = control_input

    # Apply control inputs
    m_driver_inputs.m_steering = np.clip(delta_steer, -1, 1)
    m_driver_inputs.m_throttle = np.clip(acceleration, 0, 1)

    return m_driver_inputs

def mppi_elevation(mppi_elevation_controller, m_vehicle, m_terrain, m_driver_inputs):
    current_pos = m_vehicle.GetVehicle().GetPos()
    current_yaw = m_vehicle.GetVehicle().GetRot().Q_to_Euler123().z
    current_speed = m_vehicle.GetVehicle().GetSpeed()

    # Compute control input using MPPI + Elevation Map
    control_input = mppi_elevation_controller.compute_control(current_pos, current_yaw, m_vehicle, m_terrain)
    delta_steer, acceleration = control_input

    # Apply control inputs
    m_driver_inputs.m_steering = np.clip(delta_steer, -1, 1)
    m_driver_inputs.m_throttle = np.clip(acceleration, 0, 1)

    return m_driver_inputs

def get_current_label(terrain_labels, m_vehicle_pos, terrain_length, terrain_width):
    label_rows, label_cols = terrain_labels.shape
    
    vehicle_x = m_vehicle_pos.x + 2.5 // 2
    vehicle_y = -m_vehicle_pos.y 
    grid_x = int((vehicle_x + terrain_length / 2) * (label_cols / terrain_length))
    grid_y = int((vehicle_y + terrain_width / 2) * (label_rows / terrain_width))

    # Ensure indices are within bounds
    grid_x = np.clip(grid_x, 0, label_cols - 1) 
    grid_y = np.clip(grid_y, 0, label_rows - 1)

    # Retrieve the terrain label
    current_label = terrain_labels[grid_y, grid_x]
    
    return current_label


def run_simulation(render=False, use_gui=False, m_isRigid=False, m_isFlat = False):
    # System and Terrain Setup
    m_system = chrono.ChSystemNSC()
    m_system.Set_G_acc(chrono.ChVectorD(0, 0, -9.81))
    m_system.SetCollisionSystemType(chrono.ChCollisionSystem.Type_BULLET)
    
    patches_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                  "../envs/data/terrain_bitmaps/BenchMaps/sampled_maps/patches")
    patch_size = 17
    terrain_patches = divide_terrain_image(terrain_path, patches_folder, patch_size)
    
    num_clusters = 5
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
    
    m_vehicle.SetChassisVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetWheelVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetTireVisualizationType(veh.VisualizationType_MESH)
    m_vehicle.SetSuspensionVisualizationType(veh.VisualizationType_PRIMITIVES)
    m_vehicle.SetSteeringVisualizationType(veh.VisualizationType_PRIMITIVES)
    
    m_chassis_body = m_vehicle.GetChassisBody()
    
    if m_isRigid == False:
        # Add texture to SCM terrain
        m_terrain.AddMovingPatch(m_chassis_body, chrono.ChVectorD(0, 0, 0), chrono.ChVectorD(5, 3, 1))
        texture_file = texture_file_options[-1]
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
        
    m_driver_inputs = m_driver.GetInputs()
    
    # Continuous speed
    speed = 4.0 if not use_gui else 0.0 
    start_time = m_system.GetChTime()
    
    roll_angles = []
    pitch_angles = []
    actual_path = []
    
    while vis.Run() if render else True:
        time = m_system.GetChTime()
        
        if render:
            vis.BeginScene()
            vis.Render()
            vis.EndScene()
            
        m_vehicle_pos = m_vehicle.GetVehicle().GetPos()
        m_vector_to_goal_noNoise = m_goal - m_vehicle_pos
        current_label = get_current_label(terrain_labels, m_vehicle_pos, m_terrain_length, m_terrain_width)
        planners = ['PID', 'PID_Elevation', 'Astar_Pure', 'MPPI', 'MPPI_Elevation']
        if current_label == 1:
            selected_planner = planners[3]
        elif current_label == 2:
            selected_planner = planners[3]
        elif current_label == 3:
            selected_planner = planners[4]
        elif current_label == 4:
            selected_planner = planners[3]
        elif current_label == 5:
            selected_planner = planners[4]
        
        # selected_planner = planners[4]
            
        print(f"Current label: {current_label}, Selected planner: {selected_planner}")
        
        if use_gui:
            m_driver_inputs = m_driver.GetInputs()
        else:
            m_vector_to_goal = m_goal - m_vehicle_pos
            vector_to_goal_local = m_chassis_body.GetRot().RotateBack(m_vector_to_goal)
            goal_heading_local = np.arctan2(vector_to_goal_local.y, vector_to_goal_local.x)
            heading_diff_local = (goal_heading_local - 0 + np.pi) % (2 * np.pi) - np.pi
                    
            euler_angles = m_vehicle.GetVehicle().GetRot().Q_to_Euler123()
            roll = euler_angles.x
            pitch = euler_angles.y
            current_heading_local = euler_angles.z
            roll_angles.append(np.degrees(abs(roll)))
            pitch_angles.append(np.degrees(abs(pitch)))
            
            if selected_planner == 'PID':
                # Set PID controller for speed
                m_speedController = veh.ChSpeedController()
                m_speedController.Reset(m_vehicle.GetRefFrame())
                m_speedController.SetGains(1, 0, 0)
                # Initialize the custom PID controller for steering
                m_steeringController = PIDController(kp=3.0, ki=0.1, kd=0.5)
                m_driver_inputs = pid(m_speedController, m_steeringController, goal_heading_local, 
                                      m_step_size, m_vehicle, speed, time, m_driver_inputs)
                
            elif selected_planner == 'PID_Elevation':
                # Set PID controller for speed
                m_speedController = veh.ChSpeedController()
                m_speedController.Reset(m_vehicle.GetRefFrame())
                m_speedController.SetGains(1, 0, 0)
                # Initialize the custom PID controller for steering
                m_steeringController = PIDController(kp=3.0, ki=0.1, kd=0.5)
                m_driver_inputs = pid_elevation(m_speedController, m_steeringController, heading_diff_local, current_heading_local, 
                                                m_vehicle, speed, time, terrain_path, m_step_size, m_driver_inputs)
            
            elif selected_planner == 'Astar_Pure':
                grid_size = 80  # Grid dimensions (80x80 cells)
                grid_map = np.zeros((grid_size, grid_size))  # Create the grid map
                m_driver_inputs = astar_pure(grid_map, m_initLoc, m_goal, m_terrain_length, 
                                            m_terrain_width, m_vehicle, m_driver_inputs)
            
            elif selected_planner == 'MPPI':
                # Initialize MPPI Controller
                mppi_controller = MPPIController(num_samples=500, horizon=25, dt=0.01, lambda_=10, wheelbase=2.85, goal=m_goal)
                m_driver_inputs = mppi(mppi_controller, m_vehicle, m_terrain, m_driver_inputs)
            
            elif selected_planner == 'MPPI_Elevation':
                mppi_elevation_controller = MPPIElevationController(
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
                m_driver_inputs = mppi_elevation(mppi_elevation_controller, m_vehicle, m_terrain, m_driver_inputs)


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
    
    return None, False, 0, 0  # Return None if goal not reached

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
    m_min_terrain_height = -8  # min terrain height
    m_max_terrain_height = 8 # max terrain height
    m_terrain_length = 80  # size in X direction
    m_terrain_width = 80  # size in Y direction
    terrain_delta = 0.05 # mesh resolution for SCM terrain

    # Simulation step sizes
    m_step_size = 5e-3 # simulation update every num milliseconds
    m_control_freq = 10 # control inputs frequency
    m_steps_per_control = round(1 / (m_step_size * m_control_freq))
    
    # Define the terrain labels to choose
    selected_terrain_labels = [1, 2, 3, 4, 5]

    # Run multiple experiments 
    num_experiments = 1
    results = [] 

    for i in range(num_experiments):
        print(f"Running experiment {i + 1}/{num_experiments}")
        time_to_goal, success, avg_roll, avg_pitch = run_simulation(render=True, use_gui=False, m_isRigid=True, m_isFlat=False)
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
    