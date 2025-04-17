import pychrono as chrono
try:
    import pychrono.sensor as sens
except:
    print('Could not import Chrono Sensor')

import random
from PIL import Image, ImageDraw
import numpy as np

class Asset():
    """"Class that initializes an asset"""

    def __init__(self, visual_shape_path, scale=None, collision_shape_path=None, bounding_box=None):
        if (scale == None):
            self.scale = 1
            self.scaled = False
        else:
            self.scale = scale
            self.scaled = False

        self.visual_shape_path = visual_shape_path
        self.collision_shape_path = collision_shape_path

        # Intialize a random material
        self.material = chrono.ChContactMaterialNSC()
        # initialize the body
        self.body = chrono.ChBodyAuxRef()
        # set body as fixed
        self.body.SetFixed(True)

        # Get the visual mesh
        visual_shape_obj = chrono.GetChronoDataFile(visual_shape_path)
        visual_mesh = chrono.ChTriangleMeshConnected().CreateFromWavefrontFile(visual_shape_obj, False, False)
        visual_mesh.Transform(chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(scale))
        # Add this mesh to the visual shape
        self.visual_shape = chrono.ChVisualShapeTriangleMesh()
        self.visual_shape.SetMesh(visual_mesh)
        # Add visual shape to the mesh body
        self.body.AddVisualShape(self.visual_shape)

        # Get the collision mesh
        collision_shape_obj = None
        if (collision_shape_path == None):
            # Just use the bounding box
            if (bounding_box == None):
                self.body.EnableCollision(False)
                self.collide_flag = False
            else:
                size = bounding_box * scale
                material = chrono.ChContactMaterialNSC()
                collision_shape = chrono.ChCollisionShapeBox(material, size.x, size.y, size.z)
                self.body.AddCollisionShape(collision_shape)
                self.body.EnableCollision(True)
                self.collide_flag = True
        else:
            collision_shape_obj = chrono.GetChronoDataFile(
                collision_shape_path)
            collision_mesh = chrono.ChTriangleMeshConnected(
            ).CreateFromWavefrontFile(collision_shape_obj, False, False)
            collision_mesh.Transform(chrono.ChVector3d(0, 0, 0),
                                     chrono.ChMatrix33d(scale))
            collision_shape = chrono.ChCollisionShapeTriangleMesh(self.material, collision_mesh,
                                                                  True, True, chrono.ChVector3d(0, 0, 0), chrono.ChMatrix33d(1))
            self.body.AddCollisionShape(collision_shape)
            # Update the collision model
            self.body.EnableCollision(True)
            self.collide_flag = True

        self.collision_shape = collision_shape_obj
        self.bounding_box = bounding_box

        # Asset's position and orientation will be set by the simulation assets class
        self.pos = chrono.ChVector3d()
        self.ang = 0

    def UpdateAssetPosition(self, pos, ang):
        self.pos = pos
        self.ang = ang
        self.body.SetFrameRefToAbs(chrono.ChFramed(pos, ang))

    # Create a copy constructor for the asset
    def Copy(self):
        """Returns a copy of the asset"""
        asset = Asset(self.visual_shape_path, self.scale, self.collision_shape_path, self.bounding_box)
        return asset

class SimulationAssets():
    """Class that handles assets for the Gym Environment"""

    def __init__(self, system, length, width, terrain_path, min_height, max_height, m_isFlat):
        self.system = system
        self.length = length
        self.width = width
        self.terrain_path = terrain_path
        self.min_height = min_height
        self.max_height = max_height
        self.isFlat = m_isFlat
        self.assets_list = []
        self.positions = []
        
        # Configuration for obstacles placement
        self.SCALE_FACTOR = 1
        self.VEHICLE_SAFETY_DIST = 12 * self.SCALE_FACTOR
        self.GOAL_SAFETY_DIST = 5 * self.SCALE_FACTOR
        self.ROCK_SAFETY_DIST = 10 * self.SCALE_FACTOR

    def AddAsset(self, asset, number=1):
        """Number of such asset to be added"""
        for _ in range(number):
            new_asset = asset.Copy()
            self.assets_list.append(new_asset)
            
    def get_interpolated_height(self, terrain_array, x_float, y_float, bmp_dim_x, bmp_dim_y):
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

    # Position assets relative to goal and chassis
    def RandomlyPositionAssets(self, goal_pos, chassis_body, avoid_positions):
        """Randomly positions assets within the terrain"""
        terrain_image = Image.open(self.terrain_path)
        terrain_array = np.array(terrain_image)
        bmp_dim_y, bmp_dim_x = terrain_array.shape
        
        # Calculate transformation factors
        s_norm_x = bmp_dim_x / self.width
        s_norm_y = bmp_dim_y / self.length
        T = np.array([
            [s_norm_x, 0, 0],
            [0, s_norm_y, 0],
            [0, 0, 1]
        ])
        
        obstacles_info = {
            'rocks': [],
            'trees': []
        }
        
        # Logic to place assets along the path
        start_pos = chassis_body.GetPos()
        path_vector = goal_pos - start_pos
        path_length = path_vector.Length()
        path_direction = path_vector / path_length
        avoid_positions_ch = [chrono.ChVector3d(pos[0], pos[1], pos[2]) for pos in avoid_positions]
         
        offset_width = self.width
        offset_length = self.length
        
        for asset in self.assets_list:
            placed = False
            attempt_count = 0
            max_attempts = 500
            
            while not placed and attempt_count < max_attempts:
                t = random.uniform(0, path_length) # Random position along the path
                offset_x = random.uniform(-offset_width, offset_width) 
                offset_y = random.uniform(-offset_length, offset_length)
                pos = start_pos + path_direction * t + chrono.ChVector3d(offset_x, offset_y, 0)
                
                # Transform to bitmap coordinates
                pos_chrono = np.array([pos.x + self.width / 2, -pos.y + self.length / 2, 1])
                pos_bmp = np.dot(T, pos_chrono)
                x_bmp, y_bmp = pos_bmp[0], pos_bmp[1]
                
                # Calculate height from bitmap
                if not (0 <= x_bmp < bmp_dim_x - 1 and 0 <= y_bmp < bmp_dim_y - 1):
                    attempt_count += 1
                    continue
                
                if self.isFlat:
                    pos.z = 0.5
                else:
                    height_ratio = self.get_interpolated_height(terrain_array, x_bmp, y_bmp, bmp_dim_x, bmp_dim_y)
                    pos.z = self.min_height + height_ratio * (self.max_height - self.min_height) / self.SCALE_FACTOR
                
                # Check distances
                vehicle_dist = (pos - chassis_body.GetPos()).Length()
                goal_dist = (pos - goal_pos).Length()
                
                # Check if the position is too close to any (start, goal) pair
                close_avoid = any((pos - avoid_pos).Length() < self.VEHICLE_SAFETY_DIST for avoid_pos in avoid_positions_ch)
                
                # Initialize valid position flag
                valid_position = (vehicle_dist > self.VEHICLE_SAFETY_DIST and
                                  goal_dist > self.GOAL_SAFETY_DIST and
                                  not close_avoid)
                    
                # Check distance from other assets
                if valid_position and self.positions:
                    min_pos = min(self.positions, key=lambda x: (x - pos).Length())
                    min_dist = (pos - min_pos).Length()
                    threshold = asset.bounding_box.Length() * asset.scale
                    valid_position = min_dist > max(self.ROCK_SAFETY_DIST, threshold)

                if valid_position:
                    self.positions.append(pos)
                    asset.UpdateAssetPosition(pos, chrono.ChQuaternionD(1, 0, 0, 0))
                    self.system.GetCollisionSystem().BindItem(asset.body)
                    self.system.Add(asset.body)
                    
                    if "rock" in asset.visual_shape_path.lower():
                        obstacles_info['rocks'].append({
                            'position': (pos.x, pos.y, pos.z),
                            'scale': asset.scale
                        })
                    elif "tree" in asset.visual_shape_path.lower():
                        obstacles_info['trees'].append({
                            'position': (pos.x, pos.y, pos.z)
                        })
                        
                    placed = True

                attempt_count += 1
            
            if not placed:
                print(f"Warning: Could not place asset after {max_attempts} attempts")
        
        return obstacles_info

    def CheckContact(self, chassis_body, proper_collision=True):
        """Checks if the chassis is in contact with any asset"""
        # First check if the user wants to check for collision using mesh or bounding box
        if proper_collision:
            # Check for collision using the collision model
            for asset in self.assets_list:
                if (asset.body.GetContactForce().Length() > 0):
                    return 1
            return 0
        else:
            # Check for collision using the absolute position of the asset
            pos = chassis_body.GetPos()
            for asset_pos in self.positions:
                if (pos - asset_pos).Length() < (self.assets_list[0].scale * 2.5):
                    return 1
            return 0

