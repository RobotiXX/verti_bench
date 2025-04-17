import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose, Point, Quaternion
import numpy as np
import math
from grid_map_msgs.msg import GridMap
import cv2
from multiprocessing import Pool

class MapProcessor:
    def __init__(self):
        self.map = GridMap()
        self.map_elevation = None #np.zeros((360,360), dtype=np.float32)
        self.map_traversal = None #np.zeros((360,360), dtype=np.float32)

        self.map_pose = GridMap().info.pose
        self.map_length = GridMap().info.length_x
        self.map_width = GridMap().info.length_y
        self.map_res = GridMap().info.resolution
        #self.map_res = 0.008
        self.map_layers = GridMap().layers
        self.map_shape = None #(360, 360)
        self.map_init = False #True
        
        self.fill = 0
        self.fill_val_div = 2

        self.robot_footprint = (0.32, 0.8)
        
    #update map data
    def update_map(self, map_data):
        self.map = map_data
        self.update_attributes()
        self.map_elevation = self.process_layer(self.map.data[0].data)
        self.map_traversal = self.process_layer(self.map.data[1].data)

    #update map info data
    def update_attributes(self):
        self.map_pose = self.map.info.pose
        self.map_length = self.map.info.length_x
        self.map_width = self.map.info.length_y
        self.map_res = self.map.info.resolution
        self.map_layers = self.map.layers
        self.map_shape = (int(self.map_length//self.map_res), int(self.map_width//self.map_res))


    #process layer data to remove NaN and reshape
    def process_layer(self, layer_data):
        layer_np = np.array(layer_data, dtype=np.float32)
        if np.isnan(layer_np).sum() > 0 and (~np.isnan(layer_np)).sum() != 0:
            if self.fill == 0 : 
                layer_np[np.isnan(layer_np)] = layer_np[~np.isnan(layer_np)].min()
            if self.fill == 1 :
                layer_np[np.isnan(layer_np)] = layer_np[~np.isnan(layer_np)].max()
            if self.fill == 2 :
                layer_np[np.isnan(layer_np)] = 0.0
            if self.fill == 3 :
                layer_np[np.isnan(layer_np)] = (layer_np[~np.isnan(layer_np)].max()-layer_np[~np.isnan(layer_np)].min())/self.fill_val_div

        layer_np =  layer_np.reshape((self.map_shape))

        return layer_np

    #rotate map in robot yaw orientation
    def rotate_map(self, Pose, robot_footprint_p, layer):
        if layer == 0:
            map_data = self.map_elevation
        
        if layer == 1:
            map_data = self.map_traversal
        
        angle = Pose[5]

        offset_x = math.cos(angle) * ((self.robot_footprint[1])/2)
        offset_y = math.sin(angle) * ((self.robot_footprint[1])/2)

        t_x = int((Pose[0] - offset_x - self.map_pose.position.x)//self.map_res)
        t_y = int((Pose[1] - offset_y - self.map_pose.position.y)//self.map_res)
        center = (int(self.map_shape[0]//2), int(self.map_shape[1]//2))
        
        #Shift point to the image center
        shifted_map = np.roll(np.roll(map_data,  -t_y, axis=0), t_x, axis=1)
        
    
        #rotate around the center
        M = cv2.getRotationMatrix2D(center, -np.degrees(angle), 1.0)
        rotated_map = cv2.warpAffine(shifted_map, M, (shifted_map.shape[1], shifted_map.shape[0] ))
        

        return rotated_map 
    
    #return cropped elevation map based on the robot footprint size
    def get_elev_footprint(self, Pose, shape):
        
        robot_footprint_p = (int(self.robot_footprint[0]//self.map_res), int(self.robot_footprint[1]//self.map_res))
        map = self.rotate_map(Pose, robot_footprint_p, 0)

        y = int((map.shape[0] - robot_footprint_p[0]) // 2 )
        x = int((map.shape[1] - robot_footprint_p[1]) // 2 )
        w = robot_footprint_p[1]
        h = robot_footprint_p[0]

        map_footprint = map[y:y+h, x:x+w].astype(np.float32)

        #cv2.imshow("Elevation Image", map_footprint_img)
        #cv2.waitKey(20)

        if shape[0] > shape[1]:
            map_footprint = cv2.rotate(map_footprint, cv2.ROTATE_90_CLOCKWISE)

        if map_footprint.shape != shape:
            map_footprint = cv2.resize(map_footprint, (shape[1], shape[0]))


        return map_footprint

    #return cropped traversability map based on the robot footprint size
    def get_trav_footprint(self, Pose, shape):
        
        robot_footprint_p = (int(self.robot_footprint[0]//self.map_res), int(self.robot_footprint[1]//self.map_res))
        map = self.rotate_map(Pose, robot_footprint_p, 1)

        y = int((map.shape[0] - robot_footprint_p[0]) // 2 )
        x = int((map.shape[1] - robot_footprint_p[1]) // 2 )
        w = robot_footprint_p[1]
        h = robot_footprint_p[0]

        map_footprint = map[y:y+h, x:x+w].astype(np.float32)

        #cv2.imshow("Elevation Image", map_footprint_img)
        #cv2.waitKey(20)

        if shape[0] > shape[1]:
            map_footprint = cv2.rotate(map_footprint, cv2.ROTATE_90_CLOCKWISE)

        if map_footprint.shape != shape:
            map_footprint = cv2.resize(map_footprint, (shape[1], shape[0]))


        return map_footprint

    #return hight at specific point on gridmap
    def get_pose_height(self, Pose):
        if self.is_on_map(Pose)==False:
            return Pose[2]+0.183
        
        t_x = int((Pose[0] - self.map_pose.position.x)//self.map_res) + 1
        t_y = int((Pose[1] - self.map_pose.position.y)//self.map_res) + 1
        center = (int(self.map_shape[0]//2), int(self.map_shape[1]//2))
        h = self.map_elevation[min((center[0] + t_y, (self.map_shape[1]-1)))][min((center[1] - t_x),(self.map_shape[0]-1))]
        
        return h.item()+ 0.183

    #return true is the pose is on the gridmap
    def is_on_map(self, Pose):

        center = self.map_pose.position
        x_low = center.x - self.map_length/2
        x_high = center.x + self.map_length/2
        y_low = center.y - self.map_width/2
        y_high = center.y + self.map_width/2
        if (x_low < Pose[0]) and (Pose[0] < x_high) and (y_low < Pose[1]) and (Pose[1] < y_high):
            return True
        
        else:
            return False
