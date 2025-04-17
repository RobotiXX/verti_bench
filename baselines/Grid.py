import rospy
from geometry_msgs.msg import PoseStamped, Twist, Pose, Point, Quaternion
import numpy as np
import math
from grid_map_msgs.msg import GridMap
import cv2
from utilities import utils
from copy import deepcopy
from scipy.interpolate import griddata
import torch
import torch.nn as nn
import torchvision.transforms.functional as F
import torch.nn.parallel as parallel
from multiprocessing import Pool

class MapProcessor:
    def __init__(self):
        self.map = GridMap()
        self.map_elevation = np.zeros((600, 600), dtype=np.float32)
        self.map_traversal = np.zeros((600, 600), dtype=np.float32)

        self.map_elev_img = np.zeros((600, 600), dtype=np.uint8)
        self.map_trav_img = np.zeros((600, 600), dtype=np.uint8)

        self.map_pose = GridMap().info.pose
        self.map_length = GridMap().info.length_x
        self.map_width = GridMap().info.length_y
        self.map_res = GridMap().info.resolution
        self.map_layers = GridMap().layers
        self.map_shape = [250,250]
        self.map_init = False
        self.device = torch.device("cuda")
        self.robot_pose = Pose()
        self.map_embedding = None
        self.util = utils()
        
        self.fill = 0
        self.fill_val_div = 2

        # self.robot_footprint = (0.32, 0.8)
        self.robot_footprint = (3.2, 8)
        
    #update map data
    def update_map(self, map_data):
        self.map = map_data
        self.update_attributes()
        self.map_elevation, self.map_elev_img = self.process_layer(self.map.data[0].data)
        self.map_traversal, self.map_trav_img = self.process_layer(self.map.data[1].data)

    #process layer data to remove NaN and reshape
    def process_layer(self, layer_data):
        layer_np = np.array(layer_data)
        layer_valid = layer_np[~np.isnan(layer_np)]
        min_value = np.min(layer_valid)
        max_value = np.max(layer_valid)

        if self.fill == 0 :
            layer_np[np.isnan(layer_np)] = min_value
        if self.fill == 1 :
            layer_np[np.isnan(layer_np)] = max_value
        if self.fill == 2 :
            layer_np[np.isnan(layer_np)] = 0.0
        if self.fill == 3 :
            layer_np[np.isnan(layer_np)] =  (max_value - min_value)/self.fill_val_div
        if self.fill == 4 :
            layer_np[np.isnan(layer_np)] =  self.robot_pose.position.z
        if self.fill == 5 :
            layer_np[np.isnan(layer_np)] = 0.0

        layer_grid =  layer_np.reshape((self.map_shape))
        layer_norm = (layer_np - min_value) / (max_value - min_value)
        layer_scaled = (layer_norm * 255).astype(np.uint8)
        layer_scaled = layer_scaled.reshape((self.map_shape))

        if self.fill == 5:
            layer_grid = np.array(layer_data).reshape((self.map_shape))

        layer_grid = cv2.flip(layer_grid, 0).astype(np.float32)
        layer_scaled = cv2.flip(layer_scaled, 0).astype(np.float32)


         
        return layer_grid, layer_scaled 

    #update map info data
    def update_attributes(self):
        self.map_pose = self.map.info.pose
        self.map_length = self.map.info.length_x
        self.map_width = self.map.info.length_y
        self.map_res = self.map.info.resolution
        self.map_layers = self.map.layers
        self.map_shape = (int(self.map_length//self.map_res), int(self.map_width//self.map_res))

    #rotate map in robot yaw orientation
    def rotate_map(self, Pose, robot_footprint_p, layer):
        if layer == 0:
            map_data = self.map_elevation
            map_data_img = self.map_elev_img
        
        if layer == 1:
            map_data = self.map_traversal
            map_data_img = self.map_trav_img
        
        angle = Pose[5].item()

        offset_x = math.cos(angle) * ((self.robot_footprint[1])/2)
        offset_y = math.sin(angle) * ((self.robot_footprint[1])/2)

        t_x = int((Pose[0] - offset_x - self.map_pose.position.x)//self.map_res)
        t_y = int((Pose[1] - offset_y - self.map_pose.position.y)//self.map_res)
        center = (int(self.map_shape[0]//2), int(self.map_shape[1]//2))
        

        #Shift point to the image center
        shifted_map = np.roll(map_data,   -t_y, axis=0)
        
        shifted_map = np.roll(shifted_map, t_x, axis=1)

        # shifted_map_img = np.roll(map_data_img,   -t_y, axis=0)
        # shifted_map_img = np.roll(shifted_map_img, t_x, axis=1)
        
        start_time = rospy.Time.now()
        
        '''
        #rotate around the center
        M = cv2.getRotationMatrix2D(center, -np.degrees(angle), 1.0)
        rotated_map = cv2.warpAffine(shifted_map, M, (shifted_map.shape[1], shifted_map.shape[0] ))
        rotated_map_img = cv2.warpAffine(shifted_map_img, M, (self.map_shape))        
        '''
        shifted_map = np.expand_dims(shifted_map, axis=0)
        # shifted_map_img = np.expand_dims(shifted_map_img, axis=0)

        r_map = torch.tensor(shifted_map, dtype=torch.float).to(self.device)
        # rmap_img = torch.tensor(shifted_map_img, dtype=torch.uint8).to(self.device)

        rotated_map = np.array((F.rotate(r_map, -np.degrees(angle)).to(self.device)).squeeze().cpu(), dtype=np.float32)
        # rotated_map_img = np.array((F.rotate(rmap_img, -np.degrees(angle)).to(self.device)).squeeze().cpu(), dtype=np.uint8)
        

        return rotated_map #, rotated_map_img
    
    #return cropped elevation map based on the robot footprint size
    def get_elev_footprint(self, Pose, shape):
        

        robot_footprint_p = (int(self.robot_footprint[0]//self.map_res), int(self.robot_footprint[1]//self.map_res))

        map = self.rotate_map(Pose, robot_footprint_p, 0)

        y = int((map.shape[0] - robot_footprint_p[0]) // 2 )
        x = int((map.shape[1] - robot_footprint_p[1]) // 2 )
        w = robot_footprint_p[1]
        h = robot_footprint_p[0]

        map_footprint = map[y:y+h, x:x+w].astype(np.float32)
        # map_footprint_img = map_img[y:y+h, x:x+w].astype(np.float32)

        #cv2.imshow("Elevation Image", map_footprint_img)
        #cv2.waitKey(20)

        if shape[0] > shape[1]:

            map_footprint = cv2.rotate(map_footprint, cv2.ROTATE_90_CLOCKWISE)
            # map_footprint_img = cv2.rotate(map_footprint_img, cv2.ROTATE_90_CLOCKWISE)

        #v2.imshow("Elevation Image", map_footprint_img)
        #cv2.waitKey(20)

        if map_footprint.shape != shape:
            map_footprint = cv2.resize(map_footprint, (shape[1], shape[0]))
            # map_footprint_img = cv2.resize(map_footprint_img, (shape[1], shape[0]))

        #cv2.imshow("Elevation Image", map_footprint_img)
        #cv2.waitKey(100)


        return map_footprint #, map_footprint_img

    #return cropped traversability map based on the robot footprint size
    def get_trav_footprint(self, Pose, shape):
        
        robot_footprint_p = (int(self.robot_footprint[0]//self.map_res), int(self.robot_footprint[1]//self.map_res))

        map, map_img = self.rotate_map(Pose, robot_footprint_p, 1)

        y = int((map.shape[0] - robot_footprint_p[0]) // 2 )
        x = int((map.shape[1] - robot_footprint_p[1]) // 2 )
        w = robot_footprint_p[1]
        h = robot_footprint_p[0]

        map_footprint = map[y:y+h, x:x+w]
        map_footprint_img = map_img[y:y+h, x:x+w]

        #cv2.imshow("Elevation Image", map_footprint_img)
        #cv2.waitKey(20)

        if shape[0] > shape[1]:

            map_footprint = cv2.rotate(map_footprint, cv2.ROTATE_90_CLOCKWISE)
            map_footprint_img = cv2.rotate(map_footprint_img, cv2.ROTATE_90_CLOCKWISE)

        #v2.imshow("Elevation Image", map_footprint_img)
        #cv2.waitKey(20)

        if map_footprint.shape != shape:
            map_footprint = cv2.resize(map_footprint, (shape[1], shape[0]))
            map_footprint_img = cv2.resize(map_footprint_img, (shape[1], shape[0]))

        #cv2.imshow("Elevation Image", map_footprint_img)
        #cv2.waitKey(20)

        return map_footprint, map_footprint_img

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

    #return array of the footprints     
    def get_elev_footprint_torch(self, Pose_a, sn, shape):


        robot_footprint_p = (int(self.robot_footprint[0]//self.map_res), int(self.robot_footprint[1]//self.map_res))

        map_data = self.map_elevation
        map_data_img = self.map_elev_img
        center = (int(self.map_shape[0]//2), int(self.map_shape[1]//2))

        angel_list = []
        shifted_map_list = []
        #shifted_map_img_list = []


        for pos in Pose_a:
            Pose = pos[sn]
            angle = self.util.quaternion_to_yaw(Pose.orientation)
            angel_list.append(-np.degrees(angle))
            offset_x = math.cos(angle) * ((self.robot_footprint[1])/2)
            offset_y = math.sin(angle) * ((self.robot_footprint[1])/2)
            t_x = int((Pose.position.x - offset_x - self.map_pose.position.x)//self.map_res)
            t_y = int((Pose.position.y - offset_y - self.map_pose.position.y)//self.map_res)

            
            #Shift point to the image center
            shifted_map = np.roll(map_data, -t_y, axis=0)
            shifted_map = np.roll(shifted_map, t_x, axis=1)

            #shifted_map_img = np.roll(map_data_img, -t_y, axis=0)
            #shifted_map_img = np.roll(shifted_map_img, t_x, axis=1)

            
            shifted_map_list.append(torch.tensor(np.expand_dims(shifted_map, axis=0), dtype=torch.float32))
            #shifted_map_img_list.append(torch.tensor(np.expand_dims(shifted_map_img, axis=0), dtype=torch.uint8))
            
       
        
        f_map =[]
        #f_map_img = []

        for i in range(len(shifted_map_list)):
            f_map.append(F.rotate(shifted_map_list[i].to(self.device), angel_list[i]))
            #f_map_img.append(F.rotate(shifted_map_img_list[i].to(self.device), angel_list[i]) ) 
        

               

        f_map = torch.cat(f_map, dim=0)
        #f_map_img = torch.cat(f_map_img, dim=0)


        y = int((self.map_elevation.shape[0] - robot_footprint_p[0]) // 2 )
        x = int((self.map_elevation.shape[1] - robot_footprint_p[1]) // 2 )
        w = robot_footprint_p[1]
        h = robot_footprint_p[0]

        #map_list = np.array((F.crop(f_map, y, x, h, w)).cpu() , dtype=np.float32)
        #map_img_list = np.array((F.crop(f_map_img, y, x, h, w)).cpu() , dtype=np.uint8)

        f_map= F.crop(f_map, y, x, h, w)
        #f_map_img = F.crop(f_map_img, y, x, h, w)

        if shape[0] > shape[1]:
            f_map = F.rotate(f_map , -90)
            #f_map_img = F.rotate(f_map_img, -90)

        if f_map.shape[1] != shape[0]:
            f_map = F.resize(f_map , (shape[1], shape[0]))
            #f_map_img = F.resize(f_map_img, (shape[1], shape[0]))

        map_list =  np.array(f_map.cpu(), dtype=np.float32)
        #map_img_list = np.array(f_map_img.cpu(), dtype=np.float32)


        f_map.detach()
        #f_map_img.detach()

        torch.cuda.empty_cache()
        #cv2.imshow("Elevation Image", map_img_list[0])
        #cv2.waitKey(20)

        return map_list, #map_img_list

        map_footprint_list = []
        map_footprint_img_list = []

        for ind in range(map_list.shape[0]):
            map = map_list[ind]
            map_img = map_img_list[ind]
            y = int((map.shape[0] - robot_footprint_p[0]) // 2 )
            x = int((map.shape[1] - robot_footprint_p[1]) // 2 )
            w = robot_footprint_p[1]
            h = robot_footprint_p[0]

            map_footprint = map[y:y+h, x:x+w]
            map_footprint_img = map_img[y:y+h, x:x+w]

            if shape[0] > shape[1]:
                map_footprint = cv2.rotate(map_footprint, cv2.ROTATE_90_CLOCKWISE)
                map_footprint_img = cv2.rotate(map_footprint_img, cv2.ROTATE_90_CLOCKWISE)

            if map_footprint.shape != shape:
                map_footprint = cv2.resize(map_footprint, (shape[1], shape[0]))
                map_footprint_img = cv2.resize(map_footprint_img, (shape[1], shape[0]))

            map_footprint_list.append(map_footprint)
            map_footprint_img_list.append(map_footprint_img)

        #elapsed_time = rospy.Time.now() - start_time  
        #execution_time = elapsed_time.to_sec()
        #print("Execution time: %.4f seconds", execution_time)

        return map_footprint_list, map_footprint_img_list