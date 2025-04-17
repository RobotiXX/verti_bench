#!/usr/bin/env python3

import numpy as np
import math
import torch
from grid_map_msgs.msg import GridMap
import cv2
from geometry_msgs.msg import PoseStamped, Twist, Pose, Point, Quaternion
from std_msgs.msg import Header
from nav_msgs.msg import Odometry
import queue
import rospy
from copy import deepcopy

#Class for general functions
class utils:
    def __init__(self):
        self.queue_size = 0

    def rmap(self, value, from_min, from_max, to_min, to_max):
        # Calculate the range of the input value
        from_range = from_max - from_min

        # Calculate the range of the output value
        to_range = to_max - to_min

        # Scale the input value to the output range
        mapped_value = (value - from_min) * (to_range / from_range) + to_min

        return mapped_value

    def quaternion_to_yaw(self, quaternion):
        # Convert quaternion to yaw angle (in radians)
        quaternion_norm = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
        if (quaternion_norm == 0):
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        yaw = math.atan2(2.0 * (quaternion.w * quaternion.z + quaternion.x * quaternion.y),
                         1.0 - 2.0 * (quaternion.y**2 + quaternion.z**2))

        return yaw
    
    def quaternion_to_roll(self, quaternion):
        # Convert quaternion to roll angle (in radians)
        quaternion_norm = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
        if quaternion_norm == 0:
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        roll = math.atan2(2.0 * (quaternion.y * quaternion.z + quaternion.w * quaternion.x),
                        1.0 - 2.0 * (quaternion.x**2 + quaternion.y**2))

        return roll

    def quaternion_to_pitch(self, quaternion):
        # Convert quaternion to pitch angle (in radians)
        quaternion_norm = math.sqrt(quaternion.x**2 + quaternion.y**2 + quaternion.z**2 + quaternion.w**2)
        if quaternion_norm == 0:
            return 0.0
        quaternion.x /= quaternion_norm
        quaternion.y /= quaternion_norm
        quaternion.z /= quaternion_norm
        quaternion.w /= quaternion_norm

        pitch = math.asin(2.0 * (quaternion.w * quaternion.y - quaternion.z * quaternion.x))

        return pitch

    def yaw_to_quaternion(self, yaw):
        # Convert yaw angle (in radians) to quaternion
        quaternion = Quaternion()
        quaternion.x = 0.0
        quaternion.y = 0.0
        quaternion.z = math.sin(yaw / 2.0)
        quaternion.w = math.cos(yaw / 2.0)

        return quaternion
    
    def quaternion_to_rpy(self, quaternion):
        # Convert quaternion to roll, pitch, and yaw angles
        qw = quaternion.w
        qx = quaternion.x
        qy = quaternion.y
        qz = quaternion.z
        

        # Roll (x-axis rotation)
        sinr_cosp = 2.0 * (qw * qx + qy * qz)
        cosr_cosp = 1.0 - 2.0 * (qx * qx + qy * qy)
        roll = math.atan2(sinr_cosp, cosr_cosp)

        # Pitch (y-axis rotation)
        sinp = 2.0 * (qw * qy - qz * qx)
        if abs(sinp) >= 1:
            pitch = math.copysign(math.pi / 2, sinp)  # Use 90 degrees if out of range
        else:
            pitch = math.asin(sinp)

        # Yaw (z-axis rotation)
        siny_cosp = 2.0 * (qw * qz + qx * qy)
        cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
        yaw = math.atan2(siny_cosp, cosy_cosp)

        return roll, pitch, yaw

    def rpy_to_quaternion(self, roll, pitch, yaw):
        # Convert roll, pitch, and yaw angles to quaternion
        quaternion = Quaternion()
        cy = math.cos(yaw * 0.5)
        sy = math.sin(yaw * 0.5)
        cp = math.cos(pitch * 0.5)
        sp = math.sin(pitch * 0.5)
        cr = math.cos(roll * 0.5)
        sr = math.sin(roll * 0.5)

        quaternion.x = sr * cp * cy - cr * sp * sy
        quaternion.y = cr * sp * cy + sr * cp * sy
        quaternion.z = cr * cp * sy - sr * sp * cy
        quaternion.w = cr * cp * cy + sr * sp * sy

        return quaternion
    
    def clamp_angle(self, angles):
        angles += np.pi
        angles %= (2 * np.pi)
        angles -= np.pi
        return angles
    
    def clamp_angle_tensor_(self, angles):
        angles += np.pi
        torch.remainder(angles, 2*np.pi, out=angles)
        angles -= np.pi
        return angles

    def get_dist(self, start_pose, goal_pose):
        return math.sqrt((goal_pose.position.x - start_pose.position.x)**2 + (goal_pose.position.y - start_pose.position.y)**2)

    def create_pose_stamped(self, pose):
        # Create a PoseStamped message from a Pose message
        pose_stamped = PoseStamped()
        pose_stamped.pose = pose
        pose_stamped.header.stamp = rospy.Time.now()
        pose_stamped.header.frame_id = 'odom'  # Replace 'world' with your desired frame ID

        return pose_stamped

    def map_value(self, value, from_min, from_max, to_min, to_max):
        # Calculate the range of the input value
        from_range = from_max - from_min

        # Calculate the range of the output value
        to_range = to_max - to_min

        # Scale the input value to the output range
        mapped_value = (value - from_min) * (to_range / from_range) + to_min

        return mapped_value
    
    def make_header(self, frame_id, stamp=None):
        if stamp == None:
            stamp = rospy.Time.now()
        header = Header()
        header.stamp = stamp
        header.frame_id = frame_id
        return header

    def particle_to_posestamped(self, particle, frame_id):
        pose = PoseStamped()
        pose.header = self.make_header(frame_id)
        pose.pose.position.x = particle[0].item()
        pose.pose.position.y = particle[1].item()
        pose.pose.position.z = particle[2].item()
        pose.pose.orientation = self.rpy_to_quaternion(particle[3].item(), particle[4].item(), particle[5].item())
        return pose

    def euler_to_rotation_matrix(self, euler_angles):
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
    
    def extract_euler_angles_from_se3_batch(self, tf3_matx):
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

    def to_robot_torch(self, pose_batch1, pose_batch2):

        if pose_batch1.shape != pose_batch2.shape:
            raise ValueError("Input tensors must have same shape")

        if pose_batch1.shape[-1] != 6:
            raise ValueError("Input tensors must have last dim equal to 6")
            
        """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
        batch_size = pose_batch1.shape[0]
        ones = torch.ones_like(pose_batch2[:, 0])
        transform = torch.zeros_like(pose_batch1)
        T1 = torch.zeros((batch_size, 4, 4), device=pose_batch1.device, dtype=pose_batch1.dtype)
        T2 = torch.zeros((batch_size, 4, 4), device=pose_batch2.device, dtype=pose_batch2.dtype)

        T1[:, :3, :3] = self.euler_to_rotation_matrix(pose_batch1[:, 3:])
        T2[:, :3, :3] = self.euler_to_rotation_matrix(pose_batch2[:, 3:])
        T1[:, :3,  3] = pose_batch1[:, :3]
        T2[:, :3,  3] = pose_batch2[:, :3]
        T1[:,  3,  3] = 1
        T2[:,  3,  3] = 1 
        
        T1_inv = torch.inverse(T1)
        tf3_mat = torch.matmul(T2, T1_inv)
        
        transform[:, :3] = torch.matmul(T1_inv, torch.cat((pose_batch2[:,:3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()[:, :3]
        transform[:, 3:] = self.extract_euler_angles_from_se3_batch(tf3_mat)
        return transform

    def to_world_torch(self, Robot_frame, P_relative):

        if Robot_frame.shape != P_relative.shape:
            raise ValueError("Input tensors must have same shape")

        if Robot_frame.shape[-1] != 6:
            raise ValueError("Input tensors must have last dim equal to 6")
            
        """ Assemble a batch of SE3 homogeneous matrices from a batch of 6DOF poses """
        batch_size = Robot_frame.shape[0]
        ones = torch.ones_like(P_relative[:, 0])
        transform = torch.zeros_like(Robot_frame)
        T1 = torch.zeros((batch_size, 4, 4), device=Robot_frame.device, dtype=Robot_frame.dtype)
        T2 = torch.zeros((batch_size, 4, 4), device=P_relative.device, dtype=P_relative.dtype)

        R1 = self.euler_to_rotation_matrix(Robot_frame[:, 3:])
        R2 = self.euler_to_rotation_matrix(P_relative[:, 3:])
        
        T1[:, :3, :3] = R1
        T2[:, :3, :3] = R2
        T1[:, :3,  3] = Robot_frame[:, :3]
        T2[:, :3,  3] = P_relative[:, :3]
        T1[:,  3,  3] = 1
        T2[:,  3,  3] = 1 

        T_tf = torch.matmul(T2, T1)
        
        transform[:, :3] = torch.matmul(T1, torch.cat((P_relative[:,:3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()[:, :3]
        transform[:, 3:] = self.extract_euler_angles_from_se3_batch(T_tf)
        return transform

    def get_next_batch(self, o, xb, ps):
        o1 = o.detach().clone()   
        o1[:, 3:5] = 0.0
        n_pose = self.to_world_torch(ps, o1) 

        n_pose[:, 3:5] = o[:, 3:5]
        diff = self.to_robot_torch(ps, n_pose)
        
        xb[:, :6] = diff
        xb[:, 3] = n_pose[:, 3]
        xb[:, 4] = n_pose[:, 4]
        # xb[:,  6] = torch.sin(n_pose[:, 3])
        # xb[:,  7] = n_pose[:, 3]
        # xb[:,  8] = torch.sin(n_pose[:, 4])     
        # xb[:,  9] = n_pose[:, 4]
        # xb[:, 10] = torch.sin(n_pose[:, 5])
        # xb[:, 11] = torch.cos(n_pose[:, 5])

        return xb, n_pose
    
    def ackermann_model(self, input):
        """
        Calculates the change in pose (x, y, theta) for a batch of vehicles using the Ackermann steering model.
        
        Parameters:
        velocity (torch.Tensor): Tensor of shape (batch_size,) representing the velocity of each vehicle.
        steering (torch.Tensor): Tensor of shape (batch_size,) representing the steering angle of each vehicle.
        wheelbase (float): The distance between the front and rear axles of the vehicles.
        dt (float): Time step for the simulation.

        Returns:
        torch.Tensor: Tensor of shape (batch_size, 3) representing the change in pose (dx, dy, dtheta) for each vehicle.
        """
        # Ensure the velocity and steering tensors have the same batch size
        
        velocity = input[:, 0] / 0.75
        steering = -input[:, 1] * 0.6
        wheelbase = 0.320
        dt = 0.1

        # Calculate the change in orientation (dtheta)
        dtheta = velocity / wheelbase * torch.tan(steering) * dt

        # Calculate change in x and y coordinates
        dx = velocity * torch.cos(dtheta) * dt
        dy = velocity * torch.sin(dtheta) * dt

        # Stack the changes in x, y, and theta into one tensor
        pose_change = torch.stack((dx, dy, dx.clone()*0, dx.clone()*0, dx.clone()*0 ,  dtheta), dim=1)

        return pose_change
    
    def to_robot_se2(self, p1_batch, p2_batch):
        # Ensure the inputs are tensors
        p1_batch = torch.tensor(p1_batch, dtype=torch.float32)
        p2_batch = torch.tensor(p2_batch, dtype=torch.float32)

        # Validate inputs
        if p1_batch.shape != p2_batch.shape or p1_batch.shape[-1] != 3:
            raise ValueError("Both batches must be of the same shape and contain 3 elements per pose")

        # Extract components
        x1, y1, theta1 = p1_batch[:, 0], p1_batch[:, 1], p1_batch[:, 2]
        x2, y2, theta2 = p2_batch[:, 0], p2_batch[:, 1], p2_batch[:, 2]

        # Construct SE2 matrices
        zeros = torch.zeros_like(x1)
        ones = torch.ones_like(x1)
        T1 = torch.stack([torch.stack([torch.cos(theta1), -torch.sin(theta1), x1]),
                            torch.stack([torch.sin(theta1),  torch.cos(theta1), y1]),
                            torch.stack([zeros, zeros, ones])], dim=-1).permute(1,2,0)

        T2 = torch.stack([torch.stack([torch.cos(theta2), -torch.sin(theta2), x2]),
                            torch.stack([torch.sin(theta2),  torch.cos(theta2), y2]),
                            torch.stack([zeros, zeros, ones])], dim=-1).permute(1,2,0)

        # Inverse of T1 and transformation
        T1_inv = torch.inverse(T1)
        tf2_mat = torch.matmul(T2, T1_inv)

        # Extract transformed positions and angles
        transform = torch.matmul(T1_inv, torch.cat((p2_batch[:,:2], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()
        transform[:, 2] = torch.atan2(tf2_mat[:, 1, 0], tf2_mat[:, 0, 0])
        
        return transform

    def to_world_se2(self, p1_batch, p2_batch):
        # # Ensure the inputs are tensors
        # p1_batch = torch.tensor(p1_batch, dtype=torch.float32)
        # p2_batch = torch.tensor(p2_batch, dtype=torch.float32)

        # Validate inputs
        if p1_batch.shape != p2_batch.shape or p1_batch.shape[-1] != 3:
            raise ValueError("Both batches must be of the same shape and contain 3 elements per pose")

        # Extract components
        x1, y1, theta1 = p1_batch[:, 0], p1_batch[:, 1], p1_batch[:, 2]
        x2, y2, theta2 = p2_batch[:, 0], p2_batch[:, 1], p2_batch[:, 2]

        # Construct SE2 matrices
        zeros = torch.zeros_like(x1)
        ones = torch.ones_like(x1)
        T1 = torch.stack([torch.stack([torch.cos(theta1), -torch.sin(theta1), x1]),
                            torch.stack([torch.sin(theta1),  torch.cos(theta1), y1]),
                            torch.stack([zeros, zeros, ones])], dim=-1).permute(1,2,0)

        T2 = torch.stack([torch.stack([torch.cos(theta2), -torch.sin(theta2), x2]),
                            torch.stack([torch.sin(theta2),  torch.cos(theta2), y2]),
                            torch.stack([zeros, zeros, ones])], dim=-1).permute(1,2,0)

        # Inverse of T1 and transformation
        T_tf = torch.matmul(T2, T1)

        # Extract transformed positions and angles
        transform = torch.matmul(T1, torch.cat((p2_batch[:,:2], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()
        transform[:, 2] = torch.atan2(T_tf[:, 1, 0], T_tf[:, 0, 0])
        
        return transform

    def scale_out(self, data, scale_val, sid=0):
        # for scaling the output 
        if sid == 0:
            data[:, :3] = data[:, :3]*scale_val[0]
            data[:, 3:5] = data[:, 3:5]*scale_val[1]
            data[:, 5] = data[:, 5]*scale_val[2]
        
        # for scaling the map offset
        if sid == 1:
            data[:, :2] = data[:, :2]*scale_val[0]
            data[:, 2:] = data[:, 2:]*scale_val[1]
        return data
    
    def scale_in(self, data, scale_val, sid=0):
        # for scaling the output
        if sid == 0:
            data[:, :3] = data[:, :3] / scale_val[0]
            data[:, 3:5] = data[:, 3:5] / scale_val[1]
            data[:, 5] = data[:, 5] / scale_val[2]
        
        # for scaling the map offset
        if sid == 1:
            data[:, :2] = data[:, :2] / scale_val[0]
            data[:, 2:] = data[:, 2:] / scale_val[1]
        
        return data

    def get_next_batch_se2(self, model_output, prev_pose, scl, sid=0):
        if sid == 0:
            model_output = self.scale_out(model_output, scl, 0)
        next_pose = self.to_world_se2(prev_pose, model_output[:, [0,1,5]].clone())
        if sid == 0:
            return next_pose, self.scale_in(model_output, scl, 0)
        else:
            return next_pose, model_output

    def get_next_offsets_se2(self, map_offset_test, prev_pose, cur_pose, scl):
        diff = (cur_pose - prev_pose)[:, :2]
        map_offset_test = self.scale_out(map_offset_test, scl, 1)
        map_offset_test[:, :2] = map_offset_test[:, :2] - diff
        map_offset_test[:, 2] = torch.sin(cur_pose[:, 2])
        map_offset_test[:, 3] = torch.cos(cur_pose[:, 2])
        map_offset_test = self.scale_in(map_offset_test, scl, 1)
        return map_offset_test


#class for Neural net batch handeling
class NN_batch:
    def __init__(self):
        self.size = 2
        self.image_shape = (40, 100)
        self.batch_size = 0
        self.image_b = []
        self.pose_b = []
        self.twist_b = []
        self.ut = utils()

    def fill_array(self, b_size):
        for idx in range(b_size):
            self.batch_size = b_size 
            image = []
            pose =[]
            twist = []
            
            for i in range (self.size):
                image.append(np.zeros((self.image_shape), dtype=np.float32))
                pose.append(np.zeros((2, 3), dtype=np.float32))
                twist.append(np.zeros((2, 3), dtype=np.float32))
            
            self.image_b.append(image)
            self.pose_b.append(pose)
            self.twist_b.append(twist)

        return

    def batch_init(self, img, pose, twist, b_size): 
        self.batch_size = b_size
        self.image_b = []
        self.pose_b = []
        self.twist_b = []
        self.fill_array(b_size)  
        for idx in range(b_size): 
            for i in range (self.size):
                self.add_image(img, idx)
                self.add_pose(pose, idx)
                self.add_twist(twist, idx)       

    def add_image(self, img, idx):
        self.image_b[idx].pop(0)
        self.image_b[idx].append(img)
        
    def add_pose(self, pose, idx):
        self.pose_b[idx].pop(0)
        self.pose_b[idx].append(pose)

    def add_twist(self, twist, idx):
        self.twist_b[idx].pop(0)
        self.twist_b[idx].append(twist)

    def get_batch_asc(self):
        return self.image_b, self.pose_b, self.twist_b

    def get_batch_dsc(self):
        image_b_rev = []
        pose_b_rev = []
        twist_b_rev = []
        for i in range(self.batch_size):
            tmp1 = deepcopy(self.image_b[i])
            tmp1.reverse()
            image_b_rev.append(tmp1)
            tmp2 = deepcopy(self.pose_b[i])
            tmp2.reverse()
            pose_b_rev.append(tmp2)
            tmp3 = deepcopy(self.twist_b[i])
            tmp3.reverse()
            twist_b_rev.append(tmp3)

        return image_b_rev, pose_b_rev, twist_b_rev
    
    def get_img_batch_dsc(self):
        image_b_rev = []
        for i in range(self.batch_size):
            tmp1 = deepcopy(self.image_b[i])
            tmp1.reverse()
            image_b_rev.append(tmp1)

        return image_b_rev
    
    def get_rp_batch_asc(self):
        rp_batch = []
        for i in range(self.batch_size):
            r_b = []
            p_b = []
            for p in self.pose_b[i]:
                r, p = p[3], p[4]
                r_b.append(np.degrees(r))
                p_b.append(np.degrees(p))
            rp_b = p_b + r_b  
            rp_batch.append(rp_b)
        
        return rp_batch
    
    def get_rp_batch_dsc(self):
        rp_batch = []
        for i in range(self.batch_size):
            r_b = []
            p_b = []
            for p in self.pose_b[i]:
                r, p = p[3].item(), p[4].item()
                r_b.append(np.degrees(r))
                p_b.append(np.degrees(p))
            p_b.reverse()
            r_b.reverse()
            rp_b = p_b + r_b  
            rp_batch.append(rp_b)
        
        return rp_batch

    def get_rp_to_orientation(self, pred, yaw):
        r, p, y = np.radians(pred[1]), np.radians(pred[0]), yaw
        return self.ut.rpy_to_quaternion(r, p, y)


#Class for Odometry
class odom_processor:
    def __init__(self):
        self.odomframe = Odometry()
        self.pose = Pose()
        self.twist = Twist()

        self.initialized = False
        self.roll_bias = 0
        self.pitch_bias = 0
        self.yaw_bias = 0
        self.odom_loss = False
        
        self.cal_cnt = 1
        self.cal_delay = 50
        self.robot_pose = Pose()
        self.ut = utils()
        
    def calibrate(self, msg):
        if self.cal_cnt > 0 and self.cal_cnt < self.cal_delay and ~self.initialized:
            r, p, y = self.ut.quaternion_to_rpy(msg.pose.pose.orientation)
            self.roll_bias += r
            self.pitch_bias += p
            #self.yaw_bias += y
            self.cal_cnt += 1
        
        else:
            if ~self.initialized:
                self.roll_bias = self.roll_bias / self.cal_delay
                self.pitch_bias = self.pitch_bias / self.cal_delay
                #self.yaw_bias = self.yaw_bias / self.cal_delay
                self.initialized = True
    
    def update(self, msg):
        if self.initialized == False:
            self.calibrate(msg)
        
        else:
            self.odomframe = msg
            self.pose = msg.pose.pose
            self.twist = msg.twist.twist
            self.robot_pose = self.pose

            r, p, y = self.ut.quaternion_to_rpy(msg.pose.pose.orientation)
            
            if r == 0 and p == 0:
                self.odom_loss = True

            r = r - self.roll_bias
            p = p - self.pitch_bias
            y = y - self.yaw_bias

            self.robot_pose.orientation = self.ut.rpy_to_quaternion(r, p, y)

    def reset(self):
        self.initialized = False
        self.roll_bias = 0
        self.pitch_bias = 0        
        self.yaw_bias = 0
        self.cal_cnt = 1
        self.odom_loss = False

#hashable pose from geomettry pose message 
class HashablePose:
    def __init__(self, pose, cost):
        self.Pose = pose
        self.position = pose.position
        self.orientation = pose.orientation
        self.ut = utils()
        self.cost = cost

    def __hash__(self):
        r, p, y = self.ut.quaternion_to_rpy(self.orientation)
        return hash((self.position.x, self.position.y, self.position.z, r, p, y))
    
    def __lt__(self, other):
        return self.cost < other.cost
    
    def get_pose(self):
        return self.Pose
