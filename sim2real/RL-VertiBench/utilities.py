import numpy as np
import math
from grid_map_msgs.msg import GridMap
import cv2
from geometry_msgs.msg import PoseStamped, Twist, Pose, Point, Quaternion
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
    
    def get_rp_batch_asc(self):
        rp_batch = []
        for i in range(self.batch_size):
            r_b = []
            p_b = []
            for p in self.pose_b[i]:
                r, p, y = self.ut.quaternion_to_rpy(p.orientation)
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
                r, p, y = self.ut.quaternion_to_rpy(p.orientation)
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
