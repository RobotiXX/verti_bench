#!/usr/bin/env python3

import numpy as np

import torch

from std_msgs.msg import Header
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, Pose, PoseStamped, PoseArray, Quaternion, PolygonStamped,Polygon, Point32, PoseWithCovarianceStamped, PointStamped
import tf.transformations
import tf
import matplotlib.pyplot as plt

def angle_to_quaternion(angle):
    """Convert an angle in radians into a quaternion _message_."""
    return Quaternion(*tf.transformations.quaternion_from_euler(0, 0, angle))

def quaternion_to_angle(q):
    """Convert a quaternion _message_ into an angle in radians.
    The angle represents the yaw.
    This is not just the z component of the quaternion."""
    x, y, z, w = q.x, q.y, q.z, q.w
    roll, pitch, yaw = tf.transformations.euler_from_quaternion((x, y, z, w))
    return roll, pitch, yaw


# # Scary experimental canadian strategy:
# def _new_imod(self, a):
#     return torch.fmod(self, a, out=self)
# torch.Tensor.__imod__ = _new_imod

# modifies angles to be in range [-pi, pi]
def clamp_angle(angles):
    angles += np.pi
    angles %= (2 * np.pi)
    angles -= np.pi
    return angles

def clamp_angle_tensor_(angles):
    angles += np.pi
    torch.remainder(angles, 2*np.pi, out=angles)
    angles -= np.pi

def rotation_matrix(theta):
    c, s = np.cos(theta), np.sin(theta)
    return np.matrix([[c, -s], [s, c]])

def particle_to_posestamped(particle, frame_id):

    pose = PoseStamped()
    pose.header = make_header(frame_id)
    pose.pose.position.x = particle[0]
    pose.pose.position.y = particle[1]
    pose.pose.orientation = angle_to_quaternion(particle[2])
    return pose

def particle_to_pose(particle):
    pose = Pose()
    pose.position.x = particle[0]
    pose.position.y = particle[1]
    pose.orientation = angle_to_quaternion(particle[2])
    return pose

def particles_to_poses(particles):
    return map(particle_to_pose, particles)

def make_header(frame_id, stamp=None):
    if stamp == None:
        stamp = rospy.Time.now()
    header = Header()
    header.stamp = stamp
    header.frame_id = frame_id
    return header

def point(npt):
    pt = Point32()
    pt.x = npt[0]
    pt.y = npt[1]
    return pt

def points(arr):
    return map(point, arr)

def map_to_world(poses,map_info):
    scale = map_info.resolution
    angle = quaternion_to_angle(map_info.origin.orientation)

    # rotate

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp       + c*poses[:,1]

    # scale
    poses[:,:2] *= float(scale)

    # translate
    poses[:,0] += map_info.origin.position.x
    poses[:,1] += map_info.origin.position.y
    poses[:,2] += angle

def world_to_map(poses, map_info):
    # equivalent to map_to_grid(world_to_map(poses))
    # operates in place
    scale = map_info.resolution
    angle = -quaternion_to_angle(map_info.origin.orientation)

    # translation
    poses[:,0] -= map_info.origin.position.x
    poses[:,1] -= map_info.origin.position.y

    # scale
    poses[:,:2] *= (1.0/float(scale))

    # rotation
    c, s = np.cos(angle), np.sin(angle)
    # we need to store the x coordinates since they will be overwritten
    temp = np.copy(poses[:,0])
    poses[:,0] = c*poses[:,0] - s*poses[:,1]
    poses[:,1] = s*temp       + c*poses[:,1]
    poses[:,2] += angle

def world_to_map_torch(poses, map_info, angle, c, s):
    # equivalent to map_to_grid(world_to_map(poses))
    # operates in place
    scale = map_info.resolution

    # translation
    xs = poses[:,0]
    ys = poses[:,1]
    xs -= map_info.origin.position.x
    ys -= map_info.origin.position.y

    # scale
    poses[:,:2] *= (1.0 / float(scale))

    # we need to store the x coordinates since they will be overwritten
    temp = xs.clone()
    xs *= c
    # xs -= s * ys
    xs -= ys * s
    ys *= c
    ys += temp * s
    poses[:,2] += angle

# Ackermann calculation
def ackermann(throttle, steering, wheelbase):
    del_yaw = (throttle * torch.tan(steering)) / wheelbase
    #calculate the next position and add that in the trajectory
    del_x = throttle * torch.cos(del_yaw)
    del_y =  throttle * torch.sin(del_yaw)
    
    return del_x, del_y

# SE(3) transformation functions
def euler_to_rotation_matrix(euler_angles):
    """ Convert Euler angles to a rotation matrix """
    cos = torch.cos(euler_angles)
    sin = torch.sin(euler_angles)
    zero = torch.zeros_like(euler_angles[:, 0])
    one = torch.ones_like(euler_angles[:, 0])

    # Constructing rotation matrices (assuming 'xyz' convention for Euler angles)
    R_x = torch.stack([one, zero, zero, zero, cos[:, 0], -sin[:, 0], zero, sin[:, 0], cos[:, 0]], dim=1).view(-1, 3, 3)
    R_y = torch.stack([cos[:, 1], zero, sin[:, 1], zero, one, zero, -sin[:, 1], zero, cos[:, 1]], dim=1).view(-1, 3, 3)
    R_z = torch.stack([cos[:, 2], -sin[:, 2], zero, sin[:, 2], cos[:, 2], zero, zero, zero, one], dim=1).view(-1, 3, 3)

    return torch.matmul(torch.matmul(R_z, R_y), R_x)

def extract_euler_angles_from_se3_batch(tf3_matx):
    """ Extract Euler angles from SE3 homogeneous transformation matrices """
    if tf3_matx.shape[1:] != (4, 4):
        raise ValueError("Input tensor must have shape (batch, 4, 4)")

    rotation_matrices = tf3_matx[:, :3, :3]
    batch_size = tf3_matx.shape[0]
    euler_angles = torch.zeros((batch_size, 3), device=tf3_matx.device, dtype=tf3_matx.dtype)

    euler_angles[:, 0] = torch.atan2(rotation_matrices[:, 2, 1], rotation_matrices[:, 2, 2])  # Roll
    euler_angles[:, 1] = torch.atan2(-rotation_matrices[:, 2, 0], torch.sqrt(rotation_matrices[:, 2, 1] ** 2 + rotation_matrices[:, 2, 2] ** 2))  # Pitch
    euler_angles[:, 2] = torch.atan2(rotation_matrices[:, 1, 0], rotation_matrices[:, 0, 0])  # Yaw

    return euler_angles

def to_robot_torch(pose_batch1, pose_batch2):
    """ Transform poses from world frame to robot frame using SE3 """
    if pose_batch1.shape != pose_batch2.shape:
        raise ValueError("Input tensors must have same shape")
    if pose_batch1.shape[-1] != 6:
        raise ValueError("Input tensors must have last dim equal to 6")
        
    batch_size = pose_batch1.shape[0]
    ones = torch.ones_like(pose_batch2[:, 0])
    transform = torch.zeros_like(pose_batch1)
    T1 = torch.zeros((batch_size, 4, 4), device=pose_batch1.device, dtype=pose_batch1.dtype)
    T2 = torch.zeros((batch_size, 4, 4), device=pose_batch2.device, dtype=pose_batch2.dtype)

    T1[:, :3, :3] = euler_to_rotation_matrix(pose_batch1[:, 3:])
    T2[:, :3, :3] = euler_to_rotation_matrix(pose_batch2[:, 3:])
    T1[:, :3,  3] = pose_batch1[:, :3]
    T2[:, :3,  3] = pose_batch2[:, :3]
    T1[:,  3,  3] = 1
    T2[:,  3,  3] = 1 
    
    T1_inv = torch.inverse(T1)
    tf3_mat = torch.matmul(T2, T1_inv)
    
    transform[:, :3] = torch.matmul(T1_inv, torch.cat((pose_batch2[:,:3], ones.unsqueeze(-1)), dim=1).unsqueeze(2)).squeeze()[:, :3]
    transform[:, 3:] = extract_euler_angles_from_se3_batch(tf3_mat)
    return transform

# Speed calculation
def calculate_linear_speed_from_odom(prev_pose, current_pose, dt, f_size=7):
    # Transform current_pose to robot frame using SE(3) transformation
    if not isinstance(prev_pose, torch.Tensor):
        prev_pose = torch.tensor(prev_pose ,dtype=torch.float32)

    if not isinstance(current_pose, torch.Tensor):
        current_pose = torch.tensor(current_pose ,dtype=torch.float32)
    
    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt ,dtype=torch.float32)

    transformed_pose = to_robot_torch(prev_pose, current_pose)

    # Ensure dt is aligned with transformed_pose
    dt_length = min(dt.size(0), transformed_pose.size(0))
    dt = dt[:dt_length]  # Slice to match transformed_pose length

    # Compute translational speed (3D Euclidean distance)
    odom_linear_speed = torch.nn.functional.conv1d(transformed_pose[:, :1].t(), torch.ones((1,1,f_size))/f_size, padding='same') / dt
    # convolved = np.convolve(transformed_pose[:, 0].numpy(), np.ones(f_size) / f_size, mode='same')

    if odom_linear_speed.isinf().any(): 
        print("error in linear speed calculation")
    
    return torch.cat([torch.tensor([0]), odom_linear_speed.t().ravel()], dim=0)

def calculate_angular_speed_from_odom(prev_pose, current_pose, dt, f_size=7):
    # Transform current_pose to robot frame using SE(3) transformation
    if not isinstance(prev_pose, torch.Tensor):
        prev_pose = torch.tensor(prev_pose ,dtype=torch.float32)

    if not isinstance(current_pose, torch.Tensor):
        current_pose = torch.tensor(current_pose ,dtype=torch.float32)
    
    if not isinstance(dt, torch.Tensor):
        dt = torch.tensor(dt ,dtype=torch.float32)

    transformed_pose = to_robot_torch(prev_pose, current_pose)

    # Ensure dt is aligned with transformed_pose
    dt_length = min(dt.size(0), transformed_pose.size(0))
    dt = dt[:dt_length]  # Slice to match transformed_pose length

    # Compute translational speed (3D Euclidean distance)
    odom_angular_speed = torch.nn.functional.conv1d(transformed_pose[:, 1:2].t(), torch.ones((1,1,f_size))/f_size, padding='same') / dt
    # convolved = np.convolve(transformed_pose[:, 0].numpy(), np.ones(f_size) / f_size, mode='same')

    if odom_angular_speed.isinf().any(): 
        print("error in linear speed calculation")
    
    return torch.cat([torch.tensor([0]), odom_angular_speed.t().ravel()], dim=0)

# Label and input calculation
def calculate_vw_labels_and_inputs(cmd_vel, prev_pose, current_pose, dt, f_size=7):
    # Convert cmd_vel to a tensor
    if not isinstance(cmd_vel, torch.Tensor):
        cmd_vel = torch.tensor(cmd_vel ,dtype=torch.float32)

    odom_linear_speed = calculate_linear_speed_from_odom(prev_pose, current_pose, dt, f_size)
    odom_angular_speed = calculate_angular_speed_from_odom(prev_pose, current_pose, dt, f_size)
    
    L = 0.523
    # Calculate the difference between commanded speed and odometry speed
    real_linear_speed, real_angular_speed = ackermann(cmd_vel[:, :1]* 1.7941, cmd_vel[:, 1:]*0.7, L )
    v_diff = real_linear_speed - odom_linear_speed.unsqueeze(1)
    w_diff = real_angular_speed - odom_angular_speed.unsqueeze(1)

    # Combine the v_diff and w_diff for input to the model
    traversability = torch.cat((v_diff, w_diff), dim=1)

    return traversability

def calculate_xyz_labels_and_inputs(pose):
    # Assuming pose is a list of numpy arrays, convert it to a numpy array first
    pose = np.array(pose)

    # Calculate the differences (assuming pose[:, 0] is x, pose[:, 1] is y, pose[:, 2] is z)
    x_diff = torch.tensor(pose[:, 0]).float()
    y_diff = torch.tensor(pose[:, 1]).float()
    roll_diff = torch.tensor(pose[:, 3]).float()
    pitch_diff = torch.tensor(pose[:, 4]).float()

    # Reshape the differences to be 2D (N, 1) instead of 1D (N,)
    x_diff = x_diff.unsqueeze(1)
    y_diff = y_diff.unsqueeze(1)
    roll_diff = roll_diff.unsqueeze(1)
    pitch_diff = pitch_diff.unsqueeze(1)

    # Now concatenate along the second dimension
    traversability = torch.cat((x_diff, y_diff, roll_diff, pitch_diff), dim=1)

    return traversability
