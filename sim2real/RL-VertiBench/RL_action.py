
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor, MlpExtractor
from stable_baselines3.common.utils import get_linear_fn
import rospy
import numpy as np
from geometry_msgs.msg import PoseStamped, Twist, Pose
from nav_msgs.msg import Odometry
from grid_map_msgs.msg import GridMap
from copy import deepcopy
from swae_model import SWAE, LatentSpaceMapper
from Grid import MapProcessor
from utilities import utils, odom_processor
import math
from torch.distributions import Normal

class ChQuaternion:
    def __init__(self, e0, e1, e2, e3):
        self.m_data = [e0, e1, e2, e3]    
        
    def rotate_back(self, A):
        e0e0 = +self.m_data[0] * self.m_data[0]
        e1e1 = +self.m_data[1] * self.m_data[1]
        e2e2 = +self.m_data[2] * self.m_data[2]
        e3e3 = +self.m_data[3] * self.m_data[3]
        e0e1 = -self.m_data[0] * self.m_data[1]
        e0e2 = -self.m_data[0] * self.m_data[2]
        e0e3 = -self.m_data[0] * self.m_data[3]
        e1e2 = +self.m_data[1] * self.m_data[2]
        e1e3 = +self.m_data[1] * self.m_data[3]
        e2e3 = +self.m_data[2] * self.m_data[3]        
        
        x = ((e0e0 + e1e1) * 2 - 1) * A[0] + ((e1e2 - e0e3) * 2) * A[1] + ((e1e3 + e0e2) * 2) * A[2]
        y = ((e1e2 + e0e3) * 2) * A[0] + ((e0e0 + e2e2) * 2 - 1) * A[1] + ((e2e3 - e0e1) * 2) * A[2]
        z = ((e1e3 - e0e2) * 2) * A[0] + ((e2e3 + e0e1) * 2) * A[1] + ((e0e0 + e3e3) * 2 - 1) * A[2]        
        
        return np.array([x, y, z])

def constant_schedule(initial_value):
    """
    Constant learning rate schedule.

    :param initial_value: The constant learning rate value.
    :return: A function that returns the constant learning rate.
    """
    def schedule(_):
        return initial_value
    return schedule

class StudentPolicy(nn.Module):
    def __init__(self, input_dim=18):  # 16 + 2 for additional inputs
        super().__init__()
        self.policy_network = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, 2)  # Output: [speed, steering]
        )
    
    def forward(self, x):
        return self.policy_network(x)
    
    def predict(self, observation, deterministic=True):
        # Add this method to match the PPO interface
        with torch.no_grad():
            observation = torch.FloatTensor(observation).to(next(self.parameters()).device)
            action = self(observation)
        return action.numpy(), None

class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(
        self,
        observation_space,
        action_space,
        lr_schedule,
        net_arch=None,
        activation_fn=nn.ReLU,
        features_extractor_class=None,
        features_extractor_kwargs=None,
        optimizer_class=torch.optim.Adam,
        optimizer_kwargs=None,
    ):
        # Call nn.Module's init first
        nn.Module.__init__(self)
        
        self.observation_space = observation_space
        self.action_space = action_space
        self.lr_schedule = lr_schedule
        
        if net_arch is None:
            net_arch = [dict(pi=[64, 64], vf=[64, 64])]
        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = {} if optimizer_kwargs is None else optimizer_kwargs

        # Create MLPExtractor directly without feature extractor
        self.mlp_extractor = MlpExtractor(
            observation_space.shape[0],
            net_arch=self.net_arch,
            activation_fn=self.activation_fn,
        )

        # Create action net
        latent_dim_pi = self.mlp_extractor.latent_dim_pi
        self.action_net = nn.Linear(latent_dim_pi, action_space.shape[0])

        # Add log_std parameter
        self.log_std = nn.Parameter(torch.zeros(action_space.shape[0]))

        # Create value net
        latent_dim_vf = self.mlp_extractor.latent_dim_vf
        self.value_net = nn.Linear(latent_dim_vf, 1)

    def forward(self, obs, deterministic=False):
        # Convert obs to tensor if needed
        if isinstance(obs, np.ndarray):
            obs = torch.as_tensor(obs).float()
            
        # Extract features - changed to pass only obs once
        latent_pi, latent_vf = self.mlp_extractor(obs)
        
        # Get mean actions
        mean_actions = self.action_net(latent_pi)
        
        # Create distribution using log_std
        distribution = Normal(mean_actions, self.log_std.exp())
        
        # Get actions
        if deterministic:
            actions = mean_actions
        else:
            actions = distribution.sample()
            
        # Get values
        values = self.value_net(latent_vf)
        
        log_prob = distribution.log_prob(actions).sum(dim=-1)
        
        return actions, values, log_prob

    def _predict(self, observation, deterministic=True):
        with torch.no_grad():
            observation = torch.as_tensor(observation).float()
            actions, _, _ = self.forward(observation, deterministic)
            if isinstance(actions, torch.Tensor):
                actions = actions.detach().cpu().numpy()
            return actions
        
        
class RLActionGenerator:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.map_initialized = False
        self.goal_received = False
        print("Waiting for elevation map initialization...")
        
        # Define observation and action spaces
        self.features_dim = 16
        low_bound = np.concatenate(([-1] * self.features_dim, [-1, -1]))
        high_bound = np.concatenate(([1] * self.features_dim, [1, 1]))
        self.observation_space = gym.spaces.Box(
            low=low_bound, 
            high=high_bound, 
            shape=(self.features_dim + 2,), 
            dtype=np.float32
        )
        self.action_space = gym.spaces.Box(
            low=np.array([-1.0, -1.0]), 
            high=np.array([1.0, 1.0]),
            shape=(2,), 
            dtype=np.float32
        )

        # Learning rate schedule
        self.lr_schedule = constant_schedule(5e-4)

        # Load the PPO model
        self.model = self.load_ppo_model()
        self.swae = self.load_swae_model()
        self.current_speed = 0.0
        self.mp = MapProcessor()
        self.uts = utils()
        self.odp = odom_processor()
        self.startpose = Pose()
        self.goalpose = Pose()
        self.robotpose = Pose()
        self.val_goal = False
        
        # Load and initialize other components
        self.latent_space_mapper = LatentSpaceMapper(64, self.features_dim).to(self.device)
        self.min_vectorBench = torch.tensor(np.load("/home/jetson/Documents/RL-VertiBench/min_vectorBench.npy")).to(self.device)
        self.max_vectorBench = torch.tensor(np.load("/home/jetson/Documents/RL-VertiBench/max_vectorBench.npy")).to(self.device)

        # Initialize goalpose to be the same as robotpose initially
        self.reset_goal_pose()

        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        rospy.Subscriber('/elevation_mapping/elevation_map_raw', GridMap, self.gridMap_callback, queue_size=1)
        #rospy.Subscriber('dlio/odom_node/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/natnet_ros/crawler/odom', Odometry, self.odom_callback, queue_size=1)

    def reset_goal_pose(self):
        """Initialize goalpose to match the robot's initial pose."""
        self.goalpose.position.x = self.robotpose.position.x
        self.goalpose.position.y = self.robotpose.position.y
        self.goalpose.position.z = self.robotpose.position.z
        self.goalpose.orientation = deepcopy(self.robotpose.orientation)

    def load_ppo_model(self):
        policy_kwargs = dict(
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],  
            activation_fn=nn.ReLU,
        )

        model = CustomActorCriticPolicy(
            observation_space=self.observation_space,
            action_space=self.action_space,
            lr_schedule=self.lr_schedule,
            features_extractor_class=None,
            features_extractor_kwargs=None,
            net_arch=policy_kwargs['net_arch'],
            activation_fn=policy_kwargs['activation_fn'],
        )

        model.load_state_dict(torch.load("/home/jetson/Documents/RL-VertiBench/teacher_grass.pt", map_location=self.device))
        model.to(self.device)
        model.eval()

        return model
    
    def load_student_model(self):
        model = StudentPolicy(input_dim=18).to(self.device)
        model.load_state_dict(torch.load("/home/jetson/Documents/RL-VertiBench/student_policy.pth", 
                                       map_location=self.device))
        model.eval()
        return model

    def load_swae_model(self):
        swae = SWAE(in_channels=1, latent_dim=64)
        swae.load_state_dict(torch.load("/home/jetson/Documents/RL-VertiBench/BenchElev.pth", map_location=self.device))
        swae.freeze_encoder()
        swae.to(self.device)
        swae.eval()
        return swae

    def odom_callback(self, odom_msg):
        self.odp.update(odom_msg)
        self.robotpose = deepcopy(self.odp.robot_pose)
        self.current_speed = odom_msg.twist.twist.linear.x
        # print(f"Current speed: {self.current_speed}")
        if not self.val_goal:
            self.reset_goal_pose()
    
    def goal_callback(self, goal_msg):
        if not self.map_initialized:
            print("Waiting for map initialization before accepting goals...")
            return
            
        print("Goal Received!")
        self.goalpose = deepcopy(goal_msg.pose)
        self.startpose = deepcopy(self.robotpose)
        self.val_goal = True
        self.goal_received = True

    def gridMap_callback(self, gridmap_msg):
        self.mp.update_map(gridmap_msg)
        if not self.map_initialized:
            print("Elevation map initialized, ready for goals!")
            self.map_initialized = True
        self.mp.map_init = True

    def get_goal_and_speed(self):
        # Extract positions from the current robot pose and goal pose
        robot_x = self.robotpose.position.x
        robot_y = self.robotpose.position.y
        goal_x = self.goalpose.position.x
        goal_y = self.goalpose.position.y

        # Compute the difference in positions (relative position in global frame)
        delta_x = goal_x - robot_x
        delta_y = goal_y - robot_y

        # Create the vector to the goal in the global frame
        m_vector_to_goal = np.array([delta_x, delta_y, 0])
        q_w = self.robotpose.orientation.w
        q_x = self.robotpose.orientation.x
        q_y = self.robotpose.orientation.y
        q_z = self.robotpose.orientation.z

        # Create the ChQuaternion object
        robot_orientation = ChQuaternion(q_w, q_x, q_y, q_z)    # Transform the goal's position to the vehicle's local coordinate frame using rotate_back
        vector_to_goal_local = robot_orientation.rotate_back(m_vector_to_goal)    # Calculate the heading to the goal in the vehicle's local coordinate frame
        target_heading_local = np.arctan2(vector_to_goal_local[1], vector_to_goal_local[0])    # Calculate the heading difference in the local frame (assuming current heading is 0)
        heading_diff_local = (target_heading_local - 0 + np.pi) % (2 * np.pi) - np.pi
        normalized_heading_diff = heading_diff_local / np.pi
        normalized_heading_diff = (normalized_heading_diff)
        
        print(f"Heading diff norm: {normalized_heading_diff}")
        
        # Get the current speed of the vehicle
        speed = self.current_speed
        speed = np.clip(speed, -1.0, 1.0)
        
        return normalized_heading_diff, speed

    def get_observation(self):
        if self.robotpose is None:
            rospy.logwarn("Robot pose is None, returning zeros as observation.")
            return np.zeros((self.features_dim + 2,), dtype=np.float32)

        _, elevation_map = self.mp.get_elev_footprint(self.robotpose, (64, 64))

        # Normalize the elevation_map to be in the range [-1, 1]
        min_val = np.min(elevation_map)
        max_val = np.max(elevation_map)
        if max_val > min_val:
            elevation_map = 2 * ((elevation_map - min_val) / (max_val - min_val)) - 1
        else:
            elevation_map = np.zeros_like(elevation_map)

        elevation_map = torch.tensor(elevation_map, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(self.device)

        _, _, z = self.swae(elevation_map) #64*64 -> 64*1
        z_normalized = 2 * (z - self.min_vectorBench) / (self.max_vectorBench - self.min_vectorBench) - 1
        mapped_features_tensor = self.latent_space_mapper(z_normalized) # 64*1 -> 16*1
        mapped_features_array = mapped_features_tensor.cpu().detach().numpy().flatten()
        
        heading_to_goal, speed = self.get_goal_and_speed()
        observation = np.concatenate((mapped_features_array, [heading_to_goal, speed])).astype(np.float32)
        return observation

    def compute_command(self):
        obs = self.get_observation()
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.model._predict(obs_tensor, deterministic=True)
        return action.flatten()

### Main ###
if __name__ == '__main__':
    rospy.init_node('RL_action_generator', anonymous=True)
    generator = RLActionGenerator()
    action = Twist()
    action_pub = rospy.Publisher("/cmd_vel", Twist, queue_size=1)
    rate = rospy.Rate(30)
    
    # Wait for map initialization
    while not generator.map_initialized and not rospy.is_shutdown():
        print("Waiting for elevation map...", end='\r')
        rate.sleep()
    print("\nMap initialized! Waiting for goal...")
    
    while not rospy.is_shutdown():
        if not generator.goal_received:
            print("Waiting for goal...", end='\r')
            rate.sleep()
            continue
        
        vw = generator.compute_command()
        print(f"Computed command: {vw}")
        if vw.size != 2:
            rospy.logwarn("Action does not have the correct dimensions!")
        else:
            # data[0] is steering
            # data[1] is throttle
            action.linear.x = vw[1] / 4
            action.angular.z = -vw[0]
            action_pub.publish(action)
        rate.sleep()
