import rospy
from nav_msgs.msg import Odometry, Path
from tf2_ros import TransformBroadcaster
from geometry_msgs.msg import TransformStamped
import os

no_of_robots = 4

class RobotStatePublisher:
    def __init__(self):
        # Initialize the ROS node
        rospy.init_node('robot_state_publisher', anonymous=True)

        # Parameters for robot configuration
        self.odom_frame_id = rospy.get_param('~odom_frame_id', 'odom')
        self.robot_frame_id = []
        self.robot_description_param = []

        for i in range(no_of_robots):
            self.robot_frame_id.append(rospy.get_param(f'~robot_frame_id_{i}', f'base_link_{i}'))

        for i in range(no_of_robots):
            self.robot_description_param.append(rospy.get_param(f'~robot_description_param_{i}', f'/robot_description_{i}'))

        # Publish the robot description
        self.publish_robot_description()

        # Create a Transform Broadcaster
        self.tf_broadcaster = TransformBroadcaster()

        # Subscribe to odometry topic
        rospy.Subscriber('/natnet_ros/crawler/odom', Odometry, self.odom_callback)
        
        # rospy.Subscriber('/paths_gt', Path, self.path_callback)

        rospy.loginfo("Robot State Publisher Node Started")
        self.i = 0

    def publish_robot_description(self):
        urdf_path = rospy.get_param('~urdf_path', '/home/tong/Documents/verti_bench/Sim2Real/vw_visual/vertiwheeler.urdf')
        # urdf_path_2 = rospy.get_param('~urdf_path', '/home/aniket/Documents/MPPI_vertiFormer/vertiwheeler_2.urdf')
        if not os.path.isfile(urdf_path):
            rospy.logerr(f"URDF file not found: {urdf_path}")
            return

        for i in range(no_of_robots):
            with open(urdf_path, 'r') as urdf_file:
                urdf_content = urdf_file.read()
            urdf_content = urdf_content.replace(f'base_link', f'base_link_{i}')
            rospy.set_param(self.robot_description_param[i], urdf_content)
            rospy.loginfo(f"Robot description loaded from {urdf_path}_{i}")
            # print(urdf_content)
            # print(self.robot_description_param[i])

    def odom_callback(self, odometry_msg):
        # Extract pose information from odometry message
        position = odometry_msg.pose.pose.position
        orientation = odometry_msg.pose.pose.orientation

        i = self.i
        if i == -1:
            self.i += 1
            return

        # Create a TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.odom_frame_id
        transform.child_frame_id = self.robot_frame_id[i]
    
        # Fill in the transform data
        transform.transform.translation.x = position.x
        transform.transform.translation.y = position.y
        transform.transform.translation.z = position.z
        transform.transform.rotation = orientation

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)
        self.i += 1
        if self.i == no_of_robots:
            self.i = 0

    def path_callback(self, path_msg):
        
        i = self.i
        if i == -1:
            self.i += 1
            return
        
        start_pose = path_msg.poses[-1]            
        ## first Pose
        # Create a TransformStamped message
        position = start_pose.pose.position
        orientation = start_pose.pose.orientation

        # Create a TransformStamped message
        transform = TransformStamped()
        transform.header.stamp = rospy.Time.now()
        transform.header.frame_id = self.odom_frame_id
        transform.child_frame_id = self.robot_frame_id[i]
    
        # Fill in the transform data
        transform.transform.translation.x = position.x
        transform.transform.translation.y = position.y
        transform.transform.translation.z = position.z
        transform.transform.rotation = orientation

        # Broadcast the transform
        self.tf_broadcaster.sendTransform(transform)
        self.i += 1
        if self.i == no_of_robots:
            self.i = 0
    

if __name__ == '__main__':
    try:
        # Create the Robot State Publisher instance
        robot_state_publisher = RobotStatePublisher()

        # Keep the node running
        rospy.spin()
    except rospy.ROSInterruptException:
        rospy.loginfo("Robot State Publisher Node Terminated")
