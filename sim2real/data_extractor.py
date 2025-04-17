import os
import rosbag
import numpy as np
import utils as Utils
import pickle
from Grid import MapProcessor

bag_path = "/home/tong/Documents/verti_bench/Sim2Real/RL-RTW/ER"
pickle_dir = "/home/tong/Documents/verti_bench/Sim2Real/RL-RTW/ER"
pickle_file_name = "/home/tong/Documents/verti_bench/Sim2Real/RL-RTW/ER/data_ER.pickle"

# Topics to extract (with and without leading slash)
topic_options = {
    't1': ['/cmd_vel1', 'cmd_vel1'],
    't2': ['/rpmVal_data', 'rpmVal_data'],
    't3': ['/elevation_mapping/elevation_map_raw', 'elevation_mapping/elevation_map_raw'],
    't4': ['/natnet_ros/crawler/odom', '/dlio/odom_node/odom', 'rtabmap/odom'],
    't5': ['/depth_to_rgb/image_raw/compressed', 'depth_to_rgb/image_raw/compressed'],
    't6': ['/rgb/image_raw/compressed', 'rgb/image_raw/compressed'],
    't7': ['/imu', 'imu']
}

def find_bag_files(directory):
    """Recursively finds all .bag files in a directory and its subdirectories."""
    bag_files = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith(".bag"):
                bag_files.append(os.path.join(root, file))
    return bag_files

def get_valid_topic(bag, topic_options_list):
    available_topics = bag.get_type_and_topic_info().topics.keys()
    # Print available topics for debugging
    print(f"Available topics: {available_topics}")
    
    for topic_option in topic_options_list:
        if topic_option in available_topics:
            print(f"Found valid topic: {topic_option}")
            return topic_option
    
    return None

if __name__ == "__main__":
    bags_files = find_bag_files(bag_path)
    bags_files.sort()

    faulty_bags = []

    if os.path.exists(pickle_file_name):
        with open(pickle_file_name, "rb") as f:
            bag_data = pickle.load(f)
            if 'total_records' not in bag_data:
                bag_data['total_records'] = []
            if 'bag_name' not in bag_data:
                bag_data['bag_name'] = []
            if 'data' not in bag_data:
                bag_data['data'] = []
            total_records = sum(bag_data['total_records'])
            print("File exists, loading data from pickle and appending to it")
    else:
        bag_data = {'bag_name': [], 'data': [], 'total_records': []}
        total_records = 0

    for f in bags_files:
        extracted_data = {'cmd_vel': [], 'elevation_map': [], 'footprint': [], 'pose': [], 'motor_speed': [], 'dt': [], 'map_offset': []}
        print('Loading from bag:', f)
        bag = rosbag.Bag(f)

        # Determine the correct topic versions for this bag
        t1 = get_valid_topic(bag, topic_options['t1'])
        t2 = get_valid_topic(bag, topic_options['t2'])
        t3 = get_valid_topic(bag, topic_options['t3'])
        t4 = get_valid_topic(bag, topic_options['t4'])
        t5 = get_valid_topic(bag, topic_options['t5'])
        t6 = get_valid_topic(bag, topic_options['t6'])
        t7 = get_valid_topic(bag, topic_options['t7'])

        # Modify this condition to make t2 (rpm data) optional
        if not t1 or not t3 or not t4:  # Removed t2 from this check
            print(f'No valid essential topics found in {f}')
            if not t1: print("  Missing: cmd_vel1")
            if not t3: print("  Missing: elevation_mapping/elevation_map_raw")
            if not t4: print("  Missing: odom topic") 
            faulty_bags.append(f)
            continue

        # Collect only the topics that are valid
        topics_with_valid_names = [t for t in [t1, t2, t3, t4] if t is not None]

        # Initialize variables for data holding
        cmd_throttle, cmd_steering = [], []
        rpm = 0
        robot_pose = None
        last_t = None
        msg_count = 0
        map_count = 0
        mp = MapProcessor()
        map_pose = []
        # fmap = f"elev_map/{os.path.basename(f).split('.', 1)[0]}_{map_count}.npy"
        
        # Create required directories if they don't exist
        os.makedirs(f"{pickle_dir}/elev_map_check", exist_ok=True)
        os.makedirs(f"{pickle_dir}/footprint_check", exist_ok=True)
        
        for topic, msg, t in bag.read_messages(topics=topics_with_valid_names):  # type: ignore
            if topic == t1:
                cmd_throttle.append(msg.data[1])
                cmd_steering.append(msg.data[0])

            elif topic == t2 and t2 is not None:  # Check if t2 exists before using it
                rpm = msg.data[4]

            elif topic == t3:
            # if topic == t3:
                mp.update_map(msg)
                # fmap = f"elev_map/{os.path.basename(f).split('.', 1)[0]}_{map_count}.npy"
                fmap = f"{pickle_dir}/elev_map_check/{os.path.basename(f).split('.', 1)[0]}_{map_count}.npy"
                map_count += 1
                np.save(fmap, mp.map_elevation)
                map_pose = [mp.map_pose.position.x, mp.map_pose.position.y]

            elif topic == t4:
                timenow = msg.header.stamp
                x, y, z = msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z
                roll, pitch, yaw = Utils.quaternion_to_angle(msg.pose.pose.orientation)

                if last_t is None:
                    last_t = timenow.to_sec()
                    continue

                if robot_pose is None:
                    robot_pose = [x, y, z, roll, pitch, yaw]
                    continue

                if mp.map_elevation is None or len(cmd_throttle) == 0:
                    continue

                dt = timenow.to_sec() - last_t
                if dt < 0.1:
                    continue

                diff = np.abs(np.array([x, y, z], dtype=np.float32) - np.array(robot_pose, dtype=np.float32)[:3])
                if np.max(diff) < 0.000001 or np.max(diff) > 0.5:
                    print(f"Invalid data in {f}: diff={diff}")
                    continue

                robot_pose = [x, y, z, roll, pitch, yaw]
                footprint = mp.get_elev_footprint(robot_pose, (40, 100))

                # ffoot = f"footprint/{os.path.basename(f).split('.', 1)[0]}_{msg_count}.npy"
                ffoot = f"{pickle_dir}/footprint_check/{os.path.basename(f).split('.', 1)[0]}_{msg_count}.npy"


                # Save the datapoint
                extracted_data['cmd_vel'].append([np.mean(cmd_throttle), np.mean(cmd_steering)])
                extracted_data['elevation_map'].append(fmap)
                extracted_data['footprint'].append(ffoot)
                extracted_data['pose'].append(robot_pose)
                extracted_data['motor_speed'].append(rpm)
                extracted_data['dt'].append(dt)
                map_offset = [map_pose[0] - x, map_pose[1] - y]
                extracted_data['map_offset'].append(map_offset)

                np.save(ffoot, footprint)

                # Reset variables if needed
                last_t = timenow.to_sec()
                msg_count += 1
                cmd_throttle, cmd_steering = [], []

                if msg_count % 50 == 0:
                    print('.')

        bag.close()
        if msg_count == 0:
            print(f'No data found in {f}')
            faulty_bags.append(f)
            continue

        print(f'Imported {msg_count} data points from', f)
        bag_data['bag_name'].append(f)
        bag_data['data'].append(extracted_data)
        bag_data['total_records'].append(msg_count)
        total_records += msg_count
        with open(pickle_file_name, "wb") as f:
            pickle.dump(bag_data, f)

    print(f"Total records: {total_records}")
    with open(f"{pickle_dir}/faulty_bags.txt", "a") as f:
        f.write("\n".join(faulty_bags))