import rosbag
import numpy as np
import math

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
from scipy.interpolate import griddata
import os

bagFile = os.path.dirname(os.path.realpath(
                    __file__)) + "/../data/terrain_bitmaps/OriginalMaps/elev_map_new.bag"

bag = rosbag.Bag(bagFile)
# Path to the output file
output_file_path = 'elevation_data.txt'
# Desired topic
topic_name = 'elevation_mapping/elevation_map_raw'

topics = bag.get_type_and_topic_info()[1].keys()
for topic in topics:
    print(topic)

# with open(output_file_path, 'w') as output:
#     for topic, msg, t in bag.read_messages(topics=[topic_name]):
#         data = str(msg)
#         output.write(data + '\n')
# print("Data extraction complete.")

# Initialize a variable to hold the last message's data
last_msg_data = None
for topic, msg, t in bag.read_messages(topics=[topic_name]):
    Seq = msg.info.header.seq
    # print(Seq)
    last_msg_data = list(msg.data[0].data)

bag.close()

if last_msg_data is not None:
    elevation_data = np.array(last_msg_data).reshape(360, 360)

    # To find the minimum and maximum values while ignoring NaN values
    min_value = np.nanmin(elevation_data)
    max_value = np.nanmax(elevation_data)

    print(f"The minimum elevation value is: {min_value}")
    print(f"The maximum elevation value is: {max_value}")

    # Linear
    # Create a grid of x, y coordinates
    x = np.arange(0, elevation_data.shape[1])
    y = np.arange(0, elevation_data.shape[0])
    x, y = np.meshgrid(x, y)

    # Mask for valid (non-NaN) values
    mask = ~np.isnan(elevation_data)

    # Coordinates and values of non-NaN points
    x_non_nan = x[mask]
    y_non_nan = y[mask]
    z_non_nan = elevation_data[mask]

    # Perform linear interpolation
    elevation_data_filled = griddata((x_non_nan, y_non_nan), z_non_nan, (x, y), method='linear')

    # Normalize the filled data for image creation
    image_data = (elevation_data_filled - np.nanmin(elevation_data_filled)) / (np.nanmax(elevation_data_filled) - np.nanmin(elevation_data_filled)) * 255
    # img = Image.fromarray(image_data.astype(np.uint8))
    # img.save("elevation_data_linear2.bmp")

    # Plotting the filled elevation data
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(x, y, image_data, cmap='gray', alpha=1)
    ax.plot_wireframe(x, y, image_data, color='gray', linewidth=0.05)
    ax.set_xlabel('X Coordinate')
    ax.set_ylabel('Y Coordinate')
    ax.set_zlabel('Elevation')
    ax.set_title('3D Elevation Map with Linear Interpolation')
    plt.show()
    