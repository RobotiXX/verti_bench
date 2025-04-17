import numpy as np
import cv2 as cv
from PIL import Image, ImageFilter
import matplotlib.pyplot as plt
from threading import Lock
import pychrono as chrono
import os

def load_image(file_path):
    return cv.imread(file_path, cv.IMREAD_GRAYSCALE)

def save_image(image_array, file_name):
    cv.imwrite(file_name, image_array)

def generate_intermediate_images(start_img, end_img, num_stages, folder):
    # Generate and save intermediate images
    for i in range(num_stages):
        # Calculate the interpolation weight
        alpha = i / (num_stages - 1)
        # Linearly interpolate between start and end images
        blended_image = cv.addWeighted(start_img, 1 - alpha, end_img, alpha, 0)
        # Save the resulting image
        save_image(blended_image, os.path.join(folder, f'level_{i+1}.bmp'))

train_folder = os.path.dirname(os.path.realpath(__file__)) + "/../data/terrain_bitmaps/TrainLevels"
test_folder = os.path.dirname(os.path.realpath(__file__)) + "/../data/terrain_bitmaps/TestLevels"
# Create directories if they don't exist
os.makedirs(train_folder, exist_ok=True)
os.makedirs(test_folder, exist_ok=True)

# Load the original terrain image
flat_terrain = np.zeros((150, 150), dtype=np.uint8)
rugged_terrain = load_image(os.path.dirname(os.path.realpath(
                        __file__)) + "/../data/terrain_bitmaps/bumpy_coarse_3x3.bmp")


# Generate and save the transition stages for training and testing
generate_intermediate_images(flat_terrain, rugged_terrain, num_stages=100, folder=train_folder)
generate_intermediate_images(flat_terrain, rugged_terrain, num_stages=10, folder=test_folder)
