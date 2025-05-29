

frames_path = "data/frames/Double_Game_Point_Carleton_vs._Stanford_Women's.mp4/0:15_0:17"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sam import *
from PIL import Image



def display_image(image_path):
    # Read the image
    image = cv2.imread(image_path)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def on_click(event):
    # Get the coordinates of the click
    x, y = int(event.xdata), int(event.ydata)
    print(f"Clicked coordinates: ({x}, {y})")

    # Store the coordinates in a variable
    global frisbee_coordinates
    frisbee_coordinates = (x, y)

    # Close the plot after clicking
    plt.close()


def crop_image(image, crop_size, cordinates):
    x, y = cordinates
    x1 = max(0, x - crop_size // 2)
    y1 = max(0, y - crop_size // 2)
    x2 = min(image.shape[1], x + crop_size // 2)
    y2 = min(image.shape[0], y + crop_size // 2)
    return image[y1:y2, x1:x2]

def load_image_and_convert_rgb(path):
    # Read the image
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def run_sam(image):
    image_array, detections = grounded_segmentation(
    image=image,
    labels=["a frisbee."],
    threshold=0.3,
    polygon_refinement=True, 
    detector_id = "IDEA-Research/grounding-dino-tiny", 
    segmenter_id = "facebook/sam-vit-base"
    )

    plot_detections(image_array, detections)

if __name__ == "__main__":
    # Load image path
    image_path = f"{frames_path}/frame_00004.jpg"
    image_rgb = load_image_and_convert_rgb(image_path)

    # Set up the figure and click event
    fig = plt.figure()
    fig.canvas.mpl_connect('button_press_event', on_click)
    plt.imshow(image_rgb)
    plt.axis('off')
    plt.show()
    print(f"Initial frisbee coordinates: {frisbee_coordinates}")

    next_image_path = f"{frames_path}/frame_00005.jpg"
    # make a crop of 100 by 100 around the frisbee coordinates in the next_image
    next_image_rgb = load_image_and_convert_rgb(next_image_path)
    cropped_image = crop_image(next_image_rgb, 500, frisbee_coordinates)
    
    # Display the cropped image
    plt.imshow(cropped_image)
    plt.axis('off')
    plt.show()

    # Run SAM on the cropped image
    cropped_image_PIL = Image.fromarray(cropped_image)
    run_sam(cropped_image_PIL)

   