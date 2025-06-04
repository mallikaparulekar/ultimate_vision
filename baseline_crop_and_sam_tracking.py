

frames_path = "data/frames/Double_Game_Point_Carleton_vs._Stanford_Women's.mp4/0_15_0_17"

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sam import *
from PIL import Image
import os
import torch
from resnet import CoordResNet18
from torchvision import transforms

frisbee_coordinates = None



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
    return image[y1:y2, x1:x2], (x1, y1)

def load_image_and_convert_rgb(path):
    # Read the image
    image = cv2.imread(path)
    if image is None:
        raise FileNotFoundError(f"Image not found: {path}")
    # Convert to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image_rgb

def find_center_bounding_box(bounding_box):
    # example bounding box: [[119, 240, 144, 251]]
    x_min, y_min, x_max, y_max = bounding_box
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return [x_center, y_center]

    

def run_sam(image):
    image_array, detections = grounded_segmentation(
    image=image,
    labels=["a frisbee"],
    threshold=0.3,
    polygon_refinement=True, 
    detector_id = "IDEA-Research/grounding-dino-tiny", 
    segmenter_id = "facebook/sam-vit-base"
    )
    
    plot_detections(image_array, detections)
    # currently assuming only one bounding box. if run into case with multiple bounding boxes, choose most likely one.
    boxes, probabilities = get_boxes_and_probabilities(detections)
    if len(boxes)== 0:
        return None
    box = boxes[0][0]  # First [0] gets rid of extra wrapping bracket
    return find_center_bounding_box(box)

def run_autoregresive_sam_baseline(pass_folder, crop_size):

    global frisbee_coordinates

    num_frames_in_pass = len(os.listdir(pass_folder))

    for i in range(num_frames_in_pass):
        image_path = f"{pass_folder}/frame_{i:05d}.jpg"
        image_rgb = load_image_and_convert_rgb(image_path)

        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.imshow(image_rgb)
        plt.title(f"Click to select frisbee - Frame {i:05d}")
        plt.axis('off')
        plt.show()

        if frisbee_coordinates is not None:
            print(f"Initial frisbee coordinates set from frame {i:05d}: {frisbee_coordinates}")
            break
    else:
        raise RuntimeError("No frisbee location selected after iterating through all frames.")

    # Now track the frisbee in all subsequent frames 
    for j in range(i + 1, num_frames_in_pass):
        image_path = f"{pass_folder}/frame_{j:05d}.jpg"
        image_rgb = load_image_and_convert_rgb(image_path)
        cropped_image, (crop_x_min, crop_y_min) = crop_image(image_rgb, crop_size, frisbee_coordinates)

        # Optional display
        plt.imshow(cropped_image)
        plt.title(f"Cropped input to SAM - Frame {j:05d}")
        plt.axis('off')
        plt.show()

        cropped_image_PIL = Image.fromarray(cropped_image)
        center_in_crop = run_sam(cropped_image_PIL)

        if (center_in_crop is None):
            print(f"Frame {j:05d} — no frisbee detected, skipping to next frame, assume occlusion or terrible angle")
            continue

        # Map to original image coordinates
        center_in_original = (
            int(center_in_crop[0] + crop_x_min),
            int(center_in_crop[1] + crop_y_min)
        )
        frisbee_coordinates = center_in_original
        print(f"Frame {j:05d} — updated frisbee coordinates: {frisbee_coordinates}")




def run_autoregresive_sam_with_resnet(pass_folder, crop_size):

     # load model from .pth
    print("Loading model...")
    model = CoordResNet18().to(device='cpu')
    model.load_state_dict(torch.load("data/best_coordresnet18.pth", map_location='cpu'))
    model.eval()
    print("Model loaded successfully.")
  

    global frisbee_coordinates

    num_frames_in_pass = len(os.listdir(pass_folder))

    for i in range(num_frames_in_pass):
        image_path = f"{pass_folder}/frame_{i:05d}.jpg"
        image_rgb = load_image_and_convert_rgb(image_path)

        fig = plt.figure()
        fig.canvas.mpl_connect('button_press_event', on_click)
        plt.imshow(image_rgb)
        plt.title(f"Click to select frisbee - Frame {i:05d}")
        plt.axis('off')
        plt.show()

        if frisbee_coordinates is not None:
            print(f"Initial frisbee coordinates set from frame {i:05d}: {frisbee_coordinates}")
            break
    else:
        raise RuntimeError("No frisbee location selected after iterating through all frames.")

    # Now track the frisbee in all subsequent frames 
    for j in range(i + 1, num_frames_in_pass):
        image_path = f"{pass_folder}/frame_{j:05d}.jpg"
        image_rgb = load_image_and_convert_rgb(image_path)
        cropped_image, (crop_x_min, crop_y_min) = crop_image(image_rgb, crop_size, frisbee_coordinates)

       
        model.eval()
        # Convert cropped image to tensor
        cropped_image_tensor = transforms.ToTensor()(cropped_image).unsqueeze(0)
        # Forward pass through the model
        with torch.no_grad():
            localized_output = model(cropped_image_tensor).cpu().numpy() * crop_size
        
        # RECROP to 80 by 80 around localized_output
        recropped_image, (recrop_x_min, recrop_y_min) = crop_image(cropped_image, 80,  (int(localized_output[0][0]), int(localized_output[0][1])))


        # Optional display
        # plt.imshow(cropped_image)
        # plt.title(f"Cropped input to SAM - Frame {j:05d}")
        # plt.axis('off')
        # plt.show()

        recropped_image_PIL = Image.fromarray(recropped_image)
        center_in_crop = run_sam(recropped_image_PIL)

        if (center_in_crop is None):
            print(f"Frame {j:05d} — no frisbee detected, skipping to next frame, assume occlusion or terrible angle")
            continue

        # undo the recrop
        center_in_crop = (
            center_in_crop[0] + recrop_x_min,
            center_in_crop[1] + recrop_y_min
        )

        # Map to original image coordinates
        center_in_original = (
            int(center_in_crop[0] + crop_x_min),
            int(center_in_crop[1] + crop_y_min)
        )
        frisbee_coordinates = center_in_original
        print(f"Frame {j:05d} — updated frisbee coordinates: {frisbee_coordinates}")



  

        


if __name__ == "__main__":
    frisbee_coordinates = None
    pass_folder = "data/frames/Double_Game_Point_Carleton_vs._Stanford_Women's.mp4/1:44_1:50/"
    run_autoregresive_sam_with_resnet(pass_folder, crop_size=250)   


# notes: failure cases:
# in 0:05_0:13, the frisbee is detected as a shoe in frame_0005. possible: have human intervention and re-click when the probability is low or bounding box too small,. in this case bounding box was 8 by 8, usually it is 13 by 13. also prob is 0.38, prev probs were >= 0.5
# correct: 4/42
# in 0:15_0:17, the frisbee is obstructed in frame_0002. if no frisbee detected do human intervention.
# correct: 2/3

# in 0:20_0:22: could not detect frisbee in very firts frame

# in 0:22_0:24: not visible to human eye - SKIP THIS

# in 0:29_0:31: in frame 0003 - could not detect frisbee when in players hand


# in 1:11_1:16: frisbee not visible in first frame, adding functionality to click thru frames until visible. now it failed in frame_0006 when the person was occluding it, said no frisbee found
# use body position?? when can't find frisbee detect a person

# 1:28 - 1:30: frisbee in frame 0001 in player hand, can't detect

# in 1:38-1:39: frisbee in frame 0002 is occluded, can't see it

# ** in 1:44-1:50: frisbee in player's hand in frame `0001, can't detect

# ** in 2:20-2:22: seems to work best in overhead view (frisbee looks the roundestt) failed in frame 00010 (very last one where it merges w player), detected no frisbee

# in 2:22-2:28 WORKED ALL THE WAY TILL V LAST FRAME! then for some reason couldnt find it on the ground. crop was great!

# in 2:34-37, clicked in frame 5, got correct crop on frame 6, but could not detect

# 2:45-47 similar to above, clicked frame 00003, got correct crop on frame 00004, but could not detect hmm

# in 2:57-3:01: could not detect in frame 003, even tho right crop! 