

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
import pandas as pd
import chardet

frisbee_coordinates = None
EVAL_DISTANCE_ERROR = 10



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

# def find_center_bounding_box(bounding_box):
#     print(f"Bounding box: {bounding_box}")
#     # example bounding box: [[119, 240, 144, 251]]
#     x_min, y_min, x_max, y_max = bounding_box
#     x_center = (x_min + x_max) / 2
#     y_center = (y_min + y_max) / 2
#     return [x_center, y_center]

def find_center_bounding_box(bounding_box):
    """Find the center (x, y) of a bounding box."""
    x_min = bounding_box.xmin
    y_min = bounding_box.ymin
    x_max = bounding_box.xmax
    y_max = bounding_box.ymax
    x_center = (x_min + x_max) / 2
    y_center = (y_min + y_max) / 2
    return (x_center, y_center)

    

# def run_sam(image, viz=True):
#     if (image.height == 0 or image.width == 0):
#         print("Image has zero height or zero width, skipping SAM processing.")
#         return None
#     image_array, detections = grounded_segmentation(
#     image=image,
#     labels=["a frisbee"],
#     threshold=0.3,
#     polygon_refinement=True, 
#     detector_id = "IDEA-Research/grounding-dino-tiny", 
#     segmenter_id = "facebook/sam-vit-base"
#     )
    
#     if viz:
#         print("Visualizing detections...")
#         plot_detections(image_array, detections)
#     # currently assuming only one bounding box. if run into case with multiple bounding boxes, choose most likely one.
#     boxes, probabilities = get_boxes_and_probabilities(detections)
#     if len(boxes)== 0:
#         return None
#     box = boxes[0][0]  # First [0] gets rid of extra wrapping bracket
#     return find_center_bounding_box(box)

def run_sam(image: Image.Image, viz: bool = True):
    """
    Run grounded segmentation (GroundingDINO + SAM) on a given image.

    Args:
        image (PIL.Image): The input image to segment.
        viz (bool): Whether to visualize the detections.

    Returns:
        center (tuple or None): Center of the detected bounding box (x, y).
        mask (np.ndarray or None): Segmentation mask (same size as input image).
    """
    if image.height == 0 or image.width == 0:
        print("Image has zero height or zero width, skipping SAM processing.")
        return None, None

    image_array, detections = grounded_segmentation(
        image=image,
        labels=["a frisbee"],
        threshold=0.3,
        polygon_refinement=True, 
        detector_id="IDEA-Research/grounding-dino-tiny", 
        segmenter_id="facebook/sam-vit-base"
    )

    if detections is None or len(detections) == 0:
        print("No detections found.")
        return None, None

    # if viz:
    #     print("Visualizing detections...")
    #     plot_detections(image_array, detections)

    # Assume first detection is the most confident
    detection = detections[0]
    
    # Extract bounding box and mask
    box = detection.box
    mask = detection.mask

    if mask is None:
        print("Detection found but mask is missing.")
        return None, None

    # Find center of bounding box
    center = find_center_bounding_box(box)

    return center, mask

def run_autogressive_sam_baseline_step(pass_folder, crop_size, j, display=True):
    image_path = f"{pass_folder}/frame_{j:05d}.jpg"
    image_rgb = load_image_and_convert_rgb(image_path)
    cropped_image, (crop_x_min, crop_y_min) = crop_image(image_rgb, crop_size, frisbee_coordinates)

    if display:
        plt.imshow(cropped_image)
        plt.title(f"Cropped input to SAM - Frame {j:05d}")
        plt.axis('off')
        plt.show()

    cropped_image_PIL = Image.fromarray(cropped_image)
    center_in_crop = run_sam(cropped_image_PIL, viz=display)

    if (center_in_crop is None):
        print(f"Frame {j:05d} — no frisbee detected, skipping to next frame, assume occlusion or terrible angle")
        return None

    # Map to original image coordinates
    center_in_original = (
        int(center_in_crop[0] + crop_x_min),
        int(center_in_crop[1] + crop_y_min)
    )
    return center_in_original
    
# def run_autogressive_sam_resnet_step(pass_folder, crop_size, j, model, display=True):
#     image_path = f"{pass_folder}/frame_{j:05d}.jpg"
#     image_rgb = load_image_and_convert_rgb(image_path)
#     cropped_image, (crop_x_min, crop_y_min) = crop_image(image_rgb, crop_size, frisbee_coordinates)

#     model.eval()
#     # Convert cropped image to tensor
#     # check if cuda aavailable
#     if torch.cuda.is_available():
#         device = 'cuda'
#     else:
#         device = 'cpu'
#     cropped_image_tensor = transforms.ToTensor()(cropped_image).unsqueeze(0).to(device)
#     # Forward pass through the model
#     with torch.no_grad():
#         localized_output = model(cropped_image_tensor).cpu().numpy() * crop_size
#         print(f"Localized output for frame {j:05d}: {localized_output[0]}")
#         if display:
#             # plot this on cropped image
#             plt.imshow(cropped_image)
#             plt.scatter(localized_output[0][0], localized_output[0][1], color='red', label='Localized Output')
#             plt.title(f"Localized Output - Frame {j:05d}")
#             plt.axis('off')
#             plt.legend()
#             plt.show()
    
#     # RECROP to 80 by 80 around localized_output
#     if crop_size == 250:
#         recrop_size = 80
#     elif crop_size == 500:
#         recrop_size = 150
#     elif crop_size == 750:
#         recrop_size = 230
#     recropped_image, (recrop_x_min, recrop_y_min) = crop_image(cropped_image, recrop_size,  (int(localized_output[0][0]), int(localized_output[0][1])))

#     recropped_image_PIL = Image.fromarray(recropped_image)
#     center_in_crop = run_sam(recropped_image_PIL, viz=display)

#     if (center_in_crop is None):
#         print(f"Frame {j:05d} — no frisbee detected, skipping to next frame, assume occlusion or terrible angle")
#         return None

#     # undo the recrop
#     center_in_crop = (
#         center_in_crop[0] + recrop_x_min,
#         center_in_crop[1] + recrop_y_min
#     )

#     # Map to original image coordinates
#     center_in_original = (
#         int(center_in_crop[0] + crop_x_min),
#         int(center_in_crop[1] + crop_y_min)
#     )

#     return center_in_original

def run_autogressive_sam_resnet_step(pass_folder, crop_size, j, model, display=True):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    image_path = f"{pass_folder}/frame_{j:05d}.jpg"
    image_rgb = load_image_and_convert_rgb(image_path)  # original full image
    cropped_image, (crop_x_min, crop_y_min) = crop_image(image_rgb, crop_size, frisbee_coordinates)

    model.eval()
    cropped_image_tensor = transforms.ToTensor()(cropped_image).unsqueeze(0).to(device)

    with torch.no_grad():
        localized_output = model(cropped_image_tensor).cpu().numpy() * crop_size
        if localized_output.shape[0] == 0:
            print(f"Frame {j:05d} — model produced no output, skipping")
            return None, None
        # if display:
        #     plt.imshow(cropped_image)
        #     plt.scatter(localized_output[0][0], localized_output[0][1], color='red', label='Localized Output')
        #     plt.title(f"Localized Output - Frame {j:05d}")
        #     plt.axis('off')
        #     plt.legend(loc='upper right')
        #     plt.show()

    recrop_size_map = {250: 80, 500: 150, 750: 230}
    if crop_size not in recrop_size_map:
        raise ValueError(f"Unexpected crop_size: {crop_size}")
    recrop_size = recrop_size_map[crop_size]

    recropped_image, (recrop_x_min, recrop_y_min) = crop_image(
        cropped_image, recrop_size, (int(localized_output[0][0]), int(localized_output[0][1]))
    )
    recropped_image_PIL = Image.fromarray(recropped_image)

    center_in_crop, mask_in_recrop = run_sam(recropped_image_PIL, viz=display)
    if center_in_crop is None or mask_in_recrop is None:
        print(f"Frame {j:05d} — no frisbee detected, skipping to next frame, assume occlusion or terrible angle")
        return None, None

    # Step 1: shift mask back into original cropped image
    h_cropped, w_cropped = cropped_image.shape[:2]
    h_original, w_original = image_rgb.shape[:2]

    full_cropped_mask = np.zeros((h_cropped, w_cropped), dtype=np.uint8)

    mask_h, mask_w = mask_in_recrop.shape
    x_start = recrop_x_min
    y_start = recrop_y_min
    x_end = recrop_x_min + mask_w
    y_end = recrop_y_min + mask_h

    full_cropped_mask[y_start:y_end, x_start:x_end] = mask_in_recrop

    # Step 2: shift full cropped mask back into original full image
    full_original_mask = np.zeros((h_original, w_original), dtype=np.uint8)

    x_start_orig = crop_x_min
    y_start_orig = crop_y_min
    x_end_orig = crop_x_min + w_cropped
    y_end_orig = crop_y_min + h_cropped

    full_original_mask[y_start_orig:y_end_orig, x_start_orig:x_end_orig] = full_cropped_mask

    # Map center back too
    center_in_crop = (
        center_in_crop[0] + recrop_x_min,
        center_in_crop[1] + recrop_y_min
    )
    center_in_original = (
        int(round(center_in_crop[0] + crop_x_min)),
        int(round(center_in_crop[1] + crop_y_min))
    )

    return center_in_original, full_original_mask

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
        center_in_original = run_autogressive_sam_baseline_step(pass_folder, crop_size, j,  display=True)
        if center_in_original is None:
            continue
        else:
            frisbee_coordinates = center_in_original
        print(f"Frame {j:05d} — updated frisbee coordinates: {frisbee_coordinates}")


def get_label_for_frame(frame_path, labels_csv):
    labels_df = pd.read_csv(labels_csv)

    # Normalize the frame paths in CSV
    labels_df['filename'] = labels_df['filename'].apply(lambda x: x.replace("\\", "/").strip())

    # Normalize input frame_path
    frame_path = frame_path.replace("\\", "/").replace(":", "_").strip()
    
    parts = frame_path.split("/")
    pass_name = parts[-2]  # e.g., 1_11_1_16
    frame_file = parts[-1]  # e.g., frame_00000.jpg

    # Now use forward slashes
    lookup_name = "./data/frames/Double_Game_Point_Carleton_vs._Stanford_Women's.mp4/" + pass_name + frame_file

    lookup_name = lookup_name.strip()  # strip any extra spaces

    # Find match
    match = labels_df[labels_df['filename'] == lookup_name]

    if match.empty:
        raise ValueError(f"No label found for frame {frame_path} (lookup `{lookup_name}`)")
    
    x = match.iloc[0]['x_coord']
    y = match.iloc[0]['y_coord']
    return x, y

from tqdm import tqdm

def run_eval(root_dir, labels_csv, crop_size, method, grounded=False, display=False):
    correct = 0
    total = 0

    # pass_folders are all the subfolders in root_dir
    pass_folders = [os.path.join(root_dir, f) for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]

    for pass_folder in pass_folders:
        if pass_folder == ".ipynb_checkpoints":
            continue
        # get the label of the first pass:
        first_frame = os.path.join(pass_folder, "frame_00000.jpg")
        print("first frame:", first_frame)

        x, y = get_label_for_frame(first_frame, labels_csv)
        print(f"Initial frisbee coordinates from label: ({x}, {y})")

        # use the x, y to kick off the tracking
        global frisbee_coordinates

        frisbee_coordinates = (x, y)

        num_frames_in_pass = len(os.listdir(pass_folder))

        if method == "autogressive_sam_resnet":
            model = CoordResNet18().to(device="cuda" if torch.cuda.is_available() else "cpu")
            if (crop_size == 250):
                model.load_state_dict(torch.load("data/best_coordresnet18_crop_250.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
                # model.load_state_dict(torch.load("/content/drive/MyDrive/cs231n/projects/data/best_coordresnet18_crop_250.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
            elif( crop_size == 500):
                model.load_state_dict(torch.load("data/best_coordresnet18_crop_500.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
                # model.load_state_dict(torch.load("/content/drive/MyDrive/cs231n/projects/data/best_coordresnet18_crop_500.pth",  map_location="cuda" if torch.cuda.is_available() else "cpu"))
            elif( crop_size == 750):
                # model.load_state_dict(torch.load("data/best_coordresnet18_crop_500.pth", map_location="cuda" if torch.cuda.is_available() else "cpu"))
                model.load_state_dict(torch.load("/content/drive/MyDrive/cs231n/projects/data/best_coordresnet18_crop_750.pth",  map_location="cuda" if torch.cuda.is_available() else "cpu"))
            
        for j in range(1, num_frames_in_pass):
            if method == "autogressive_sam_baseline":
                center_coordinates = run_autogressive_sam_baseline_step(pass_folder, crop_size, j, display=display)
            elif method == "autogressive_sam_resnet":
                center_coordinates, mask = run_autogressive_sam_resnet_step(pass_folder, crop_size, j, model, display=display)
                if display:
                    if center_coordinates is not None:
                        print("HERE")
                        # display the mask on the image
                        image_path = os.path.join(pass_folder, f"frame_{j:05d}.jpg")
                        image_rgb = load_image_and_convert_rgb(image_path)
                        plt.imshow(image_rgb)
                        plt.imshow(mask, alpha=0.5, cmap='jet')  # Overlay mask with transparency
                        plt.title(f"Mask Overlay - Frame {j:05d}")
                        plt.axis('off')
                        pass_folder_name = os.path.basename(pass_folder)
                        os.makedirs(f"data/gif/{pass_folder_name}", exist_ok=True)
                        plt.savefig(f"data/gif/{pass_folder_name}/frame_{j:05d}.png", bbox_inches='tight', pad_inches=0, dpi=100)
                        #also save the original image
                        plt.imsave(f"data/gif/{pass_folder_name}/original_frame_{j:05d}.png", image_rgb)
                    else:
                        # visualize just the image part
                        image_path = os.path.join(pass_folder, f"frame_{j:05d}.jpg")
                        image_rgb = load_image_and_convert_rgb(image_path)
                        plt.imshow(image_rgb)
                        plt.title(f"No Mask Detected - Frame {j:05d}")
                        plt.axis('off')
                        
                        pass_folder_name = os.path.basename(pass_folder)
                        os.makedirs(f"data/gif/{pass_folder_name}", exist_ok=True)
                        plt.savefig(f"data/gif/{pass_folder_name}/frame_{j:05d}.png", bbox_inches='tight', pad_inches=0, dpi=100)
                        plt.imsave(f"data/gif/{pass_folder_name}/original_frame_{j:05d}.png", image_rgb)
                    
            if center_coordinates is None:
                if not grounded:
                    print(f"Frame {j:05d} — no frisbee detected, skipping to next frame, assume occlusion or terrible angle")
                    continue
                else:
                # run sam on the full image if you can't find in crop
                    print(f"Frame {j:05d} — no frisbee detected in crop, running SAM on full image.")
                    image_path = os.path.join(pass_folder, f"frame_{j:05d}.jpg")
                    image_rgb = load_image_and_convert_rgb(image_path)
                    center_coordinates, mask = run_sam(Image.fromarray(image_rgb), viz=display)
                    if display:
                        if center_coordinates is not None:
                            # display the mask on the image
                            image_path = os.path.join(pass_folder, f"frame_{j:05d}.jpg")
                            image_rgb = load_image_and_convert_rgb(image_path)
                            plt.imshow(image_rgb)
                            plt.imshow(mask, alpha=0.5, cmap='jet')  # Overlay mask with transparency
                            plt.title(f"Mask Overlay - Frame {j:05d}")
                            plt.axis('off')
                            # find the pass name
                            pass_folder_name = os.path.basename(pass_folder)
                            os.makedirs(f"data/gif/{pass_folder_name}", exist_ok=True)
                            plt.savefig(f"data/gif/{pass_folder_name}/frame_{j:05d}.png", bbox_inches='tight', pad_inches=0, dpi=100)
                            plt.imsave(f"data/gif/{pass_folder_name}/original_frame_{j:05d}.png", image_rgb)
                           
        
                        else:
                            # visualize just the image part
                            image_path = os.path.join(pass_folder, f"frame_{j:05d}.jpg")
                            image_rgb = load_image_and_convert_rgb(image_path)
                            pass_folder_name = os.path.basename(pass_folder)
                            # make a directory for the pass if it doesn't exist\
                            os.makedirs(f"data/gif/{pass_folder_name}", exist_ok=True)
                            plt.imshow(image_rgb)
                            plt.title(f"No Mask Detected - Frame {j:05d}")
                            plt.axis('off')
                            plt.savefig(f"data/gif/{pass_folder_name}/frame_{j:05d}.png", bbox_inches='tight', pad_inches=0, dpi=100)
                            plt.imsave(f"data/gif/{pass_folder_name}/original_frame_{j:05d}.png", image_rgb)

                    if center_coordinates is None:    
                        print(f"Frame {j:05d} — no frisbee detected in full image, skipping to next frame.")
                        continue
                    else:
                        # make sure they are ints
                        frisbee_coordinates = (int(center_coordinates[0]), int(center_coordinates[1]))
                    
            else:
                frisbee_coordinates = center_coordinates

          
            # get the label for the current frame
            frame_path = os.path.join(pass_folder, f"frame_{j:05d}.jpg")
            x_label, y_label = get_label_for_frame(frame_path, labels_csv)
            print(f"Frame {j:05d} — label coordinates: ({x_label}, {y_label})")
            print(f"Frame {j:05d} — detected coordinates: {frisbee_coordinates}")

            print("frisbee_coordinates:", frisbee_coordinates)



            if abs(frisbee_coordinates[0] - x_label) <= EVAL_DISTANCE_ERROR and abs(frisbee_coordinates[1] - y_label) <= EVAL_DISTANCE_ERROR:
                correct += 1
    

          
            total  += 1
    accuracy = correct / total if total > 0 else 0.0
    print(f"\nEvaluated {total} frames.")
    print(f"Correct: {correct} — Missed: {total - correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return {
        "accuracy": accuracy
    }

             
import pandas as pd
from tqdm import tqdm
import re

def clean_path(path):
        """
        Clean up the path to format: 2:22_2:28/frame_00018.jpg
        """
        # Extract the part after 'data\\frames\\...\\'
        match = re.search(r'data\\frames\\[^\\]+\\(.+)', path)
        if match:
            sub_path = match.group(1)
        else:
            sub_path = path  # fallback if regex fails

        frame_idx = sub_path.find('frame_')
        prefix = sub_path[:frame_idx]  # e.g., '2_22_2_28'
        filename = sub_path[frame_idx:]  # e.g., 'frame_00018.jpg'

        # Now, split prefix into two groups (start_time and end_time)
        times = prefix.strip('_').split('_')
        if len(times) >= 4:
            start_time = f"{times[0]}:{times[1]}"  # e.g., 2:22
            end_time = f"{times[2]}:{times[3]}"    # e.g., 2:28
            prefix_clean = f"{start_time}_{end_time}"
        else:
            # fallback if somehow format is wrong
            prefix_clean = prefix.replace('_', ':')

        return f"{prefix_clean}/{filename}"

def is_point_in_box(point, box):
    x, y = point
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max

def evaluate_sam_hit_miss(pass_folder, ground_truth_csv, root_dir, display=False):
    df = pd.read_csv(ground_truth_csv)
    correct = 0
    total = 0
    missed_frames = []


    for _, row in tqdm(df.iterrows(), total=len(df), desc="Evaluating frames"):
        frame_id = row["filename"]
        frame_id = clean_path(frame_id)
        frame_id = os.path.join(root_dir, frame_id)
        gt_center = (float(row["x_coord"]), float(row["y_coord"]))
        image_path = frame_id

        if not os.path.exists(image_path):
            print(f"Warning: Frame {frame_id} not found.")
            continue

        image_rgb = load_image_and_convert_rgb(image_path)
        image_pil = Image.fromarray(image_rgb)

        image_array, detections = grounded_segmentation(
            image=image_pil,
            labels=["a frisbee"],
            threshold=0.3,
            polygon_refinement=True,
            detector_id="IDEA-Research/grounding-dino-tiny",
            segmenter_id="facebook/sam-vit-base"
        )
        if display:
            plot_detections(image_array, detections)

        boxes, _ = get_boxes_and_probabilities(detections)
        if len(boxes) == 0:
            missed_frames.append(frame_id)
            total += 1
            continue

        predicted_box = boxes[0][0]  # assuming only one box
        # if is_point_in_box(gt_center, predicted_box):
        #     correct += 1
        # else:
        #     missed_frames.append(frame_id)
        # total += 1

        predicted_center = find_center_bounding_box(predicted_box)
        if (abs(predicted_center[0] - gt_center[0]) <= EVAL_DISTANCE_ERROR and
            abs(predicted_center[1] - gt_center[1]) <= EVAL_DISTANCE_ERROR):
            correct += 1
        else:
            missed_frames.append(frame_id)
        total += 1

    accuracy = correct / total if total > 0 else 0.0
    print(f"\nEvaluated {total} frames.")
    print(f"Correct: {correct} — Missed: {total - correct}")
    print(f"Accuracy: {accuracy * 100:.2f}%")

    return {
        "accuracy": accuracy,
        "missed_frames": missed_frames
    }











  


if __name__ == "__main__":
    # parse argument passed into main
    import argparse
    parser = argparse.ArgumentParser(description="Run SAM tracking on a video pass.")
    # argument is the name of the method
    parser.add_argument("--method", type=str, choices=["autogressive_sam_baseline", "autogressive_sam_resnet", "baseline"], required=True, help="Method to use for tracking.")
    parser.add_argument("--crop_size", type=int, default=250, help="Crop size to use for tracking. Default is 250.")
    parser.add_argument("--grounded", type=bool, default=True, help="use baseline as fallback if grounded.")

    frisbee_coordinates = None

    # root_dir = '/content/drive/MyDrive/cs231n/projects/frames_for_CNN/Double_Game_Point_Carleton_vs._Stanford_Women\'s.mp4'
    root_dir = 'data/frames/Double_Game_Point_Carleton_vs._Stanford_Women\'s.mp4'
    # FOR COLLAB
    # pass_folder = '/content/drive/MyDrive/cs231n/projects/frames_for_CNN/Double_Game_Point_Carleton_vs._Stanford_Women\'s.mp4'
    # FOR LOCAL TESTING
    pass_folder = 'data/frames/Double_Game_Point_Carleton_vs._Stanford_Women\'s.mp4'
    # FOR COLLAB
    # ground_truth_csv = '/content/drive/MyDrive/cs231n/projects/frames_for_CNN/Double_Game_Point_Carleton_vs._Stanford_Women\'s.mp4/centered.csv'
    # FOR LOCAL TESTING
    ground_truth_csv = 'data/centered_frames/Double_Game_Point_Carleton_vs._Stanford_Women\'s.mp4/centered.csv'

    method = parser.parse_args().method
    crop_size = parser.parse_args().crop_size
    grounded = parser.parse_args().grounded
    if method == "autogressive_sam_baseline" or method == "autogressive_sam_resnet":
        run_eval(pass_folder, ground_truth_csv, crop_size, method=method, grounded=grounded, display=True)
    elif method == "baseline":
        evaluate_sam_hit_miss(pass_folder, ground_truth_csv, root_dir, display=True)
    else:
        # raise error
        raise ValueError(f"Unknown method: {method}. Choose from 'autogressive_sam_baseline', 'autogressive_sam_resnet', or 'baseline'.")





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