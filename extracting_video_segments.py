import os
import cv2
import pandas as pd

def extract_segment_frames(video_name, start_time, end_time, fps=5):
    """
    Extract frames from a video segment between start_time and end_time.
    
    Args:
        video_path (str): path to the video file (.mp4)
        start_time (float): start time minutes:seconds 
        end_time (float): end time in minutes:seconds
        fps (int): number of frames per second to extract
    """
    # extract seconds from minutes:seconds
    start_time_seconds = int(start_time.split(':')[0]) * 60 + int(start_time.split(':')[1])
    end_time_seconds = int(end_time.split(':')[0]) * 60 + int(end_time.split(':')[1])

    video_path = "data/vids/" + video_name
    output_dir = f"data/frames/{video_name}/{start_time}_{end_time}"
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)

    start_frame = int(start_time_seconds * video_fps)
    end_frame = int(end_time_seconds * video_fps)
    interval = int(video_fps // fps)

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    count = start_frame
    saved = 0
    while count <= end_frame:
        ret, frame = cap.read()
        if not ret:
            break
        if (count - start_frame) % interval == 0:
            frame_path = os.path.join(output_dir, f"frame_{saved:05d}.jpg")
            cv2.imwrite(frame_path, frame)
            saved += 1
        count += 1

    cap.release()
    print(f"Saved {saved} frames to {output_dir}")


def extract_frames_given_csv(video_name, csv_path, fps=5):
    """
    Extract frames from video segments specified in a CSV file.
    
    Args:
        csv_path (str): path to the CSV file containing video segments
        fps (int): number of frames per second to extract
    """

    df = pd.read_csv(csv_path)
    for index, row in df.iterrows():
        start_time = row['start_time']
        end_time = row['end_time']
        extract_segment_frames(video_name, start_time, end_time, fps)


if __name__ == "__main__":
    # extract_segment_frames(video_name="Double_Game_Point_Carleton_vs._Stanford_Women's.mp4", 
    #                         start_time="0:05", 
    #                         end_time="0:13", 
    #                         fps=5)
    extract_frames_given_csv(video_name="Double_Game_Point_Carleton_vs._Stanford_Women's.mp4", 
                             csv_path="data/segment_start_end_times/Double_Game_Point_Carleton_vs._Stanford_Women's.csv", 
                             fps=5)