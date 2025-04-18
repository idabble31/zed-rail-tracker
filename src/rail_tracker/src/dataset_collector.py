#!/usr/bin/env python3
"""
Dataset Collector with Automatic Split for ZED Camera
------------------------------------------------------
This script captures frames from the ZED camera using the pyzed.sl SDK,
displays them in a window, and saves each frame to a session folder. When
recording stops (press 'q'), the script automatically splits the dataset
into train, val, and test folders based on a configurable ratio.
 
Folder structure after recording:

  dataset/
    raw/
      20230525_154512/              # Raw captured images
        frame_000000.png
        frame_000001.png
        ...
      20230525_154512_split/        # Automatically created after splitting
        train/
        val/
        test/
 
This structure is ready for upload to Roboflow for training a YOLOv5 segmentation model.
"""

import pyzed.sl as sl
import cv2
import os
from datetime import datetime
import random
import shutil

def create_session_folder(base_dir="dataset"):
    # Create a timestamped folder inside dataset/raw/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_folder = os.path.join(base_dir, "raw", timestamp)
    os.makedirs(session_folder, exist_ok=True)
    return session_folder

def split_dataset(session_folder, split_ratio=(0.7, 0.15, 0.15)):
    """
    Splits the images in session_folder into train, val, and test subfolders.
    
    split_ratio: A tuple representing percentages: (train, val, test).
                 Default is 70% train, 15% val, 15% test.
    """
    # Create split output folder inside the session folder.
    split_folder = session_folder + "_split"
    os.makedirs(split_folder, exist_ok=True)
    
    train_dir = os.path.join(split_folder, "train")
    val_dir = os.path.join(split_folder, "val")
    test_dir = os.path.join(split_folder, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(val_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # Get list of all image files (.png)
    image_files = [f for f in os.listdir(session_folder) if f.endswith('.png')]
    random.shuffle(image_files)
    
    total = len(image_files)
    n_train = int(total * split_ratio[0])
    n_val = int(total * split_ratio[1])
    
    train_files = image_files[:n_train]
    val_files = image_files[n_train:n_train+n_val]
    test_files = image_files[n_train+n_val:]
    
    # Move files to corresponding folders
    for f in train_files:
        shutil.move(os.path.join(session_folder, f), os.path.join(train_dir, f))
    for f in val_files:
        shutil.move(os.path.join(session_folder, f), os.path.join(val_dir, f))
    for f in test_files:
        shutil.move(os.path.join(session_folder, f), os.path.join(test_dir, f))
    
    print(f"Dataset split completed:")
    print(f"  Train: {len(train_files)} images")
    print(f"  Val  : {len(val_files)} images")
    print(f"  Test : {len(test_files)} images")
    print(f"Split folder created: {split_folder}")

def main():
    # Initialize ZED Camera
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.camera_fps = 30

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit(1)

    runtime_params = sl.RuntimeParameters()
    image = sl.Mat()

    # Create a new session folder for this recording session.
    session_folder = create_session_folder()
    print(f"Dataset recording started. Saving images to: {session_folder}")
    print("Press 'q' to stop recording.")

    frame_count = 0
    while True:
        if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()

            # Convert from RGBA to BGR for saving/viewing with OpenCV.
            bgr_frame = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)

            # Display frame live
            cv2.imshow("Dataset Collection", bgr_frame)
            
            # Save the frame as a PNG file.
            filename = os.path.join(session_folder, f"frame_{frame_count:06d}.png")
            cv2.imwrite(filename, bgr_frame)
            frame_count += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cv2.destroyAllWindows()
    zed.close()
    
    print(f"Dataset recording stopped. {frame_count} frames saved in: {session_folder}")
    
    # Automatically split the dataset into train, val, and test.
    split_dataset(session_folder)

if __name__ == "__main__":
    main()
