## zed_rail_tracker

**zed_rail_tracker** is a rail tracking system that uses YOLOv5 segmentation and a ZED 2i camera to detect rail tracks. It also includes an automatic dataset collection script that captures video frames from the ZED camera, then automatically organizes and splits the dataset into train, validation, and test folders. This dataset is ready to be uploaded to Roboflow or used for training your YOLOv5 segmentation model.

---

## Folder Structure

```
rail_tracker_ws/
├── src/
│   └── rail_tracker/
│       ├── scripts/
│       │   ├── main.py               # Main pipeline (ZED + YOLO segmentation + Depth)
│       │   ├── yolov5_segmentor.py   # YOLOv5 segmentation wrapper
│       │   ├── zed_wrapper.py        # ZED camera and depth retrieval wrapper
│       │   └── dataset_collector.py  # Dataset collection & automatic split script
│       ├── weights/
│       │   └── rail_seg.pt               # Your trained YOLOv5 segmentation model
│       ├── launch/
│       │   └── rail_tracker.launch   # (Not yet)
│       ├── CMakeLists.txt
│       └── package.xml
├── yolov5/                          # Cloned YOLOv5 repository (required for model loading)
└── dataset/
    └── raw/
        ├── 20230525_154512/           # Session folder (raw captured frames)
        │   ├── frame_000000.png
        │   ├── frame_000001.png
        │   └── ... (other frames)
        └── 20230525_154512_split/     # Split folder for the same session
            ├── train/
            │   ├── frame_000000.png   # ~70% of frames
            │   └── ...
            ├── val/
            │   ├── frame_000010.png   # ~15% of frames
            │   └── ...
            └── test/
                ├── frame_000015.png   # ~15% of frames
                └── ...
```

---

## File Descriptions

- **main.py**  
  Runs the primary pipeline: initializes the ZED camera, runs YOLOv5 segmentation on captured frames, retrieves depth, displays detections, and prints depth information.

- **yolov5_segmentor.py**  
  A wrapper module that loads your custom YOLOv5 segmentation model (stored in the `weights/` folder) and exposes a `predict()` function.

- **zed_wrapper.py**  
  A module that initializes the ZED camera (using `pyzed.sl`), retrieves images and depth data, and converts the image for processing (from RGBA to RGB).

- **dataset_collector.py**  
  Captures frames from the ZED camera and saves them to a timestamped folder in the `dataset/raw/` directory. When recording is stopped (by pressing `q`), it automatically splits the images into three subfolders: `train`, `val`, and `test` (default split: 70%, 15%, 15%). Each recording session generates a separate pair of session and split folders.

- **weights/**  
  Contains your trained YOLOv5 segmentation model file (e.g. `best.pt`).

- **yolov5/**  
  Contains the cloned YOLOv5 repository, required for loading the custom model using `torch.hub`.

- **dataset/raw/**  
  Contains all the raw captured frames organized into timestamped session folders and their corresponding split folders.

---

## How to Run

### 1. Setup Environment

- Install the required dependencies (see `requirements.txt` below):

  ```bash
  pip install -r requirements.txt
  ```

### 2. Running the Pipeline

- To test the full detection and depth pipeline:

  ```bash
  cd ~/rail_tracker_ws/src/rail_tracker/scripts
  python3 main.py
  ```

### 3. Running the Dataset Collector

- To capture and automatically split a dataset, run:

  ```bash
  cd ~/rail_tracker_ws/src/rail_tracker/scripts
  python3 dataset_collector.py
  ```

- A live window will display the feed; press **`q`** to stop recording.
- After stopping, check the `dataset/raw/` folder for your new session folder and its corresponding split folder.

---