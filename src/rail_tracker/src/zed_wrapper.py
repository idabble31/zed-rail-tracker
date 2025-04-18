import pyzed.sl as sl
import cv2

class ZEDWrapper:
    def __init__(self, resolution=sl.RESOLUTION.HD720, fps=30):
        self.zed = sl.Camera()
        init_params = sl.InitParameters(camera_resolution=resolution, camera_fps=fps)
        if self.zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
            raise RuntimeError("Failed to open ZED camera")
        self.runtime_parameters = sl.RuntimeParameters()
        self.image = sl.Mat()
        self.depth = sl.Mat()

    def get_frame_and_depth(self):
        if self.zed.grab(self.runtime_parameters) == sl.ERROR_CODE.SUCCESS:
            self.zed.retrieve_image(self.image, sl.VIEW.LEFT)
            self.zed.retrieve_measure(self.depth, sl.MEASURE.DEPTH)
            frame = self.image.get_data()
            return cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB), self.depth
        return None, None

    def close(self):
        self.zed.close()
