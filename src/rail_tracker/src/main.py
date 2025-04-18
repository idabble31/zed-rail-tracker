import cv2
import math
from yolov5_segmentor import YOLOv5Segmentor
from zed_wrapper import ZEDWrapper

def main():
    detector = YOLOv5Segmentor(model_path='../../weights/best.pt', repo_path='../yolov5')
    camera = ZEDWrapper()

    try:
        while True:
            frame, depth = camera.get_frame_and_depth()
            if frame is None:
                continue

            results = detector.predict(frame)
            results.render()

            for *xyxy, conf, cls in results.xyxy[0]:
                x1, y1, x2, y2 = map(int, xyxy)
                cx = int((x1 + x2) / 2)
                cy = int((y1 + y2) / 2)

                status, depth_val = depth.get_value(cx, cy)
                label = detector.names[int(cls)]

                if not math.isnan(depth_val):
                    print(f"[{label}] at ({cx},{cy}) -> Depth: {depth_val:.2f} m")
                    cv2.putText(results.imgs[0], f"{depth_val:.2f} m", (cx, cy - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.imshow("YOLOv5 Segmentation on ZED", results.imgs[0])
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cv2.destroyAllWindows()
        camera.close()

if __name__ == '__main__':
    main()
