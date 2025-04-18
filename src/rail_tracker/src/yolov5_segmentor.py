import torch

class YOLOv5Segmentor:
    def __init__(self, model_path, repo_path='yolov5'):
        self.model = torch.hub.load(repo_path, 'custom', path=model_path, source='local')
        self.model.eval()
        self.names = self.model.names

    def predict(self, frame):
        results = self.model(frame)
        return results
