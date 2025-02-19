from torch.utils.data._utils import worker
from ultralytics import YOLO

if __name__ == '__main__':
    # Load model
    model = YOLO("yolo11n.pt")

    # Train using GPU
    model.train(data=r"D:\projectKHMT\data.yaml",
                epochs=100, imgsz=640, device="cpu")  # hoặc device=0 nếu có nhiều GPU

#update code