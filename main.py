from ultralytics import YOLO

# create a yolov8 detection model
model = YOLO("yolov8s.pt")

results = model.train(
    data="data.yaml",
    epochs=20,
    batch=64,
    imgsz=256,
    device=0,
    project="yolo_log",
    name="train",
)

## adjust parameters and apply data augmentation
model_t = YOLO("yolov8s.pt")
results = model_t.train(
    data="tune.yaml",   # add more data augmentation process
    epochs=20,
    batch=32,   # adjust batch size
    imgsz=256,
    device=0,
    project="yolo_log",
    name="tune",
    # cos_lr=True,  # use cosine lr scheduler
)

model_t.save("LGG_best.pt")
