from ultralytics import YOLO
import os
import cv2

def calculate_iou(bbox1, bbox2):
    x1, y1, x2, y2 = bbox1
    x3, y3, x4, y4 = bbox2
    w1, h1 = x2-x1, y2-y1
    w2, h2 = x4-x3, y4-y3
    x5, y5, x6, y6 = max(x1, x3), max(y1, y3), min(x2, x4), min(y2, y4)
    if x5 > x6 or y5 > y6:
        return 0
    intersection = (x6-x5)*(y6-y5)
    union = w1*h1 + w2*h2 - intersection
    return intersection/union

def calculate_all_ious(bboxes):
    ious = []
    num = bboxes.shape[0]
    for i in range(num):
        for j in range(i+1, num):
            ious.append(calculate_iou(bboxes[i], bboxes[j]))
    return ious

model = YOLO("LGG_best.pt")

test_path = r"datasets/images/test/"

results = model.predict(test_path, imgsz=256, device=3, stream=True, conf=0.3, iou=0.6, max_det=1
                        )
bad = []
FP = []
FP_confs = []
confs = []
ious = []

for result in results:
    FP_flag = False
    img_path = result.path

    im_bgr = result.plot()  # BGR-order numpy array
    W, H = im_bgr.shape[1], im_bgr.shape[0]
    
    boxes = result.boxes.cpu().numpy()
    if boxes.shape[0] > 1:
        bad.append(img_path)
        ious.extend(calculate_all_ious(boxes.xyxy))
    
    # load gt labels
    label_path = img_path.replace("images", "labels").replace(".jpg", ".txt")
    if os.path.isfile(label_path):
        with open(label_path, "r") as f:
            for line in f:
                line = line.strip().split()
                x, y, w, h = map(float, line[1:])
                x1, y1, x2, y2 = int((x-w/2)*W), int((y-h/2)*H), int((x+w/2)*W), int((y+h/2)*H)
                cv2.rectangle(im_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)
                # put GT text on the image right bottom of the bbox
                text_x = x2
                text_y = y2+20
                cv2.putText(im_bgr, "GT", (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    elif boxes.shape[0]>0:
        FP_flag = True
        FP.append(img_path)
        FP_confs.append(boxes.conf.item())
    if not FP_flag and boxes.shape[0]==1:
        confs.append(boxes.conf.item())
        
    basename = os.path.basename(img_path)
    cv2.imwrite(f"results/{basename}", im_bgr)   
    if boxes.shape[0] > 1:
        cv2.imwrite(f"results_bad/{basename}", im_bgr) 

print("="*10+"bad"+"="*10)
print(bad)
print("="*10+"FP"+"="*10)
print(FP)
print(FP_confs)
print("="*10+"confs"+"="*10)
print(min(confs))
print("="*10+"ious"+"="*10)
print(ious)
print(max(ious))