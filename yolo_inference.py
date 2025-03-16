from ultralytics import YOLO


#model = YOLO('yolov8x')
#model = YOLO('models/best_3.pt')
model = YOLO('models/tracker_darts.pt')

results = model.predict('inputs/DartsInput2.mp4', save=True)
print(results[0])
print('-----------------------')
for box in results[0].boxes:
    print(box)