from ultralytics import YOLO

model=YOLO("last.pt")

results = model.predict(source="0",show=True)
print(results)