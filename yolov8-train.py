from ultralytics import YOLO

#调用yolov8模型利用我们的数据集对其进行训练
model = YOLO('yolov8n.pt')

#调用我们数据集的配置文件并设置训练
model.train(data='yolo.yaml',workers=0,epochs=50,batch=16)