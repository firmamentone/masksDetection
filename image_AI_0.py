from imageai.Detection.Custom import CustomObjectDetection

detector = CustomObjectDetection()
detector.setModelTypeAsYOLOv3()
detector.setModelPath("detection_model-ex-012--loss-0003.944.h5")
#detection_model-ex-011--loss-0004.233.h5
#detection_model-ex-012--loss-0003.684.h5
detector.setJsonPath("detection_config.json")
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image="inputimage.jpg", output_image_path="detected.jpg",minimum_percentage_probability=40)

for detection in detections:
    print(detection["name"], " : ", detection["percentage_probability"], " : ", detection["box_points"])