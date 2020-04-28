#based on custom object detection example of imageAI  --https://imageai.readthedocs.io/en/latest/customdetection/
#For model tranning: https://colab.research.google.com/drive/1nv6TluehJkFGKLdbnHayWXa0gVzI2voB

#Apr 28 2020. T.I. Created this file 

from imageai.Detection.Custom import CustomVideoObjectDetection
import os
import cv2


execution_path = os.getcwd()
camera = cv2.VideoCapture(0)

video_detector = CustomVideoObjectDetection()
video_detector.setModelTypeAsYOLOv3()

video_detector.setModelPath("detection_model-ex-012--loss-0003.944.h5")
#Please download the h5 modle from "https://drive.google.com/file/d/1sNdDf_gCu8QZ16xxMkq_U3THAyewsVxT/view?usp=sharing"

video_detector.setJsonPath("detection_config.json")
video_detector.loadModel()

video_detector.detectObjectsFromVideo(camera_input=camera,
                                          output_file_path=os.path.join(execution_path, "maskesDetection"),
                                          frames_per_second=2,
                                          minimum_percentage_probability=60,
                                          log_progress=True,
                                          )
