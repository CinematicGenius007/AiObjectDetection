import cv2 #pip install opencv-python
# import matplotlib.pyplot as plt #pip install matplot lib
import os
import numpy as np
import skvideo.io

"""
This is pre code, very important!
"""

config_file = "ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt"
frozen_model = "frozen_inference_graph.pb"

model = cv2.dnn_DetectionModel(frozen_model, config_file)

classLabels = [] #empty list of python
file_name = 'Labels.txt'
with open(file_name, 'rt') as fpt:
    classLabels = fpt.read().rstrip('\n').split('\n')
    #classLabels.append(fpt.read

model.setInputSize(320, 320)
model.setInputScale(1.0/127.5) # 255/2=127.5
model.setInputMean((127.5,127.5,127.5)) #mobilenet => [-1,1]
model.setInputSwapRB(True)

def show(url=None):
    if url:
        if 'mp4' in url:
            cap = cv2.VideoCapture(url) #check if the video is opened correctly if not cap.isOpened(): cap = cv2.VideoCapture(0) if not cap.isOpened(): raise IDEerror("Cannot open video")

            font_scale = 3
            font = cv2.FONT_HERSHEY_PLAIN

            # out = cv2.VideoWriter('static/video/project.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, (cap.get(3), cap.get(4)))

            out_source = []
            
            while True:
                ret, frame = cap.read()
                # print("This is called!")
                if ret == False:
                    cap.release()
                    break
                    
            
                ClassIndex, confidence, bbox = model.detect(frame, confThreshold=0.55)
            
                # print(ClassIndex)
                if (len(ClassIndex) != 0):
                    for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                        if (ClassInd<=80):
                            cv2.rectangle(frame,boxes,(255, 0, 0), 2)
                            cv2.putText(frame,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale, color=(0, 250, 0), thickness=3)
                out_source.append(frame)
                # weigth, height, som = frame.shape
                # print(f"frame shape: {frame.shape}")
                # print(frame)
                
                if cv2.waitKey(2) & 0xFF == ord('q'):
                    cap.release()
                    break
        #     with skvideo.io.FFmpegWriter('static/video/frames.mp4', inputdict={'-pix_fmt':'bgr24'},
        # outputdict={'-c:v': 'libx264','-preset': 'medium',
        # '-profile:v':'high', '-level': '4.0','-pix_fmt': 'yuv420p'},
        # verbosity=10) as writer:
        #         writer._proc = None

        #         # outputdata = np.random.random(size=(5, 480, 680, 3)) * 255
        #         # outputdata = outputdata.astype(np.uint8)
        #         outputdata = np.array(out_source)
        #         outputdata = outputdata.astype(np.uint8)
                
        #         for i in range(len(out_source)):
        #             writer.writeFrame(outputdata[i, :, :, :])

                
            out_video = np.empty([len(out_source), 608, 512, 3], dtype = np.uint8)
            out_video =  out_video.astype(np.uint8)
            for i in range(len(out_source)):
                out_video[i] = out_source[i]
            skvideo.io.vwrite('static/video/frames.mp4', out_video)
            
            return os.path.join('static/video/', 'frames.mp4')
        else:
            img = cv2.imread(url)
            # plt.imshow(img) #bgr format
            # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ClassIndex, confidence, bbox = model.detect(img, confThreshold=0.5)
            font_scale = 3
            font = cv2.FONT_HERSHEY_PLAIN
            for ClassInd, conf, boxes, in zip(ClassIndex.flatten(), confidence.flatten(), bbox):
                cv2.rectangle(img,boxes,(255, 0, 0), 2)
                cv2.putText(img,classLabels[ClassInd-1],(boxes[0]+10,boxes[1]+40), font, fontScale=font_scale, color=(0, 250, 0), thickness=3)
                # plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            # final_image = 
            # new_file_path = os.path.join('static/', 'new_image.jpg')
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            cv2.imwrite(os.path.join('static/image/', 'new_image.jpg'), cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            return os.path.join('static/image/', 'new_image.jpg')



"""
php commands: 

localhost/phpmyadmin/

localhost/BorderControlProtocol/login.php
"""