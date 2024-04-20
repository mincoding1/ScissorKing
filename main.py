from PySide2.QtWidgets import *
from PySide2.QtGui import *
from PySide2.QtCore import *

from ui import Ui_Form
import cv2
import numpy as np
from datetime import datetime
import os
import random
import argparse
import sys
import time
from threading import Thread
import importlib.util
import random

from picamera2 import Picamera2

device = 'RPI'
#device = 'JETSON_NANO'

class VideoStream:

    def __init__(self,resolution=(640,480),framerate=30):
        global device
        if device == 'RPI' : 
            self.rpi_camera_init()
        if device == 'JETSON_NANO':
            self.jetsonNano_camera_init()

        self.stopped = False

    def rpi_camera_init(self):
        self.stream = Picamera2()
        self.stream.configure(self.stream.create_still_configuration(main={"format": 'BGR888', "size": (640, 480)}))
        self.stream.set_controls({"ExposureTime": 20000, "AnalogueGain": 1.0})
        self.stream.start()
        self.frame = self.stream.capture_array()       
        # self.frame = cv2.resize(self.frame, (640, 480))

    def jetsonNano_camera_init(self):
        self.stream = cv2.VideoCapture('nvarguscamerasrc ! video/x-raw(memory:NVMM), width=640, height=480, format=(string)NV12, framerate=(fraction)20/1 ! nvvidconv ! video/x-raw, format=(string)BGRx ! videoconvert ! video/x-raw, format=(string)BGR ! appsink', cv2.CAP_GSTREAMER)
        _, self.frame = self.stream.read()

    def start(self):
	# Start the thread that reads frames from the video stream
        Thread(target=self.update,args=()).start()
        return self

    def update(self):
        global device
        # Keep looping indefinitely until the thread is stopped
        while True:
            # If the camera is stopped, stop the thread
            if self.stopped:
                return

            if device == 'RPI' : 
                self.frame = self.stream.capture_array()
                self.frame = cv2.resize(self.frame, (640, 480))
            if device == 'JETSON_NANO':
                _, self.frame = self.stream.read()

    def read(self):
	# Return the most recent frame
        return self.frame

    def stop(self):
	# Indicate that the camera and thread should be stopped
        self.stopped = True

# Define and parse input arguments
parser = argparse.ArgumentParser()
parser.add_argument('--modeldir', help='Folder the .tflite file is located in',
                    default='tflite')
parser.add_argument('--graph', help='Name of the .tflite file, if different than detect.tflite',
                    default='detect.tflite')
parser.add_argument('--labels', help='Name of the labelmap file, if different than labelmap.txt',
                    default='labelmap.txt')
parser.add_argument('--threshold', help='Minimum confidence threshold for displaying detected objects',
                    default=0.5)
parser.add_argument('--resolution', help='Desired webcam resolution in WxH. If the webcam does not support the resolution entered, errors may occur.',
                    default='640x480')
parser.add_argument('--edgetpu', help='Use Coral Edge TPU Accelerator to speed up detection',
                    action='store_true')

args = parser.parse_args()

MODEL_NAME = args.modeldir
GRAPH_NAME = args.graph
LABELMAP_NAME = args.labels
min_conf_threshold = float(args.threshold)
resW, resH = args.resolution.split('x')
imW, imH = int(resW), int(resH)
use_TPU = args.edgetpu

# Import TensorFlow libraries
# If tflite_runtime is installed, import interpreter from tflite_runtime, else import from regular tensorflow
# If using Coral Edge TPU, import the load_delegate library
pkg = importlib.util.find_spec('tflite_runtime')
if pkg:
    from tflite_runtime.interpreter import Interpreter
    if use_TPU:
        from tflite_runtime.interpreter import load_delegate
else:
    from tensorflow.lite.python.interpreter import Interpreter
    if use_TPU:
        from tensorflow.lite.python.interpreter import load_delegate

# If using Edge TPU, assign filename for Edge TPU model
if use_TPU:
    # If user has specified the name of the .tflite file, use that name, otherwise use default 'edgetpu.tflite'
    if (GRAPH_NAME == 'detect.tflite'):
        GRAPH_NAME = 'edgetpu.tflite'       

# Get path to current working directory
CWD_PATH = os.getcwd()

# Path to .tflite file, which contains the model that is used for object detection
PATH_TO_CKPT = os.path.join(CWD_PATH,MODEL_NAME,GRAPH_NAME)

# Path to label map file
PATH_TO_LABELS = os.path.join(CWD_PATH,MODEL_NAME,LABELMAP_NAME)

# Load the label map
with open(PATH_TO_LABELS, 'r') as f:
    labels = [line.strip() for line in f.readlines()]

# Have to do a weird fix for label map if using the COCO "starter model" from
# https://www.tensorflow.org/lite/models/object_detection/overview
# First label is '???', which has to be removed.
if labels[0] == '???':
    del(labels[0])

# Load the Tensorflow Lite model.
# If using Edge TPU, use special load_delegate argument
if use_TPU:
    interpreter = Interpreter(model_path=PATH_TO_CKPT,
                              experimental_delegates=[load_delegate('libedgetpu.so.1.0')])
    print(PATH_TO_CKPT)
else:
    interpreter = Interpreter(model_path=PATH_TO_CKPT)

interpreter.allocate_tensors()

# Get model details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
height = input_details[0]['shape'][1]
width = input_details[0]['shape'][2]

floating_model = (input_details[0]['dtype'] == np.float32)

input_mean = 127.5
input_std = 127.5

# Check output layer name to determine if this model was created with TF2 or TF1,
# because outputs are ordered differently for TF2 and TF1 models
outname = output_details[0]['name']

if ('StatefulPartitionedCall' in outname): # This is a TF2 model
    boxes_idx, classes_idx, scores_idx = 1, 3, 0
else: # This is a TF1 model
    boxes_idx, classes_idx, scores_idx = 0, 1, 2

# Initialize frame rate calculation
frame_rate_calc = 1
freq = cv2.getTickFrequency()

# Initialize video stream
videostream = VideoStream(resolution=(imW,imH),framerate=30).start()
time.sleep(1)


cpu_state = 'ready'
user_state = 'ready'


class BoundCatch:
    def __init__(self):
        super().__init__()        
        self.bound_state = None

    flag = False
    def run(self):
        self.flag = True
        while self.flag:
            frame = videostream.read()            
            global frame_rate_calc
            global input_details, output_details
            global height, width
            global user_state

            t1 = cv2.getTickCount()

            # Acquire frame and resize to expected shape [1xHxWx3]
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_resized = cv2.resize(frame_rgb, (width, height))
            input_data = np.expand_dims(frame_resized, axis=0)

            # Normalize pixel values if using a floating model (i.e. if model is non-quantized)
            if floating_model:
                input_data = (np.float32(input_data) - input_mean) / input_std

            # Perform the actual detection by running the model with the image as input
            interpreter.set_tensor(input_details[0]['index'],input_data)
            interpreter.invoke()

            # Retrieve detection results
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # Bounding box coordinates of detected objects
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # Class index of detected objects
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # Confidence of detected objects

            # Calculate framerate
            t2 = cv2.getTickCount()
            time1 = (t2-t1)/freq
            frame_rate_calc= 1/time1

            self.bound_state = (boxes, classes, scores)

    def get_bound_info(self):
        return self.bound_state

    def start(self):
        Thread(target=self.run).start()
        return self
    
    def stop(self):
        self.flag = False

boundstream = BoundCatch().start()


class MyThread(QThread):
    mySignal = Signal(QPixmap)
    #mySignal = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()

    flag = False
    def run(self):
        self.flag = True
        while self.flag:
            frame1 = videostream.read()

            if boundstream.get_bound_info() is not None:
                frame1 = self.make_bound_image(frame1)

            self.shot_image(frame1)
            #time.sleep(0.1)

    def make_bound_image(self, frame):
        global frame_rate_calc
        global input_details, output_details
        global height, width
        global user_state
        global frame_rate_calc

        boxes = boundstream.get_bound_info()[0]
        classes = boundstream.get_bound_info()[1]
        scores = boundstream.get_bound_info()[2]

        # Loop over all detections and draw detection box if confidence is above minimum threshold
        for i in range(len(scores)):
            if ((scores[i] > min_conf_threshold) and (scores[i] <= 1.0)):

                # Get bounding box coordinates and draw box
                # Interpreter can return coordinates that are outside of image dimensions, need to force them to be within image using max() and min()
                ymin = int(max(1,(boxes[i][0] * imH)))
                xmin = int(max(1,(boxes[i][1] * imW)))
                ymax = int(min(imH,(boxes[i][2] * imH)))
                xmax = int(min(imW,(boxes[i][3] * imW)))
                
                cv2.rectangle(frame, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)

                # Draw label
                object_name = labels[int(classes[i])] # Look up object name from "labels" array using class index
                user_state = object_name
                label = '%s: %d%%' % (object_name, int(scores[i]*100)) # Example: 'person: 72%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.7, 2) # Get font size
                label_ymin = max(ymin, labelSize[1] + 10) # Make sure not to draw label too close to top of window
                cv2.rectangle(frame, (xmin, label_ymin-labelSize[1]-10), (xmin+labelSize[0], label_ymin+baseLine-10), (255, 255, 255), cv2.FILLED) # Draw white box to put label text in
                cv2.putText(frame, label, (xmin, label_ymin-7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2) # Draw label text

        # Draw framerate in corner of frame
        cv2.putText(frame,'FPS: {0:.2f}'.format(frame_rate_calc),(30,50),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,0),2,cv2.LINE_AA)
        return frame


    def shot_image(self, img):
        h, w, byte = img.shape
        img = QImage(img, w, h, byte*w, QImage.Format_RGB888)
        q_img = QPixmap(img)
        self.mySignal.emit(q_img)

    def stop(self):
        self.flag = False




class ComputerPlay(QThread):
    mySignal = Signal(QPixmap)
    #mySignal = pyqtSignal(QPixmap)

    def __init__(self):
        super().__init__()        

    flag = False
    def run(self):
        global cpu_state
        self.flag = True
        while self.flag:           
            if cpu_state == 'ready':
                self.cap = cv2.VideoCapture('computer.mp4') 
                self.ready()
                self.cap.release()
            elif cpu_state == 'paper':
                cpu_state = 'ready'
                self.cap = cv2.VideoCapture('computer.mp4') 
                self.paper()
                self.cap.release()
                self.wait_()            
            elif cpu_state == 'rock':
                cpu_state = 'ready'
                self.cap = cv2.VideoCapture('computer.mp4') 
                self.rock()
                self.cap.release()
                self.wait_()
            elif cpu_state == 'scissor':
                cpu_state = 'ready'
                self.cap = cv2.VideoCapture('computer.mp4') 
                self.scissor()
                self.cap.release()
                self.wait_()
            
            time.sleep(0.01)


    #stop video play
    def wait_(self):
            frame_cnt = 0
            while True:
                frame_cnt += 1
                if self.flag == False:
                    break
                if cpu_state != 'ready':
                    break

                if frame_cnt == 30: break
                time.sleep(0.1)

    def ready(self):
        frame_cnt = 0
        while self.cap.isOpened():
            frame_cnt += 1
            if self.flag == False:
                break
                
            _, img = self.cap.read()            
            if img is None: 
                break

            if cpu_state != 'ready':
                break

            self.shot_image(img)
            if frame_cnt < 20:
                time.sleep(0.06)
            else:
                break

    def paper(self):
        frame_cnt = 0
        while self.cap.isOpened():
            frame_cnt += 1
            if self.flag == False:
                break
                
            _, img = self.cap.read()
            if img is None: 
                break

            self.shot_image(img)
            if frame_cnt < 40:
                time.sleep(0.04)
            elif frame_cnt < 60:
                time.sleep(0.01)
            else:
                break

    def rock(self):
        frame_cnt = 0
        while self.cap.isOpened():
            frame_cnt += 1
            if self.flag == False:
                break
                
            _, img = self.cap.read()
            if img is None: 
                break

            self.shot_image(img)
            if frame_cnt < 40:
                time.sleep(0.03)
            elif frame_cnt < 60:
                time.sleep(0.01)
            elif frame_cnt < 70:
                time.sleep(0.02)
            else:
                break

    def scissor(self):
        frame_cnt = 0
        while self.cap.isOpened():
            frame_cnt += 1
            if self.flag == False:
                break
                
            _, img = self.cap.read()
            if img is None: 
                break

            self.shot_image(img)
            if frame_cnt < 40:
                time.sleep(0.04)
            elif frame_cnt < 50:
                time.sleep(0.01)
            else:
                break

    def shot_image(self, img):
        h, w, byte = img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = QImage(img, w, h, byte*w, QImage.Format_RGB888)
        q_img = QPixmap(img)
        self.mySignal.emit(q_img)

    def stop(self):
        self.flag = False


class GamePlay(QThread):
    mySignal = Signal(str, int)
    #mySignal = pyqtSignal(str, int)

    def __init__(self):
        super().__init__()

    flag = False
    def run(self):                
        global cpu_state, user_state
        self.flag = True
        while self.flag:
            self.shot_text('READY!!!')
            cpu_state = 'ready'
            time.sleep(2)
            cpu_answer = random.choice(['paper', 'rock', 'scissor'])
            cpu_state = cpu_answer
            for i in range(2, -1, -1):                                
                self.shot_text(str(i) + ' !!')
                time.sleep(1)
            self.shot_text('SHOT')
            time.sleep(1.2)
            self.shot_text('HUM...')
            time.sleep(1.2)
            print(user_state, cpu_answer)
            self.shot_text(self.who_win(cpu_answer, user_state), 80)
            user_state = 'ready'
            time.sleep(2)

    def who_win(self, cpu, user):
        c = 0
        u = 0
        if user == 'ready':
            return 'You Lose'
        
        if user == 'SSAFY':
            return 'Incredibly Win'

        if user == 'Rock' : u = 1
        if user == 'Scissor' : u = 2
        if user == 'Paper' : u = 3

        if cpu == 'rock' : c = 1
        if cpu == 'scissor' : c = 2
        if cpu == 'paper' : c = 3

        if c == u : return 'Draw'
        if c == 1 and u == 2 : return 'You Lose'
        if c == 1 and u == 3 : return 'You Win'

        if c == 2 and u == 1 : return 'You Win'
        if c == 2 and u == 3 : return 'You Lose'

        if c == 3 and u == 1 : return 'You Lose'
        if c == 3 and u == 2 : return 'You Win'
        return 'Unknown'

    def shot_text(self, txt, size=40):        
        self.mySignal.emit(txt, size)

    def stop(self):
        self.flag = False


class MyApp(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.main()

    def main(self):
        self.th = MyThread()
        self.th.mySignal.connect(self.setImage1)
        self.th.start()   

        self.th2 = ComputerPlay()
        self.th2.mySignal.connect(self.setImage2)
        self.th2.start()     

        self.th3 = GamePlay()
        self.th3.mySignal.connect(self.setGameStatus)
        self.th3.start()

    def setImage1(self, img):
        self.label1.setPixmap(img)

    def setImage2(self, img):
        self.label2.setPixmap(img)

    def setGameStatus(self, txt, size):
        self.game_label.setStyleSheet('font: 500 ' + str(size) + 'pt;')
        self.game_label.setText(txt)

    def closeEvent(self, event):
        self.th.stop()        
        self.th2.stop()
        self.th3.stop()
        boundstream.stop()
        videostream.stop()
        time.sleep(1)
        self.close()
        cv2.destroyAllWindows()

if __name__ == '__main__':
    app = QApplication([])
    win = MyApp()
    win.show()
    app.exec_()
