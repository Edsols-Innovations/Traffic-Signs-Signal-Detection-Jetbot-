#!/usr/bin/env python3
#
# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.
#

from jetson_inference import detectNet
from jetson_utils import videoSource, videoOutput
from threading import Thread
import time
import argparse
import sys  
from jetbot import Robot

net = detectNet(argv=["--model=/home/jetbot/jetson-inference/python/training/detection/ssd/models/tf_Signs-mobileNet_v1.onnx","--labels=/home/jetbot/jetson-inference/python/training/detection/ssd/models/labels.txt","--input-blob=input_0","--output-cvg=scores","--output-bbox=boxes","--Confidence=80"],threshold=0.5)
camera = videoSource("csi://0")      # '/dev/video0' for V4L2
display = videoOutput("display://0") # 'my_video.mp4' for file

robot=Robot()

def start_car(detection):
	robot.stop()
	if detection.ClassID == 0 or detection.ClassID == 11 :   #BACKGROUND & signal green
		time.sleep(5)		
		robot.forward(0.8)
	elif detection.ClassID == 1 or detection.ClassID == 12 :   #cross walk & Road Cross walk
		robot.stop()
		time.sleep(5)
		robot.forward(0.1)
	elif detection.ClassID == 2 :   #gate closed
		robot.stop()
	elif detection.ClassID == 3 :   #gate opened
		robot.forward(0.8)
	elif detection.ClassID == 4 :   #stop
		robot.stop()
		time.sleep(20)
		robot.forward(0.1)
	elif detection.ClassID == 5 :   #no entry
		robot.stop()
		time.sleep(3)
		robot.right(0.1)
		time.sleep(3)
		robot.forward(0.8)
	elif detection.ClassID == 6 :   #speed limit 50	
		robot.forward(0.3)
		time.sleep(10)
		robot.forward(0.15)
	elif detection.ClassID == 7 or detection.ClassID == 8 or detection.ClassID == 9 or detection.ClassID == 10 or detection.ClassID == 11 :
		robot.forward(0.8)
	elif detection.ClassID == 12 :
		robot.forward(0.8)

#thread = Thread(target=start_car(detection))
while True:
	img = camera.Capture()
	detections = net.Detect(img)
	# print the detections
	print("detected {:d} objects in image".format(len(detections)))

	
	for detection in detections:
		print(detection)
		start_car(detection)

	display.Render(img)
	display.SetStatus("Object Detection | Network {:.0f} FPS".format(net.GetNetworkFPS()))
	
	# print out performance info
	net.PrintProfilerTimes()
	
	if not camera.IsStreaming() or not display.IsStreaming():
		#jetson_utils.gstCamera.Close()  
		break

	try:
		print("ctrl+c to close")
	except KeyboardInterrupt:
		break
	else:
		print("_______________")




