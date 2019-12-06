#!/usr/bin/env python
import sys
import numpy as np
import os
import cv2 as cv
import tensorflow as tf
import roslib
import rospy
import label_map_util
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from realsense_perception.srv import DetectObjects, DetectObjectsResponse
from realsense_perception.msg import DetectedObject, DetectedObjectsArray

class DetectionServer():
	def __init__(self):
		# # Read the graph.
		with tf.gfile.FastGFile('/home/gina/cam_ws/src/realsense_perception/src/frozen_inference_graph.pb', 'rb') as f:
		    self.graph_def = tf.GraphDef()
		    self.graph_def.ParseFromString(f.read())
		PATH_TO_LABELS = '/home/gina/cam_ws/src/realsense_perception/src/label_map.pbtxt'
		# Categories
		self.NUM_CLASSES = 601
		self.label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
		self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=self.NUM_CLASSES, use_display_name=True)
		self.category_index = label_map_util.create_category_index(self.categories)
		self.bridge = CvBridge()
		s = rospy.Service('detect', DetectObjects, self.run_detector)
	def run_detector(self,image):
		with tf.Session() as sess:
		    # Restore session
		    print("callback fn")
		    sess.graph.as_default()
		    
		    tf.import_graph_def(self.graph_def, name='')

		    # Convert sensor_msg Image to CV image and preprocess
		    img = self.bridge.imgmsg_to_cv2(image.img, "bgr8")
		    rows = img.shape[0]
		    cols = img.shape[1]
		    inp = cv.resize(img, (300, 300))
		    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

		    # Run the model
		    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
		                    sess.graph.get_tensor_by_name('detection_scores:0'),
		                    sess.graph.get_tensor_by_name('detection_boxes:0'),
		                    sess.graph.get_tensor_by_name('detection_classes:0')],
		                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

		    # Construct detected objects message
		    num_detections = int(out[0][0])
		    det_list = []
		    msg = DetectedObjectsArray
		    for i in range(num_detections):
		        classId = int(out[3][0][i])
		        score = float(out[1][0][i])
		        bbox = [float(v) for v in out[2][0][i]]
		        if score > 0.1:
		            x = bbox[1] * cols
		            y = bbox[0] * rows
		            right = bbox[3] * cols
		            bottom = bbox[2] * rows
		            obj = DetectedObject()
		            obj.xlt = x
		            obj.ylt = y
		            obj.xrb = right
		            obj.yrb = bottom
		            obj.ClassName = self.category_index[classId]["name"]
		            obj.probability = score
		            det_list.append(obj)
		    msg.detectedObjects = det_list
		    msg.count = int(out[0][0])
	    	return msg	    

	def main(self):
		rospy.spin()

if __name__ == "__main__":
	rospy.init_node('detection_server')
	det = DetectionServer()
	det.main()
