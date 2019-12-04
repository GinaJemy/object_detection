import numpy as np
import tensorflow as tf
import cv2 as cv
import roslib
import rospy
import sys
import label_map_util
from std_msgs.msg import String
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
# from object_detection.utils import visualization_utils as vis_util
from srv import DetectObjects, DetectObjectsResponse
from msg import DetectedObject, DetectedObjectsArray

class RosTensorFlow():
	def __init__(self):
		# # Read the graph.
		with tf.gfile.FastGFile('frozen_inference_graph.pb', 'rb') as f:
		    self.graph_def = tf.GraphDef()
		    self.graph_def.ParseFromString(f.read())
		PATH_TO_LABELS = 'label_map.pbtxt'
		# Categories
		self.NUM_CLASSES = 601
		self.label_map = label_map_util.load_labelmap(self.PATH_TO_LABELS)
		self.categories = label_map_util.convert_label_map_to_categories(self.label_map, max_num_classes=NUM_CLASSES, use_display_name=True)
		self.category_index = label_map_util.create_category_index(self.categories)
		self.sess = tf.Session
		self.bridge = CvBridge()

	def run_detector(self,image):
	# with tf.Session() as sess:
	    # Restore session
	    self.sess.graph.as_default()
	    tf.import_graph_def(self.graph_def, name='')

	    # Read and preprocess an image.
	    # img = cv.imread('/home/gina/image5.png')
	    img = self.bridge.imgmsg_to_cv2(image, "bgr8")
	    rows = img.shape[0]
	    cols = img.shape[1]
	    inp = cv.resize(img, (300, 300))
	    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

	    # Run the model
	    out = self.sess.run([self.sess.graph.get_tensor_by_name('num_detections:0'),
	                    self.sess.graph.get_tensor_by_name('detection_scores:0'),
	                    self.sess.graph.get_tensor_by_name('detection_boxes:0'),
	                    self.sess.graph.get_tensor_by_name('detection_classes:0')],
	                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

	    # Visualize detected bounding boxes.
	    num_detections = int(out[0][0])
	    msg = DetectedObjectsArray
	    for i in range(num_detections):
	        classId = int(out[3][0][i])
	        score = float(out[1][0][i])
	        bbox = [float(v) for v in out[2][0][i]]
	        if score > 0.3:
	            x = bbox[1] * cols
	            y = bbox[0] * rows
	            right = bbox[3] * cols
	            bottom = bbox[2] * rows
	            # print(classId)
	            # print(score)
	            # print(self.category_index[classId])
	            obj = DetectedObject
	            obj.xlt = x
	            obj.ylt = y
	            obj.xrb = right
	            obj.yrb = bottom
	            obj.ClassName = self.category_index[classId]["name"]
	            obj.probability = score
	            msg.append(obj)
	            # cv.rectangle(img, (int(x), int(y)), (int(right), int(bottom)), (125, 255, 51), thickness=2)


	    
	    return msg
		# cv.imshow('TensorFlow FasterRCNN', img)


		# # cv.waitKey()= img.shape[0]
	    

def detection_server():
    rospy.init_node('detection_server')
    s = rospy.Service('detect', DetectObjects, run_detector)
    rospy.spin()

if __name__ == "__main__":
    detection_server()
