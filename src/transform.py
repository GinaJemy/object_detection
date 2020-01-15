#!/usr/bin/env python3
import numpy as np
import roslib
import rospy
from realsense_perception.msg import DetectedObject, DetectedObjectsArray

theta1 =35.583455 
d1 = 0
a1 = 0.075
alpha1 = np.pi / 2.0

theta2 = -53.747694
d2 = 0
a2 = 0.3
alpha2 = 0

theta3 = -55.641353 
d3 = 0
a3 = 0.075
alpha3 = np.pi / 2.0

theta4 = -0.003467 
d4 = 0.32
a4 = 0 
alpha4 = -np.pi / 2.0

theta5 =  -0.008734
d5 = 0
a5 = 0 
alpha5 = np.pi / 2.0 

theta6 =  0.004491
d6 =0.08
a6 = 0
alpha6 = 0

T01 = np.array([[np.cos(theta1),0,np.sin(theta1),a1*np.cos(theta1)],
				[np.sin(theta1),0,-np.cos(theta1),a1*np.sin(theta1)],
				[0,1,0,0],
				[0,0,0,1]])

T12 = np.array([[np.cos(theta2),-np.sin(theta2),0,a2*np.cos(theta2)],
				[np.sin(theta2),np.cos(theta2),0,a2*np.sin(theta2)],
				[0,0,1,0],
				[0,0,0,1]])

T23 = np.array([[np.cos(theta3),0,np.sin(theta3),a3*np.cos(theta3)],
				[np.sin(theta3),0,-np.cos(theta3),a3*np.sin(theta3)],
				[0,1,0,0],
				[0,0,0,1]])

T34 = np.array([[np.cos(theta4),0,-np.sin(theta4),0],
				[np.sin(theta4),0,np.cos(theta4),0],
				[0,-1,0,d4],
				[0,0,0,1]])

T45 = np.array([[np.cos(theta5),0,np.sin(theta5),0],
				[np.sin(theta5),0,-np.cos(theta5),0],
				[0,1,0,0],
				[0,0,0,1]])

T56 = np.array([[np.cos(theta6),0,-np.sin(theta6),0],
				[np.sin(theta6),0,np.cos(theta6),0],
				[0,0,1,d6],
				[0,0,0,1]])

T06 = T01.dot(T12).dot(T23).dot(T34).dot(T45).dot(T56)



T60 = np.linalg.pinv(T06)

def callback(data):
	count = data.count
	detected = data.detectedObjects
	length = len(detected) 
	i = 0
	while i < length:
		obj_x = detected[i].x
		obj_y = detected[i].y
		obj_z = detected[i].z
		pt = np.array([[obj_x],
			   [obj_y],
			   [obj_z],
			   [1]])
		transformed_pt = T60.dot(pt)
		print(detected[i].ClassName+" "+np.array2string(transformed_pt))
		i += 1
	
	

rospy.init_node('coordinate_transformer')
rospy.Subscriber("Objects", DetectedObjectsArray, callback)
while(1):
	rospy.spin()
