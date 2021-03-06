// This example is derived from the ssd_mobilenet_object_detection opencv demo
// and adapted to be used with Intel RealSense Cameras
// and to use a yolo v2 classifier
// Please see https://github.com/opencv/opencv/blob/master/LICENSE

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/videoio.hpp>
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"
#include <librealsense2/rsutil.h>
#include "ros/ros.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "realsense_perception/DetectedObject.h"
#include "realsense_perception/DetectedObjectsArray.h"

using namespace cv;
using namespace cv::dnn;
using namespace rs2;
using namespace std;

const int inpWidth = 416;        // Width of network's input image
const int inpHeight = 416;       // Height of network's input image

const float WHRatio       = inpWidth / (float)inpHeight;
const float inScaleFactor = 1/255.0; //to scale values between 0-1 instead of 0-255
const float meanVal       = 0.5;  //mean values which are subtracted from channels to achieve data normalization
const float confidenceThreshold = 0.2f;

// Give the configuration and weight files for the model
String modelConfiguration = "/home/gina/cam_ws/src/realsense_perception/src/yolo-obj.cfg";
String modelWeights = "/home/gina/cam_ws/src/realsense_perception/src/yolo-obj6400.backup";

int main(int argc, char** argv) try
{
    //Initialize node and advertise published topic
    ros::init(argc, argv, "perception");
    ros::NodeHandle n;
    ros::Publisher pub = n.advertise<realsense_perception::DetectedObjectsArray>("Objects", 1000);

    //Load names of classes
    vector<String> classNamesVec;
    String classesFile = "/home/gina/cam_ws/src/realsense_perception/src/obj.names";
    ifstream ifs(classesFile.c_str());
    string line;
    while (getline(ifs, line)) classNamesVec.push_back(line);

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);
    net.setPreferableTarget(DNN_TARGET_CPU);

    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR)
                         .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

    auto stream = config.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics(); // Calibration data

    //Crop image to network's input size
    Size cropSize;
    if (profile.width() / (float)profile.height() > WHRatio)
    {
        cropSize = Size(static_cast<int>(profile.height() * WHRatio),
                        profile.height());
    }
    else
    {
        cropSize = Size(profile.width(),
                        static_cast<int>(profile.width() / WHRatio));
    }

    Rect crop(Point((profile.width() - cropSize.width) / 2,
                    (profile.height() - cropSize.height) / 2),
              cropSize);

    //Window to show detections
    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);

    //Video file to write output
/*    string fname = "out";
    VideoWriter opvideo;
    opvideo.open(fname+to_string(time(nullptr))+".avi",VideoWriter::fourcc('M','J','P','G'),5, cropSize ,true);
     if (!opvideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << endl;
        return -1;
    }
*/
    //Video file to use for offline testing
/*    string filename = "/media/gina/Backup Plus/test";
    VideoWriter testvideo;
    testvideo.open(filename+to_string(time(nullptr))+".avi",VideoWriter::fourcc('M','J','P','G'),5, cropSize ,true);
     if (!testvideo.isOpened())
    {
        cout  << "Could not open the output video for write: " << endl;
        return -1;
    }
*/
    while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0)
    {
        // Wait for the next set of frames
        auto data = pipe.wait_for_frames();
        // Make sure the frames are spatially aligned
        data = align_to.process(data);

        auto color_frame = data.get_color_frame();
        auto depth_frame = data.get_depth_frame();

        // If we only received new depth frame,
        // but the color did not update, continue
        static int last_frame_number = 0;
        if (color_frame.get_frame_number() == last_frame_number) continue;
        last_frame_number = color_frame.get_frame_number();

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        auto depth_mat = depth_frame_to_meters(pipe, depth_frame);

        // Crop both color and depth frames
        color_mat = color_mat(crop);
        depth_mat = depth_mat(crop);

        //Save videos for offline testing 
//        testvideo.write(color_mat);

        //Convert Mat to batch of images
        Mat inputBlob = blobFromImage(color_mat, inScaleFactor,
                                      Size(inpWidth, inpHeight), meanVal, false);
        //Set the network input
        net.setInput(inputBlob, "data");
        Mat detectionMat = net.forward("detection_out");

        //Create message and initialize count
        realsense_perception::DetectedObjectsArray msg;
        int obj_count=0;

        //Loop over all detections
        for(int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            size_t objectClass = max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);

            //Process objects with probability above min threshold
            if(confidence > confidenceThreshold)
            {
                obj_count = obj_count + 1;
                float x_center = detectionMat.at<float>(i, 0) * color_mat.cols;
                float y_center = detectionMat.at<float>(i, 1) * color_mat.rows;
                float width = detectionMat.at<float>(i, 2) * color_mat.cols;
                float height = detectionMat.at<float>(i, 3) * color_mat.rows;
                Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                Rect object(p1, p2);
                
                //Draw bounding box
                Scalar object_roi_color(0, 255, 0);
                rectangle(color_mat, object, object_roi_color);

                //Get Class Label
                String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", int(objectClass));
                String label = format("%s: %.2f", className.c_str(), confidence);
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

                //Calculate center
                auto center = (p1 + p2)*0.5;

                float centerPixel[2]; //  pixel
                float centerPoint[3]; //  point (in 3D)

                centerPixel[0] = center.x;
                centerPixel[1] = center.y;

                // Use central pixel depth
                auto dist = depth_mat.at<double>(center);

                //If pixel depth is noisy (equals zero) get depth from a neighboring pixel
                while (dist == 0)
                {
                    center.y = center.y + 1;
                    dist = depth_mat.at<double>(center);;
                }

                // Get x,y,z coordinates from pixel and depth
                rs2_deproject_pixel_to_point(centerPoint, &intrinsics, centerPixel, dist);

                //String to represent depth of object
                ostringstream ss;
                ss << setprecision(2) << dist << " meters away";
                String conf(ss.str());

                //String to represent coordinates of object
                ostringstream ss1;
                ss1 << " x :" << setprecision(2) << (centerPoint[0]);
                ss1 << " y :" << setprecision(2) << (centerPoint[1]);
                ss1 << " z :" << setprecision(2) << (centerPoint[2]);
                String conf1(ss1.str());

                //Create new object and fill its values
                realsense_perception::DetectedObject obj;
                obj.x = centerPoint[0];
                obj.y = centerPoint[1];
                obj.z = centerPoint[2];
                obj.ClassName = className;
                obj.probability = confidence;

                //Add new object to vector of detected objects
                msg.detectedObjects.push_back(obj);

                // Add label with class name
                rectangle(color_mat, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)),
                          object_roi_color, FILLED);
                putText(color_mat, label, p1 + Point(0, labelSize.height),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));

                //Add label with coordinates of object
                Size labelSize1 = getTextSize(ss1.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(color_mat, Rect(Point(x_center-labelSize1.width/2, y_center +2 *labelSize1.height ),
                    Size(labelSize1.width, labelSize1.height + baseLine)),
                    Scalar(255, 255, 255), FILLED);
                putText(color_mat, ss1.str(), Point(x_center-labelSize1.width/2, y_center + 3 *labelSize1.height ),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));

                //Add label with depth of object
                Size labelSize2 = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(color_mat, Rect(Point(x_center-labelSize1.width/2, y_center +4 *labelSize2.height ),
                    Size(labelSize2.width, labelSize2.height + baseLine)),
                    Scalar(255, 255, 255), FILLED);
                putText(color_mat, ss.str(), Point(x_center-labelSize1.width/2, y_center + 5 *labelSize2.height ),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
        }

        msg.count = obj_count;

        //Publish ROS message with detected objects
        pub.publish(msg);

        //Write frame with detections to output video
//        opvideo.write(color_mat);

        //Display detections
        imshow(window_name, color_mat);

        //Press ESC to quit 
        char c = (char)waitKey(1);
        if( c == 27 ) 
            break;

    }

//    opvideo.release();
//    testvideo.release();
    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << endl;
    return EXIT_FAILURE;
}
catch (const exception& e)
{
    cerr << e.what() << endl;
    return EXIT_FAILURE;
}
