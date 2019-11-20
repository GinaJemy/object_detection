// This example is derived from the ssd_mobilenet_object_detection opencv demo
// and adapted to be used with Intel RealSense Cameras
// Please see https://github.com/opencv/opencv/blob/master/LICENSE

#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"
#include <librealsense2/rsutil.h>
#include "ros/ros.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
using namespace cv;
using namespace cv::dnn;
using namespace rs2;

const int inpWidth = 416;        // Width of network's input image
const int inpHeight = 416;       // Height of network's input image
const float WHRatio       = inpWidth / (float)inpHeight;
const float inScaleFactor = 1/255.0;
//        0.007843f;
const float meanVal       = 0.5;
const char* classNames[]  = {"background",
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};
// Initialize the parameters
const float confThreshold = 0.7; // Confidence threshold
const float nmsThreshold = 0.4;  // Non-maximum suppression threshold

std::vector<String> classNamesVec;


class filter_options
{
public:
    filter_options(const std::string name, rs2::filter& filter);
    filter_options(filter_options&& other);
    std::string filter_name;                                   //Friendly name of the filter
    rs2::filter& filter;                                       //The filter in use
};
int main(int argc, char** argv) try
{
	ros::init(argc, argv, "perception");
	ros::NodeHandle n;



    //Load names of classes
    String classesFile = "/home/gina/cam_ws/src/darknet_ros/darknet/data/coco.names";
    std::ifstream ifs(classesFile.c_str());
    std::string line;
    while (getline(ifs, line)) classNamesVec.push_back(line);

// Give the configuration and weight files for the model
    String modelConfiguration = "/home/gina/cam_ws/src/darknet_ros/darknet_ros/yolo_network_config/cfg/yolov2.cfg";
    String modelWeights = "/home/gina/cam_ws/src/darknet_ros/darknet_ros/yolo_network_config/weights/yolov2.weights";

    // Load the network
    Net net = readNetFromDarknet(modelConfiguration, modelWeights);

//    net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    net.setPreferableTarget(DNN_TARGET_CPU);



//    Net net = readNetFromCaffe("/home/gina/camera_proj/MobileNetSSD_deploy.prototxt",
//                               "/home/gina/camera_proj/MobileNetSSD_deploy.caffemodel");

    // Start streaming from Intel RealSense Camera
    pipeline pipe;
    auto config = pipe.start();
    auto profile = config.get_stream(RS2_STREAM_COLOR)
                         .as<video_stream_profile>();
    rs2::align align_to(RS2_STREAM_COLOR);

// Declare filters
//   rs2::decimation_filter dec_filter;  // Decimation - reduces depth frame density
//   rs2::threshold_filter thr_filter;   // Threshold  - removes values outside recommended range
//   rs2::spatial_filter spat_filter;    // Spatial    - edge-preserving spatial smoothing
//   rs2::temporal_filter temp_filter;   // Temporal   - reduces temporal noise

//                                       // Declare disparity transform from depth to disparity and vice versa
//   const std::string disparity_filter_name = "Disparity";
//   rs2::disparity_transform depth_to_disparity(true);
//   rs2::disparity_transform disparity_to_depth(false);

   // Initialize a vector that holds filters and their options
//   std::vector<filter_options> filters;

   // The following order of emplacement will dictate the orders in which filters are applied
//   filters.emplace_back("Decimate", dec_filter);
//   filters.emplace_back("Threshold", thr_filter);
//   filters.emplace_back(disparity_filter_name, depth_to_disparity);

//   filters.emplace_back("Spatial", spat_filter);
//   filters.emplace_back("Temporal", temp_filter);


    auto stream = config.get_stream(RS2_STREAM_DEPTH).as<rs2::video_stream_profile>();
    auto intrinsics = stream.get_intrinsics(); // Calibration data

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

    const auto window_name = "Display Image";
    namedWindow(window_name, WINDOW_AUTOSIZE);

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

        rs2::frame filtered = depth_frame; // Does not copy the frame, only adds a reference

        /* Apply filters.
        The implemented flow of the filters pipeline is in the following order:
        1. apply decimation filter
        2. apply threshold filter
        3. transform the scene into disparity domain
        4. apply spatial filter
        5. apply temporal filter
        6. revert the results back (if step Disparity filter was applied
        to depth domain (each post processing block is optional and can be applied independantly).
        */
//        bool revert_disparity = false;
//        for (auto&& filter : filters)
//        {

//            filtered = filter.filter.process(filtered);
//            if (filter.filter_name == disparity_filter_name)
//            {
//                revert_disparity = true;
//            }

//        }
//        if (revert_disparity)
//        {
//            filtered = disparity_to_depth.process(filtered);
//        }

        // Convert RealSense frame to OpenCV matrix:
        auto color_mat = frame_to_mat(color_frame);
        auto depth_mat = depth_frame_to_meters(pipe, depth_frame);

        // Crop both color and depth frames
        color_mat = color_mat(crop);
        depth_mat = depth_mat(crop);

        Mat inputBlob = blobFromImage(color_mat, inScaleFactor,
                                      Size(inpWidth, inpHeight), meanVal, false); //Convert Mat to batch of images
        net.setInput(inputBlob, "data"); //set the network input

        Mat detectionMat = net.forward("detection_out");

//        net.forward(detectionMat, outBlobNames);

//        Mat detection = net.forward( net.getUnconnectedOutLayersNames()  ); //compute output

//        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());



        float confidenceThreshold = 0.8f;
        for(int i = 0; i < detectionMat.rows; i++)
        {
            const int probability_index = 5;
            const int probability_size = detectionMat.cols - probability_index;
            float *prob_array_ptr = &detectionMat.at<float>(i, probability_index);
            size_t objectClass = std::max_element(prob_array_ptr, prob_array_ptr + probability_size) - prob_array_ptr;
            float confidence = detectionMat.at<float>(i, (int)objectClass + probability_index);


            if(confidence > confidenceThreshold)
            {
                float x_center = detectionMat.at<float>(i, 0) * color_mat.cols;
                float y_center = detectionMat.at<float>(i, 1) * color_mat.rows;
                float width = detectionMat.at<float>(i, 2) * color_mat.cols;
                float height = detectionMat.at<float>(i, 3) * color_mat.rows;
                Point p1(cvRound(x_center - width / 2), cvRound(y_center - height / 2));
                Point p2(cvRound(x_center + width / 2), cvRound(y_center + height / 2));
                Rect object(p1, p2);
                Scalar object_roi_color(0, 255, 0);
                rectangle(color_mat, object, object_roi_color);

                String className = objectClass < classNamesVec.size() ? classNamesVec[objectClass] : cv::format("unknown(%d)", objectClass);
                String label = format("%s: %.2f", className.c_str(), confidence);
                int baseLine = 0;
                Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

//                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

//                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * color_mat.cols);
//                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * color_mat.rows);
//                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * color_mat.cols);
//                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * color_mat.rows);

//                Rect object((int)xLeftBottom, (int)yLeftBottom,
//                            (int)(xRightTop - xLeftBottom),
//                            (int)(yRightTop - yLeftBottom));



//                object = object  & Rect(0, 0, depth_mat.cols, depth_mat.rows);

//                // Calculate mean depth inside the detection region
//                // This is a very naive way to estimate objects depth
//                // but it is intended to demonstrate how one might
//                // use depht data in general
//                Scalar m = mean(depth_mat(object));

                auto center = (p1 + p2)*0.5;


                float centerPixel[2]; //  pixel
                float centerPoint[3]; //  point (in 3D)



                centerPixel[0] = center.x;
                centerPixel[1] = center.y;

                auto dist = depth_mat.at<double>(center);

                // Use central pixel depth
                rs2_deproject_pixel_to_point(centerPoint, &intrinsics, centerPixel, dist);


                std::ostringstream ss;
                ss << std::setprecision(2) << dist << " meters away";
                String conf(ss.str());

                std::ostringstream ss1;
                ss1 << " x :" << (centerPoint[0]);
                ss1 << " y :" << (centerPoint[1]);
                ss1 << " z :" << (centerPoint[2]);
                String conf1(ss1.str());

//                rectangle(color_mat, object, Scalar(0, 255, 0));
//                int baseLine = 0;
//                Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
//                center.x = center.x - labelSize.width / 2;

//                rectangle(color_mat, Rect(Point(center.x, center.y - labelSize.height),
//                    Size(labelSize.width, labelSize.height + baseLine)),
//                    Scalar(255, 255, 255), FILLED);
//                putText(color_mat, ss.str(), center,
//                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
//                auto text2_center = center;

                rectangle(color_mat, Rect(p1, Size(labelSize.width, labelSize.height + baseLine)),
                          object_roi_color, FILLED);
                putText(color_mat, label, p1 + Point(0, labelSize.height),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));

                Size labelSize1 = getTextSize(ss1.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(color_mat, Rect(Point(x_center, y_center +2 *labelSize1.height ),
                    Size(labelSize1.width, labelSize1.height + baseLine)),
                    Scalar(255, 255, 255), FILLED);
                putText(color_mat, ss1.str(), Point(x_center, y_center + 3 *labelSize1.height ),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));

                Size labelSize2 = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                rectangle(color_mat, Rect(Point(x_center, y_center +4 *labelSize2.height ),
                    Size(labelSize2.width, labelSize2.height + baseLine)),
                    Scalar(255, 255, 255), FILLED);
                putText(color_mat, ss.str(), Point(x_center, y_center + 5 *labelSize2.height ),
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
        }

        imshow(window_name, color_mat);
        if (waitKey(1) >= 0) break;
    }

    return EXIT_SUCCESS;
}
catch (const rs2::error & e)
{
    std::cerr << "RealSense error calling " << e.get_failed_function() << "(" << e.get_failed_args() << "):\n    " << e.what() << std::endl;
    return EXIT_FAILURE;
}
catch (const std::exception& e)
{
    std::cerr << e.what() << std::endl;
    return EXIT_FAILURE;
}

filter_options::filter_options(const std::string name, rs2::filter& flt) :
    filter_name(name),
    filter(flt)
{
    const std::array<rs2_option, 5> possible_filter_options = {
        RS2_OPTION_FILTER_MAGNITUDE,
        RS2_OPTION_FILTER_SMOOTH_ALPHA,
        RS2_OPTION_MIN_DISTANCE,
        RS2_OPTION_MAX_DISTANCE,
        RS2_OPTION_FILTER_SMOOTH_DELTA
    };

    //Go over each filter option and create a slider for it
    for (rs2_option opt : possible_filter_options)
    {
        if (flt.supports(opt))
        {
            rs2::option_range range = flt.get_option_range(opt);
            std::string opt_name = flt.get_option_name(opt);
            std::string prefix = "Filter ";
        }
    }
}

filter_options::filter_options(filter_options&& other) :
    filter_name(std::move(other.filter_name)),
    filter(other.filter)
{
}
