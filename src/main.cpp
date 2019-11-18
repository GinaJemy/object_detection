// This example is derived from the ssd_mobilenet_object_detection opencv demo
// and adapted to be used with Intel RealSense Cameras
// Please see https://github.com/opencv/opencv/blob/master/LICENSE

#include <opencv2/dnn.hpp>
#include <librealsense2/rs.hpp>
#include "cv-helpers.hpp"
#include <librealsense2/rsutil.h>
#include "ros/ros.h"

const size_t inWidth      = 350;
const size_t inHeight     = 350;
const float WHRatio       = inWidth / (float)inHeight;
const float inScaleFactor = 0.007843f;
const float meanVal       = 127.5;
const char* classNames[]  = {"background",
                             "aeroplane", "bicycle", "bird", "boat",
                             "bottle", "bus", "car", "cat", "chair",
                             "cow", "diningtable", "dog", "horse",
                             "motorbike", "person", "pottedplant",
                             "sheep", "sofa", "train", "tvmonitor"};
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

    using namespace cv;
    using namespace cv::dnn;
    using namespace rs2;

    Net net = readNetFromCaffe("/home/gina/camera_proj/MobileNetSSD_deploy.prototxt",
                               "/home/gina/camera_proj/MobileNetSSD_deploy.caffemodel");

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

    while (getWindowProperty(window_name, WND_PROP_AUTOSIZE) >= 0 && ros::ok())
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

        Mat inputBlob = blobFromImage(color_mat, inScaleFactor,
                                      Size(inWidth, inHeight), meanVal, false); //Convert Mat to batch of images
        net.setInput(inputBlob, "data"); //set the network input
        Mat detection = net.forward("detection_out"); //compute output

        Mat detectionMat(detection.size[2], detection.size[3], CV_32F, detection.ptr<float>());

        // Crop both color and depth frames
//        color_mat = color_mat(crop);
//        depth_mat = depth_mat(crop);

        float confidenceThreshold = 0.8f;
        for(int i = 0; i < detectionMat.rows; i++)
        {
            float confidence = detectionMat.at<float>(i, 2);

            if(confidence > confidenceThreshold)
            {
                size_t objectClass = (size_t)(detectionMat.at<float>(i, 1));

                int xLeftBottom = static_cast<int>(detectionMat.at<float>(i, 3) * color_mat.cols);
                int yLeftBottom = static_cast<int>(detectionMat.at<float>(i, 4) * color_mat.rows);
                int xRightTop = static_cast<int>(detectionMat.at<float>(i, 5) * color_mat.cols);
                int yRightTop = static_cast<int>(detectionMat.at<float>(i, 6) * color_mat.rows);

                Rect object((int)xLeftBottom, (int)yLeftBottom,
                            (int)(xRightTop - xLeftBottom),
                            (int)(yRightTop - yLeftBottom));



                object = object  & Rect(0, 0, depth_mat.cols, depth_mat.rows);

                // Calculate mean depth inside the detection region
                // This is a very naive way to estimate objects depth
                // but it is intended to demonstrate how one might
                // use depht data in general
                Scalar m = mean(depth_mat(object));

                auto center = (object.br() + object.tl())*0.5;


                float centerPixel[2]; //  pixel
                float centerPoint[3]; //  point (in 3D)

                float centerPointm[3];

                centerPixel[0] = center.x;
                centerPixel[1] = center.y;

                auto dist = depth_mat.at<double>(center);

                // Use central pixel depth
                rs2_deproject_pixel_to_point(centerPoint, &intrinsics, centerPixel, dist);

                //Use mean depth instead of depth of the central pixel
                rs2_deproject_pixel_to_point(centerPointm, &intrinsics, centerPixel, m[0]);

                std::ostringstream ss;
                ss << classNames[objectClass] << " ";
                ss << "mean" << std::setprecision(2) << m[0] << " meters away";
                ss << " center " << std::setprecision(2) << dist << " meters away";
                String conf(ss.str());

                std::ostringstream ss1;
                ss1 << " x :" << (centerPoint[0]);
                ss1 << " y :" << (centerPoint[1]);
                ss1 << " z :" << (centerPoint[2]);
                String conf1(ss1.str());

                rectangle(color_mat, object, Scalar(0, 255, 0));
                int baseLine = 0;
                Size labelSize = getTextSize(ss.str(), FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                center.x = center.x - labelSize.width / 2;

                rectangle(color_mat, Rect(Point(center.x, center.y - labelSize.height),
                    Size(labelSize.width, labelSize.height + baseLine)),
                    Scalar(255, 255, 255), FILLED);
                putText(color_mat, ss.str(), center,
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
                auto text2_center = center;
                text2_center.y = text2_center.y + 3* labelSize.height;
                rectangle(color_mat, Rect(Point(center.x, center.y +2 *labelSize.height ),
                    Size(labelSize.width, labelSize.height + baseLine)),
                    Scalar(255, 255, 255), FILLED);
                putText(color_mat, ss1.str(), text2_center,
                        FONT_HERSHEY_SIMPLEX, 0.5, Scalar(0,0,0));
            }
        }

        imshow(window_name, color_mat);
        if (waitKey(1) >= 0) break;
        ros::spinOnce();
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
