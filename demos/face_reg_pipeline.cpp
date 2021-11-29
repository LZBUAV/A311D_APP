#include "algorithm/scrfd.hpp"
#include "algorithm/arcface.hpp"

#include "utilities/cmdline.hpp"
#include "utilities/distance.hpp"
#include "utilities/timer.hpp"
#include "utilities/affine.hpp"
#include "utilities/region.hpp"

#include "common/common.h"

#include "tengine/c_api.h"
#include "common/tengine_operations.h"

#include <opencv2/opencv.hpp>

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <vector>
#include <tuple>
#include <algorithm>
#include <numeric>

const float DET_THRESHOLD   =   0.30f;
const float NMS_THRESHOLD   =   0.45f;

const int MODEL_WIDTH       =   160;
const int MODEL_HEIGHT      =   96;

void show_usage()
{
    std::cout << "-h, help" << std::endl;
    std::cout << "-d, the face detect model path. default: /lzb/models/scrfd_2.5g_bnkps.tmfile" << std::endl;
    std::cout << "-r, the face recongnition model path. default: /lzb/models/mobilefacenet_simply.tmfile" << std::endl;
    std::cout << "-f, the face feature bank path. default: /lzb/facebank/feature_average.txt" << std::endl;
    std::cout << "-a, the new added face feature store path." << std::endl;
    std::cout << "uasge1: ./face_reg_pipeline" << std::endl;
    std::cout << "uasge2: ./face_reg_pipeline -u 1" << std::endl;
    std::cout << "uasge3: ./face_reg_pipeline -d /lzb/models/scrfd_2.5g_bnkps.tmfile -r /lzb/models/mobilefacenet_simply.tmfile -f /lzb/facebank/feature_average.txt" << std::endl;
    std::cout << "uasge4: ./face_reg_pipeline -d /lzb/models/scrfd_2.5g_bnkps.tmfile -r /lzb/models/mobilefacenet_simply.tmfile -f /lzb/facebank/feature_average.txt" << std::endl;
    std::cout << "uasge5: ./face_reg_pipeline -d /lzb/models/scrfd_2.5g_bnkps.tmfile -r /lzb/models/mobilefacenet_simply.tmfile -f /lzb/facebank/feature_average.txt -a /lzb/facebank/new_feature.txt" << std::endl;
    std::cout << "uasge6: ./face_reg_pipeline -a /lzb/facebank/new_feature.txt" << std::endl;
}

int main(int argc, char* argv[])
{

    cmdline::parser cmd;

    cmd.add<std::string>("detect_model", 'd', "detection model file", false, "/lzb/models/scrfd_2.5g_bnkps.tmfile");
    cmd.add<std::string>("recongnition_model", 'r', "recongnition_model model file", false, "/lzb/models/mobilefacenet_simply.tmfile");
    cmd.add<std::string>("face_bank_path", 'f', "face bank file", false, "/lzb/facebank/feature_average.txt");
    cmd.add<std::string>("add_face", 'a', "new feature store path", false, "");
    cmd.add<std::string>("usage", 'u', "usage", false, "");

    cmd.parse_check(argc, argv);

    auto det_model_path = cmd.get<std::string>("detect_model");
    auto reg_model_path = cmd.get<std::string>("recongnition_model");
    auto face_bank_path = cmd.get<std::string>("face_bank_path");
    auto add_face_path = cmd.get<std::string>("add_face");
    auto usage = cmd.get<std::string>("usage");

    if(usage != "")
    {
        show_usage();
        return -1;
    }

    auto device = std::string("CPU");
    auto score_threshold = DET_THRESHOLD;
    auto iou_threshold = NMS_THRESHOLD;

    printf(">>> Initing display...");

    system("echo panel > /sys/class/display/mode");
    system("fbset -fb /dev/fb0 -g 1024 600 1024 600 16");
    system("echo \"0 0 1023 599\" > /sys/class/graphics/fb0/free_scale_axis");
    system("echo \"0 0 1023 599\" > /sys/class/graphics/fb0/window_axis");
    system("echo 0 > /sys/class/graphics/fb0/free_scale");
    system("echo 1 > /sys/class/graphics/fb0/freescale_mode");
    int fb_fd = open("/dev/fb0", O_RDWR);

    printf("Done\n");

    printf(">>> Initing runtime...");
    init_tengine();
    printf("Done\n");

    printf(">>> Initing camera...");
    cv::VideoCapture vp("v4l2src ! videoconvert ! video/x-raw,format=BGR,width=1920,height=1080 ! appsink", cv::CAP_GSTREAMER);
    if (!vp.isOpened())
    {
        printf("Failed\n");
        return -1;
    }
    printf("Done\n");

    cv::Mat image_ori;
    vp >> image_ori;

    if (image_ori.empty())
    {
        fprintf(stderr, "Reading image from camera was failed.\n");
        return -1;
    }

    printf(">>> Loading scrfd graph...\n");

    cv::Size input_shape(MODEL_WIDTH, MODEL_HEIGHT);
    SCRFD detector;
    auto ret = detector.Load(det_model_path, input_shape, device.c_str());
    if (!ret)
    {
        fprintf(stderr, "Load model(%s) failed.\n", det_model_path.c_str());
        return -1;
    }
    std::vector<Face> faces;
    
    recognition reg;
    ret = reg.load(reg_model_path, device);
    if (!ret)
    {
        fprintf(stderr, "Load verify model(%s) failed.\n", reg_model_path.c_str());
        return -1;
    }
    std::vector<float> feature;

    //获取注册过的人脸特征
    std::vector<std::tuple<std::string, std::vector<float>>> feature_all_person;
    
    std::ifstream feature_input;
    feature_input.open(face_bank_path, std::ios::in);
    if(!feature_input.is_open())
    {
        std::cout << "打开文件失败" << std::endl;
        return -1;
    }

    std::string feature_line;
    while(getline(feature_input, feature_line))
    {
        std::tuple<std::string, std::vector<float>> per_person;
        std::istringstream read_fearure(feature_line);
        std::string person_name, feature_col;
        std::vector<float>feature_per_person; 
        getline(read_fearure, person_name, ',');
        for(int i = 1; i < 513; ++i)
        {
            getline(read_fearure, feature_col, ',');
            feature_per_person.push_back(std::stof(feature_col));
        }
        feature_all_person.push_back(std::make_tuple(person_name, feature_per_person));
    }

    //测试facebank特征读取时用的
    // std::cout << feature_all_person.size() << std::endl;
    // std::cout << std::get<0>(feature_all_person[0]) << std::endl;
    // std::vector<float> a = std::get<1>(feature_all_person[0]);
    // for(auto i : a)
    // {
    //     std::cout << i << ",";
    // }
    // std::cout << std::endl;

    feature_input.close();
    


    //获取特征时用
    std::ofstream feature_out;
    if(add_face_path != "")
    {
        feature_out.open(add_face_path, std::ios::out | std::ios::app);
        if(!feature_out.is_open())
        {
            std::cout << "/lzb/facebank/new_feature.txt打开失败" << std::endl;
            return -1;
        }
        std::cout << "打开/lzb/facebank/new_feature.txt文件成功" << std::endl;
    }

    float FPS = 0.0;

    while (true)
    {
        Timer total_timer;

        vp >> image_ori;

        detector.Detect(image_ori, faces, score_threshold, iou_threshold);

        for (auto& face : faces)
        {
            fprintf(stderr, "confidence: %.5f at %.2f %.2f %.2f %.2f, FPS: %.6f\n", face.confidence, face.box.x, face.box.y, face.box.width, face.box.height, FPS);

            ret = reg.get_feature(image_ori, face.landmark, feature);
            if (!ret)
            {
                fprintf(stderr, "Get attack feature was failed.\n");
                return -1;
            }
            norm_feature(feature);

            //添加新 人特征
            if(add_face_path != "")
            {
                for(auto i : feature)
                {
                    feature_out << i << ',';
                }
                feature_out << std::endl;
            }

            unsigned int max_index = 0;
            float max_distance_cosin = 0.0;
            for(int i = 0; i < feature_all_person.size(); ++i)
            {
                float distance_cosin = cos_distance(feature, std::get<1>(feature_all_person[i]));
                if(distance_cosin >= max_distance_cosin)
                {
                    max_distance_cosin = distance_cosin;
                    max_index = i;
                }
            }
            // std::cout << "max_distance" << " " << max_distance_cosin << std::endl;

            // box
            cv::Rect2f rect(face.box.x, face.box.y, face.box.width, face.box.height);

            // draw box
            cv::rectangle(image_ori, rect, cv::Scalar(0, 0, 255), 2);

            if(max_distance_cosin > 0.5)
            { 
                std::string box_confidence = std::get<0>(feature_all_person[max_index]) + ": " + std::to_string(face.confidence).substr(0, 5);
                cv::putText(image_ori, box_confidence, rect.tl() + cv::Point2f(5, -15), cv::FONT_HERSHEY_TRIPLEX, 1.5f, cv::Scalar(0, 0, 255), 2, 4);
            }
            else
            {
                std::string box_confidence = "Unknow: " + std::to_string(face.confidence).substr(0, 5);
                cv::putText(image_ori, box_confidence, rect.tl() + cv::Point2f(5, -15), cv::FONT_HERSHEY_TRIPLEX, 1.5f, cv::Scalar(0, 0, 255), 2, 4);
            }

            cv::circle(image_ori, cv::Point(face.landmark[0].x, face.landmark[0].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_ori, cv::Point(face.landmark[1].x, face.landmark[1].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_ori, cv::Point(face.landmark[2].x, face.landmark[2].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_ori, cv::Point(face.landmark[3].x, face.landmark[3].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_ori, cv::Point(face.landmark[4].x, face.landmark[4].y), 2, cv::Scalar(255, 255, 0), -1);

        }

        cv::putText(image_ori, "FPS: " + std::to_string(FPS).substr(0, 6), cv::Point2f(30, 50), cv::FONT_HERSHEY_TRIPLEX, 1.5f, cv::Scalar(0, 0, 255), 2, 4);

        cv::Mat fb_frame;

        cv::resize(image_ori, fb_frame, cv::Size(1024,600));
        cv::cvtColor(fb_frame, fb_frame, cv::COLOR_BGR2BGR565);
        
        
        lseek(fb_fd, 0, SEEK_SET);
        for(int y=0; y<600; y++){
            // ofs.seekp(y*1024*2);
            write(fb_fd,(char *)fb_frame.ptr(y), 1024 * 2);
        }

        FPS = 1000.0/total_timer.Cost();

    }
    
    if(add_face_path != "")
    {
        feature_out.close();
    }
    vp.release();
    release_tengine();
    close(fb_fd);
    
    return 0;
}
