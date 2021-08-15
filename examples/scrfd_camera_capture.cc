/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * License); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * AS IS BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*
 * Copyright (c) 2021, OPEN AI LAB
 * Author: lswang@openailab.com
 */

#define _DEBUG_ 1

#include "algorithm/scrfd.hpp"

#include "util/cmdline.hpp"
#include "util/timer.hpp"


#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <numeric>

#include <opencv2/opencv.hpp>


const float DET_THRESHOLD   =   0.3f;
const float NMS_THRESHOLD   =   0.45f;

const int MODEL_WIDTH       =   160;
const int MODEL_HEIGHT      =   96;

#define MODEL_PATH  "/etc/models/scrfd_2.5g_bnkps.tmfile"

int main(int argc, char* argv[])
{
    printf(">>> Initing display...");

    system("echo panel > /sys/class/display/mode");
    system("fbset -fb /dev/fb0 -g 1024 600 1024 600 16");
    system("echo \"0 0 1023 599\" > /sys/class/graphics/fb0/free_scale_axis");
    system("echo \"0 0 1023 599\" > /sys/class/graphics/fb0/window_axis");
    system("echo 0 > /sys/class/graphics/fb0/free_scale");
    system("echo 1 > /sys/class/graphics/fb0/freescale_mode");
    int fb_fd = open("/dev/fb0", O_RDWR);

    printf("Done\n");

    auto model_path = std::string(MODEL_PATH);
    auto device = std::string("CPU");
    auto score_threshold = DET_THRESHOLD;
    auto iou_threshold = NMS_THRESHOLD;

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

    printf(">>> Loading scrfd graph...");

    cv::Size input_shape(MODEL_WIDTH, MODEL_HEIGHT);

    SCRFD detector;
    auto ret = detector.Load(model_path, input_shape, device.c_str());
    if (!ret)
    {
        fprintf(stderr, "Load model(%s) failed.\n", model_path.c_str());
        return -1;
    }

    std::vector<region> faces;

    while (true)
    {
        vp >> image_ori;

        Timer det_timer;
        detector.Detect(image_ori, faces, score_threshold, iou_threshold);
        det_timer.Stop();

        for (auto& face : faces)
        {
            fprintf(stderr, "%.5f at %.2f %.2f %.2f x %.2f\n", face.confidence, face.box.x, face.box.y, face.box.w, face.box.h);

            // box
            cv::Rect2f rect(face.box.x, face.box.y, face.box.w, face.box.h);

            // draw box
            cv::rectangle(image_ori, rect, cv::Scalar(0, 0, 255), 2);
            std::string box_confidence = "DET: " + std::to_string(face.confidence).substr(0, 5);
            cv::putText(image_ori, box_confidence, rect.tl() + cv::Point2f(5, -10), cv::FONT_HERSHEY_TRIPLEX, 0.6f, cv::Scalar(255, 255, 0));

            cv::circle(image_ori, cv::Point(face.landmark[0].x, face.landmark[0].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_ori, cv::Point(face.landmark[1].x, face.landmark[1].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_ori, cv::Point(face.landmark[2].x, face.landmark[2].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_ori, cv::Point(face.landmark[3].x, face.landmark[3].y), 2, cv::Scalar(255, 255, 0), -1);
            cv::circle(image_ori, cv::Point(face.landmark[4].x, face.landmark[4].y), 2, cv::Scalar(255, 255, 0), -1);
        }

        cv::Mat fb_frame;

        cv::resize(image_ori, fb_frame, cv::Size(1024,600));
        cv::cvtColor(fb_frame, fb_frame, cv::COLOR_BGR2BGR565);
        
        
        lseek(fb_fd, 0, SEEK_SET);
        for(int y=0; y<600; y++){
            // ofs.seekp(y*1024*2);
            write(fb_fd,(char *)fb_frame.ptr(y), 1024 * 2);
        }
    }

    release_tengine();
    close(fb_fd);

    return 0;
}
