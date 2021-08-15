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
 * Author: 942002795@qq.com
 * Update: xwwang@openailab.com
 */

#include <stdio.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <math.h>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <stdlib.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "common.h"
#include "tengine/c_api.h"
#include "tengine_operations.h"

struct Object
{
    cv::Rect_<float> rect;
    int label;
    float prob;
};

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static inline float intersection_area(const Object& a, const Object& b)
{
    cv::Rect_<float> inter = a.rect & b.rect;
    return inter.area();
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects, int left, int right)
{
    int i = left;
    int j = right;
    float p = faceobjects[(left + right) / 2].prob;

    while (i <= j)
    {
        while (faceobjects[i].prob > p)
            i++;

        while (faceobjects[j].prob < p)
            j--;

        if (i <= j)
        {
            // swap
            std::swap(faceobjects[i], faceobjects[j]);

            i++;
            j--;
        }
    }

#pragma omp parallel sections
    {
#pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
#pragma omp section
        {
            if (i < right) qsort_descent_inplace(faceobjects, i, right);
        }
    }
}

static void qsort_descent_inplace(std::vector<Object>& faceobjects)
{
    if (faceobjects.empty())
        return;

    qsort_descent_inplace(faceobjects, 0, faceobjects.size() - 1);
}

static void nms_sorted_bboxes(const std::vector<Object>& faceobjects, std::vector<int>& picked, float nms_threshold)
{
    picked.clear();

    const int n = faceobjects.size();

    std::vector<float> areas(n);
    for (int i = 0; i < n; i++)
    {
        areas[i] = faceobjects[i].rect.area();
    }

    for (int i = 0; i < n; i++)
    {
        const Object& a = faceobjects[i];

        int keep = 1;
        for (int j = 0; j < (int)picked.size(); j++)
        {
            const Object& b = faceobjects[picked[j]];

            // intersection over union
            float inter_area = intersection_area(a, b);
            float union_area = areas[i] + areas[picked[j]] - inter_area;
            // float IoU = inter_area / union_area
            if (inter_area / union_area > nms_threshold)
                keep = 0;
        }

        if (keep)
            picked.push_back(i);
    }
}

void get_input_data_yolov4_uint8(cv::Mat& img_in, uint8_t * input_data, int img_h, int img_w, const float* mean, const float* scale,
                                 float input_scale, int zero_point)
{
    /* resize process */
    cv::Mat img;
    cv::resize(img_in, img, cv::Size(img_w, img_h));
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    img.convertTo(img, CV_32FC3);
    float* img_data = (float* )img.data;

    /* nhwc to nchw */
    for (int h = 0; h < img_h; h++)
    {   for (int w = 0; w < img_w; w++)
        {
            for (int c = 0; c < 3; c++)
            {
                int in_index  = h * img_w * 3 + w * 3 + c;
                int out_index = c * img_h * img_w + h * img_w + w;
                float input_fp32 = (img_data[in_index] - mean[c]) * scale[c];

                /* quant to uint8 */
                int udata = (round)(input_fp32 / input_scale + ( float )zero_point);
                if (udata > 255)
                    udata = 255;
                else if (udata < 0)
                    udata = 0;

                input_data[out_index] = udata;
            }
        }
    }
}

static void generate_proposals(int stride,  const float* feat, float prob_threshold, std::vector<Object>& objects)
{
    static float anchors[12] = {10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319};
    int anchor_num = 3;
    int feat_w = 416 / stride;
    int feat_h = 416 / stride;
    int cls_num = 80;
    int anchor_group = 0;
    if(stride == 16)
        anchor_group = 1;
    if(stride == 32)
        anchor_group = 2;

    for (int h = 0; h <= feat_h - 1; h++)
    {
        for (int w = 0; w <= feat_w - 1; w++)
        {
            for (int anchor = 0; anchor <= anchor_num - 1; anchor++)
            {
                int class_index = 0;
                float class_score = -FLT_MAX;
                int channel_size = feat_h * feat_w;
                for (int s = 0; s <= cls_num - 1; s++)
                {
                    int score_index = anchor * (cls_num + 5) * channel_size + feat_w * h + w + (s + 5) * channel_size;
                    float score = feat[score_index];
                    if(score > class_score)
                    {
                        class_index = s;
                        class_score = score;
                    }
                }
                float box_score = feat[anchor * (cls_num + 5) * channel_size + feat_w * h + w + 4 * channel_size];
                float final_score = sigmoid(box_score) * sigmoid(class_score);
                if(final_score >= prob_threshold)
                {
                    int dx_index = anchor * (cls_num + 5) * channel_size + feat_w * h + w + 0 * channel_size;
                    int dy_index = anchor * (cls_num + 5) * channel_size + feat_w * h + w + 1 * channel_size;
                    int dw_index = anchor * (cls_num + 5) * channel_size + feat_w * h + w + 2 * channel_size;
                    int dh_index = anchor * (cls_num + 5) * channel_size + feat_w * h + w + 3 * channel_size;

                    float dx = sigmoid(feat[dx_index]);
                    float dy = sigmoid(feat[dy_index]);

                    float dw = feat[dw_index];
                    float dh = feat[dh_index];

                    float anchor_w = anchors[(anchor_group - 1) * 6 + anchor * 2 + 0];
                    float anchor_h = anchors[(anchor_group - 1) * 6 + anchor * 2 + 1];

                    float pred_x = (w + dx) * stride;
                    float pred_y = (h + dy) * stride;
                    float pred_w = exp(dw) * anchor_w ;
                    float pred_h = exp(dh) * anchor_h ;

                    float x0 = (pred_x - pred_w * 0.5f);
                    float y0 = (pred_y - pred_h * 0.5f);
                    float x1 = (pred_x + pred_w * 0.5f);
                    float y1 = (pred_y + pred_h * 0.5f);

                    Object obj;
                    obj.rect.x = x0;
                    obj.rect.y = y0;
                    obj.rect.width = x1 - x0;
                    obj.rect.height = y1 - y0;
                    obj.label = class_index;
                    obj.prob = final_score;
                    objects.push_back(obj); 
                }
            }
        }
    }
}

static void draw_objects(const cv::Mat& bgr, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    cv::Mat image = bgr.clone();

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

        fprintf(stderr, "%2d: %3.0f%%, [%4.0f, %4.0f, %4.0f, %4.0f], %s\n", obj.label, obj.prob * 100, obj.rect.x,
                obj.rect.y, obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, class_names[obj.label]);

        cv::rectangle(image, obj.rect, cv::Scalar(255, 0, 0));

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 1, 2, &baseLine);

        int x = obj.rect.x;
        int y = obj.rect.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > image.cols)
            x = image.cols - label_size.width;

        cv::rectangle(image, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)),
                      cv::Scalar(255, 255, 255), -1);

        cv::putText(image, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 1,
                    cv::Scalar(0, 0, 0));
    }
    cv::Mat fb_frame;

    cv::resize(image, fb_frame, cv::Size(1024,600));
    cv::cvtColor(fb_frame, fb_frame, cv::COLOR_BGR2BGR565);
    
    int fd = open("/dev/fb0", O_RDWR);

    for(int y=0; y<600; y++){
        // ofs.seekp(y*1024*2);
        write(fd,(char *)fb_frame.ptr(y), 1024 * 2);
    }

    close(fd);
}


int main(int argc, char* argv[])
{
    printf("YoloV4 Almogic NPU Demo.\n");


    printf(">>> Initing display...");
    system("echo panel > /sys/class/display/mode");
    system("fbset -fb /dev/fb0 -g 1024 600 1024 600 16");
    system("echo \"0 0 1023 599\" > /sys/class/graphics/fb0/free_scale_axis");
    system("echo \"0 0 1023 599\" > /sys/class/graphics/fb0/window_axis");
    system("echo 0 > /sys/class/graphics/fb0/free_scale");
    system("echo 1 > /sys/class/graphics/fb0/freescale_mode");
    printf("Done\n");

    int img_h = 416;
    int img_w = 416;
    int img_c = 3;
    const float mean[3] = {0, 0, 0};
    const float scale[3] = {0.003921, 0.003921, 0.003921};

    printf(">>> Initing camera...");
    cv::VideoCapture cap("v4l2src ! videoconvert ! video/x-raw,format=BGR,width=1920,height=1080 ! appsink", cv::CAP_GSTREAMER);
    if (!cap.isOpened()) {
        printf("Failed\n");
    }
    printf("Done\n");

    /* set runtime options */
    struct options opt;
    opt.num_thread = 6;
    opt.cluster = TENGINE_CLUSTER_ALL;
    opt.precision = TENGINE_MODE_UINT8;
    opt.affinity = 255;

    /* inital tengine */
    printf(">>> Initing runtime...");
    if (init_tengine() != 0)
    {
        printf("Failed\n");
        return -1;
    }
    printf("Done\n");

    printf("Runtime library version: %s\n", get_tengine_version());

    /* create VeriSilicon TIM-VX backend */
    printf(">>> Initing TIM-VX backend...");
    context_t timvx_context = create_context("timvx", 1);
    int rtt = set_context_device(timvx_context, "TIMVX", nullptr, 0);
    if (0 > rtt)
    {
        printf("Failed\n");
        return -1;
    }
    printf("Done\n");

    /* create graph, load tengine model xxx.tmfile */
    printf(">>> Loading graph...");
    graph_t graph = create_graph(timvx_context, "tengine", "/etc/models/yolov4-tiny_uint8.tmfile");
    if (graph == nullptr)
    {
        printf("Failed\n");
        return -1;
    }
    printf("Done\n");

    int img_size = img_h * img_w * img_c;
    int dims[] = {1, 3, img_h, img_w};
    std::vector<uint8_t> input_data(img_size);

    printf(">>> Get graph input tensor...");
    tensor_t input_tensor = get_graph_input_tensor(graph, 0, 0);
    if (input_tensor == nullptr)
    {
        printf("Failed\n");
        fprintf(stderr, "Get input tensor failed\n");
        return -1;
    }
    printf("Done\n");
    
    printf(">>> Set graph input tensor shape...");
    if (set_tensor_shape(input_tensor, dims, 4) < 0)
    {
        printf("Failed\n");
        fprintf(stderr, "Set input tensor shape failed\n");
        return -1;
    }
    printf("Done\n");

    printf(">>> Set tensor buffer...");
    if (set_tensor_buffer(input_tensor, input_data.data(), img_size) < 0)
    {
        printf("Failed\n");
        fprintf(stderr, "Set input tensor buffer failed\n");
        return -1;
    }
    printf("Done\n");

    /* prerun graph, set work options(num_thread, cluster, precision) */
    printf(">>> Config graph...");
    if (prerun_graph_multithread(graph, opt) < 0)
    {
        printf("Failed\n");
        fprintf(stderr, "Prerun multithread graph failed.\n");
        return -1;
    }
    printf("Done\n");

    /* prepare process input data, set the data mem to input tensor */
    float input_scale = 0.f;
    int input_zero_point = 0;
    
    get_tensor_quant_param(input_tensor, &input_scale, &input_zero_point, 1);
    
    printf("Start running network!\n");

    while(1){
        double start = get_current_time();

        cv::Mat img;
        cap.read(img);

        get_input_data_yolov4_uint8(img, input_data.data(), img_h, img_w, mean, scale, input_scale, input_zero_point);
        if (run_graph(graph, 1) < 0)
        {
            fprintf(stderr, "Run graph failed\n");
            return -1;
        }

        /* dequant output data */
        tensor_t p16_output = get_graph_output_tensor(graph, 1, 0);
        tensor_t p32_output = get_graph_output_tensor(graph, 0, 0);

        float p16_scale = 0.f;
        float p32_scale = 0.f;
        int p16_zero_point = 0;
        int p32_zero_point = 0;

        get_tensor_quant_param(p16_output, &p16_scale, &p16_zero_point, 1);
        get_tensor_quant_param(p32_output, &p32_scale, &p32_zero_point, 1);

        int p16_count = get_tensor_buffer_size(p16_output) / sizeof(uint8_t);
        int p32_count = get_tensor_buffer_size(p32_output) / sizeof(uint8_t);

        uint8_t* p16_data_u8 = ( uint8_t* )get_tensor_buffer(p16_output);
        uint8_t* p32_data_u8 = ( uint8_t* )get_tensor_buffer(p32_output);

        std::vector<float> p16_data(p16_count);
        std::vector<float> p32_data(p32_count);

        for (int c = 0; c < p16_count; c++)
        {
            p16_data[c] = (( float )p16_data_u8[c] - ( float )p16_zero_point) * p16_scale;
        }

        for (int c = 0; c < p32_count; c++)
        {
            p32_data[c] = (( float )p32_data_u8[c] - ( float )p32_zero_point) * p32_scale;
        }

        /* postprocess */
        const float prob_threshold = 0.45f;
        const float nms_threshold = 0.25f;

        std::vector<Object> proposals;
        std::vector<Object> objects16;
        std::vector<Object> objects32;
        std::vector<Object> objects;

        generate_proposals(32, p32_data.data(), prob_threshold, objects32);
        proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        generate_proposals(16, p16_data.data(), prob_threshold, objects16);
        proposals.insert(proposals.end(), objects16.begin(), objects16.end());

        qsort_descent_inplace(proposals);

        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        /* yolov4 tiny draw the result */
        int raw_h = img.rows;
        int raw_w = img.cols;

        float ratio_x = (float)raw_w / img_w;
        float ratio_y = (float)raw_h / img_h;

        int count = picked.size();
        fprintf(stdout, "Detect obj count: %d",count);

        objects.resize(count);
        for (int i = 0; i < count; i++)
        {
            objects[i] = proposals[picked[i]];
            float x0 = (objects[i].rect.x);
            float y0 = (objects[i].rect.y);
            float x1 = (objects[i].rect.x + objects[i].rect.width);
            float y1 = (objects[i].rect.y + objects[i].rect.height);

            x0 = x0 * ratio_x;
            y0 = y0 * ratio_y;
            x1 = x1 * ratio_x;
            y1 = y1 * ratio_y;

            x0 = std::max(std::min(x0, (float)(raw_w - 1)), 0.f);
            y0 = std::max(std::min(y0, (float)(raw_h - 1)), 0.f);
            x1 = std::max(std::min(x1, (float)(raw_w - 1)), 0.f);
            y1 = std::max(std::min(y1, (float)(raw_h - 1)), 0.f);

            objects[i].rect.x = x0;
            objects[i].rect.y = y0;
            objects[i].rect.width = x1 - x0;
            objects[i].rect.height = y1 - y0;
        }

        draw_objects(img, objects);

        double end = get_current_time();
        double cur = end - start;

        fprintf(stdout, ", Runtime Latency: %.3f\n", cur);
    }
    

    /* release tengine */
    postrun_graph(graph);
    destroy_graph(graph);
    release_tengine();
}
