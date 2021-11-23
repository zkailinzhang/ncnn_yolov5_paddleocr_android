// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2021 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#ifndef NANODET_H
#define NANODET_H

#include <opencv2/core/core.hpp>

#include <net.h>

#include "layer.h"
#include "net.h"
#include "benchmark.h"
#include "mat.h"

#include <opencv2/opencv.hpp>
#include <iostream>
#include <iostream>
#include <fstream>
#include <string>
#include <sstream>
#include <iomanip>
#include <stdexcept>
#include<opencv2/core/core.hpp>
#include <opencv2/core/utility.hpp>
#include<opencv2/opencv.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>

#include<math.h>

#include <cmath>
#include <cmath>
#include <algorithm>
#include <functional>
#include <array>
#include <iostream>
#include <numeric>
#include <vector>
#include <exception>

//paddleocr
#include "common.h"


struct Object
{
    float x;
    float y;
    float w;
    float h;
    int label;
    float prob;
    float value;
};


class NanoDet
{
public:
    NanoDet();

    //加载模型
    int load(const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);
    //都是加载模型，多一个参数，yolo5用这个
    int load(AAssetManager* mgr, const char* modeltype, int target_size, const float* mean_vals, const float* norm_vals, bool use_gpu = false);

    //int detect(const cv::Mat& rgb, std::vector<Object>& objects, float prob_threshold = 0.4f, float nms_threshold = 0.5f);
    int detect(const cv::Mat& rgb, std::vector<Object>& objects);


    int draw(cv::Mat& rgb, const std::vector<Object>& objects);
    int draw(cv::Mat& rgb, const Object &obj,float value);
    int draw(cv::Mat& rgb, const Object &obj);

    float detectvalue(cv::Mat& rgb,const Object& obj,const float min_value,const float max_value);
    float polardetect(cv::Mat& rgb,const Object& obj);
    float polardetect(cv::Mat& rgb,const Object& obj,const float min_value,const float max_value);
    float detectthree(cv::Mat& rgb,const Object& obj,const float min_value,const float max_value);
    float detectfour(cv::Mat& rgb,const Object& obj,const float min_value,const float max_value);

    void maxtwosqens(std::vector<int> &s,int& sta,int& end);

    cv::Vec3d avg_circles(std::vector<cv::Vec3f> circles, int b);
    float getDist_P2L(cv::Point2f pointP, cv::Point2f pointA, cv::Point2f pointB);
    float dist_2_pts(int x1, int y1, int x2, int y2);
    cv::Mat region_of_interest(cv::Mat &img,std::vector<std::vector<cv::Point>> &vertices);

    float detect_ocr_angle(cv::Mat& rgb,const Object& obj);
    //float detect_ocr_angle(cv::Mat& rgb,const Object& obj,const float min_value,const float max_value);

private:
    ncnn::Net nanodet;
    ncnn::Net dbNet;
    //在这添加两个模型，还是再写个类，

    int target_size;
    float mean_vals[3];
    float norm_vals[3];
    ncnn::UnlockedPoolAllocator blob_pool_allocator;
    ncnn::PoolAllocator workspace_pool_allocator;

    std::vector<float> result_value_list;
};

#endif // NANODET_H
