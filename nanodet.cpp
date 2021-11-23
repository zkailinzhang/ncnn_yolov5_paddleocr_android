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

#include "nanodet.h"

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include "cpu.h"

static inline float intersection_area(const Object& a, const Object& b)
{
    if (a.x > b.x + b.w || a.x + a.w < b.x || a.y > b.y + b.h || a.y + a.h < b.y)
    {
        // no intersection
        return 0.f;
    }

    float inter_width = std::min(a.x + a.w, b.x + b.w) - std::max(a.x, b.x);
    float inter_height = std::min(a.y + a.h, b.y + b.h) - std::max(a.y, b.y);

    return inter_width * inter_height;
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

    //     #pragma omp parallel sections
    {
        //         #pragma omp section
        {
            if (left < j) qsort_descent_inplace(faceobjects, left, j);
        }
        //         #pragma omp section
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
        areas[i] = faceobjects[i].w * faceobjects[i].h;
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

static inline float sigmoid(float x)
{
    return static_cast<float>(1.f / (1.f + exp(-x)));
}

static void generate_proposals(const ncnn::Mat& anchors, int stride, const ncnn::Mat& in_pad, const ncnn::Mat& feat_blob, float prob_threshold, std::vector<Object>& objects)
{
    const int num_grid = feat_blob.h;

    int num_grid_x;
    int num_grid_y;
    if (in_pad.w > in_pad.h)
    {
        num_grid_x = in_pad.w / stride;
        num_grid_y = num_grid / num_grid_x;
    }
    else
    {
        num_grid_y = in_pad.h / stride;
        num_grid_x = num_grid / num_grid_y;
    }

    const int num_class = feat_blob.w - 5;

    const int num_anchors = anchors.w / 2;

    for (int q = 0; q < num_anchors; q++)
    {
        const float anchor_w = anchors[q * 2];
        const float anchor_h = anchors[q * 2 + 1];

        const ncnn::Mat feat = feat_blob.channel(q);

        for (int i = 0; i < num_grid_y; i++)
        {
            for (int j = 0; j < num_grid_x; j++)
            {
                const float* featptr = feat.row(i * num_grid_x + j);

                // find class index with max class score
                int class_index = 0;
                float class_score = -FLT_MAX;
                for (int k = 0; k < num_class; k++)
                {
                    float score = featptr[5 + k];
                    if (score > class_score)
                    {
                        class_index = k;
                        class_score = score;
                    }
                }

                float box_score = featptr[4];

                float confidence = sigmoid(box_score) * sigmoid(class_score);

                if (confidence >= prob_threshold)
                {
                    // yolov5/models/yolo.py Detect forward
                    // y = x[i].sigmoid()
                    // y[..., 0:2] = (y[..., 0:2] * 2. - 0.5 + self.grid[i].to(x[i].device)) * self.stride[i]  # xy
                    // y[..., 2:4] = (y[..., 2:4] * 2) ** 2 * self.anchor_grid[i]  # wh

                    float dx = sigmoid(featptr[0]);
                    float dy = sigmoid(featptr[1]);
                    float dw = sigmoid(featptr[2]);
                    float dh = sigmoid(featptr[3]);

                    float pb_cx = (dx * 2.f - 0.5f + j) * stride;
                    float pb_cy = (dy * 2.f - 0.5f + i) * stride;

                    float pb_w = pow(dw * 2.f, 2) * anchor_w;
                    float pb_h = pow(dh * 2.f, 2) * anchor_h;

                    float x0 = pb_cx - pb_w * 0.5f;
                    float y0 = pb_cy - pb_h * 0.5f;
                    float x1 = pb_cx + pb_w * 0.5f;
                    float y1 = pb_cy + pb_h * 0.5f;

                    Object obj;
                    obj.x = x0;
                    obj.y = y0;
                    obj.w = x1 - x0;
                    obj.h = y1 - y0;
                    obj.label = class_index;
                    obj.prob = confidence;

                    objects.push_back(obj);
                }
            }
        }
    }
}


NanoDet::NanoDet()
{
    blob_pool_allocator.set_size_compare_ratio(0.f);
    workspace_pool_allocator.set_size_compare_ratio(0.f);
}


class YoloV5Focus : public ncnn::Layer
{
public:
    YoloV5Focus()
    {
        one_blob_only = true;
    }

    virtual int forward(const ncnn::Mat& bottom_blob, ncnn::Mat& top_blob, const ncnn::Option& opt) const
    {
        int w = bottom_blob.w;
        int h = bottom_blob.h;
        int channels = bottom_blob.c;

        int outw = w / 2;
        int outh = h / 2;
        int outc = channels * 4;

        top_blob.create(outw, outh, outc, 4u, 1, opt.blob_allocator);
        if (top_blob.empty())
            return -100;

#pragma omp parallel for num_threads(opt.num_threads)
        for (int p = 0; p < outc; p++)
        {
            const float* ptr = bottom_blob.channel(p % channels).row((p / channels) % 2) + ((p / channels) / 2);
            float* outptr = top_blob.channel(p);

            for (int i = 0; i < outh; i++)
            {
                for (int j = 0; j < outw; j++)
                {
                    *outptr = *ptr;

                    outptr += 1;
                    ptr += 2;
                }

                ptr += w;
            }
        }

        return 0;
    }
};

DEFINE_LAYER_CREATOR(YoloV5Focus)

int NanoDet::load(AAssetManager* mgr, const char* modeltype, int _target_size, const float* _mean_vals, const float* _norm_vals, bool use_gpu)
{
    nanodet.clear();
    dbNet.clear();

    blob_pool_allocator.clear();
    workspace_pool_allocator.clear();

    ncnn::set_cpu_powersave(2);
    ncnn::set_omp_num_threads(ncnn::get_big_cpu_count());

    ncnn::Option opt;
    opt.lightmode = true;
    opt.num_threads = 4;
    opt.blob_allocator = &blob_pool_allocator;
    opt.workspace_allocator = &workspace_pool_allocator;
    opt.use_packing_layout = true;
    opt.use_fp16_storage = true;

    //nanodet.opt = ncnn::Option();
#if NCNN_VULKAN
    opt.use_vulkan_compute = use_gpu;
#endif
    nanodet.opt = opt;
    dbNet.opt = opt;


//#if NCNN_VULKAN
//    nanodet.opt.use_vulkan_compute = use_gpu;
//#endif
//
//    nanodet.opt.num_threads = ncnn::get_big_cpu_count();
//    nanodet.opt.blob_allocator = &blob_pool_allocator;
//    nanodet.opt.workspace_allocator = &workspace_pool_allocator;
//
//    nanodet.opt.lightmode = true;
//    //nanodet.opt.num_threads = 4;
//
//    nanodet.opt.use_packing_layout = true;
//    //那变这样 都能检测
//    nanodet.opt.use_fp16_storage = true;

    // use vulkan compute
    //  if (ncnn::get_gpu_count() != 0)
    //     nanodet.opt.use_vulkan_compute = true;

    //nanodet.load_param(mgr, parampath);
    //nanodet.load_model(mgr, modelpath);
    nanodet.register_custom_layer("YoloV5Focus", YoloV5Focus_layer_creator);

//    nanodet.load_param(mgr, "best17-sim.param");
//    nanodet.load_model(mgr, "best17-sim.bin");
//
//    dbNet.load_param(mgr, "pdocrv2.0_det-op.param");
//    dbNet.load_model(mgr, "pdocrv2.0_det-op.bin");

    // init param
    {
        int ret = nanodet.load_param(mgr, "best17-sim.param");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "PaddleocrNcnn", "load_dbNet_param failed");
            return JNI_FALSE;
        }

        ret = dbNet.load_param(mgr, "pdocrv2.0_det-op.param");

        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "PaddleocrNcnn", "load_crnnNet_param failed");
            return JNI_FALSE;
        }
    }

    // init bin
    {
        int ret = nanodet.load_model(mgr, "best17-sim.bin");
        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "PaddleocrNcnn", "load_dbNet_model failed");
            return JNI_FALSE;
        }

        ret = dbNet.load_model(mgr, "pdocrv2.0_det-op.bin");

        if (ret != 0)
        {
            __android_log_print(ANDROID_LOG_WARN, "PaddleocrNcnn", "load_crnnNet_model failed");
            return JNI_FALSE;
        }
    }


    //_target_size什么意思
    //target_size = _target_size;

    target_size = 640;

    mean_vals[0] = _mean_vals[0];
    mean_vals[1] = _mean_vals[1];
    mean_vals[2] = _mean_vals[2];
    norm_vals[0] = _norm_vals[0];
    norm_vals[1] = _norm_vals[1];
    norm_vals[2] = _norm_vals[2];

    return 0;
}


std::vector<TextBox> findRsBoxes(const cv::Mat& fMapMat, const cv::Mat& norfMapMat,
                                 const float boxScoreThresh, const float unClipRatio)
{
    float minArea = 3;
    std::vector<TextBox> rsBoxes;
    rsBoxes.clear();
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(norfMapMat, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i)
    {
        float minSideLen, perimeter;
        std::vector<cv::Point> minBox = getMinBoxes(contours[i], minSideLen, perimeter);
        if (minSideLen < minArea)
            continue;
        float score = boxScoreFast(fMapMat, contours[i]);
        if (score < boxScoreThresh)
            continue;
        //---use clipper start---
        std::vector<cv::Point> clipBox = unClip(minBox, perimeter, unClipRatio);
        std::vector<cv::Point> clipMinBox = getMinBoxes(clipBox, minSideLen, perimeter);
        //---use clipper end---

        if (minSideLen < minArea + 2)
            continue;

        for (int j = 0; j < clipMinBox.size(); ++j)
        {
            clipMinBox[j].x = (clipMinBox[j].x / 1.0);
            clipMinBox[j].x = (std::min)((std::max)(clipMinBox[j].x, 0), norfMapMat.cols);

            clipMinBox[j].y = (clipMinBox[j].y / 1.0);
            clipMinBox[j].y = (std::min)((std::max)(clipMinBox[j].y, 0), norfMapMat.rows);
        }

        rsBoxes.emplace_back(TextBox{ clipMinBox, score });
    }
    reverse(rsBoxes.begin(), rsBoxes.end());

    return rsBoxes;
}

float NanoDet::detect_ocr_angle( cv::Mat& src,const Object& obj ){
    cv::Mat midd_img;
    // 方法一
    midd_img = src(cv::Rect(cv::Point(obj.x, obj.y), cv::Size(obj.w, obj.h)));
    cv::cvtColor(midd_img, midd_img, cv::COLOR_BGR2RGB);

    //std::vector<TextBox> objects;
    float final_result_angle=0.0;
    //三个参数代表什么
    float boxScoreThresh=0.4;
    float boxThresh=0.3;
    float unClipRatio=2.0;


    int width = src.cols;
    int height = src.rows;
    int target_size = 640;
    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    //注意下
    ncnn::Mat input = ncnn::Mat::from_pixels_resize(src.data, ncnn::Mat::PIXEL_RGB, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(input, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    const float meanValues[3] = { 0.485 * 255, 0.456 * 255, 0.406 * 255 };
    const float normValues[3] = { 1.0 / 0.229 / 255.0, 1.0 / 0.224 / 255.0, 1.0 / 0.225 / 255.0 };

    in_pad.substract_mean_normalize(meanValues, normValues);
    ncnn::Extractor extractor = dbNet.create_extractor();

    extractor.input("input0", in_pad);
    ncnn::Mat out;
    extractor.extract("out1", out);

    cv::Mat fMapMat(in_pad.h, in_pad.w, CV_32FC1, (float*)out.data);
    cv::Mat norfMapMat;
    norfMapMat = fMapMat > boxThresh;

    cv::dilate(norfMapMat, norfMapMat, cv::Mat(), cv::Point(-1, -1), 1);

    std::vector<TextBox> result = findRsBoxes(fMapMat, norfMapMat, boxScoreThresh, 2.0f);
    //几个结果
    for(int i = 0; i < result.size(); i++)
    {
        //该结果几个点
        for(int j = 0; j < result[i].boxPoint.size(); j++)
        {
            float x = (result[i].boxPoint[j].x-(wpad/2))/scale;
            float y = (result[i].boxPoint[j].y-(hpad/2))/scale;
            x = std::max(std::min(x,(float)(width-1)),0.f);
            y = std::max(std::min(y,(float)(height-1)),0.f);
            result[i].boxPoint[j].x = x;
            result[i].boxPoint[j].y = y;
            //__android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn ocr ", "result %d [%f , %f]",i,x,y );
        }

        //__android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn sum6t ", "%f value",p21 );

        //ocr 角度要复制过来代码
//        float ratio = distance2(result[i].boxPoint[0],result[i].boxPoint[1],result[i].boxPoint[2],result[i].boxPoint[3]);
//        for(ratio <=0.1 || ratio >=0.9){
//            len++;
//            if(result[i].boxPoint[0].x - result[i].boxPoint[0].y !=0){
//                angle_acc+=fabs()
//
//            }else{
//                angle_acc+=3.14/2;
//            }
//        }
    }
//    if(len!=0)
//        final_result_angle = angle_acc*180/(float)len/3.14;
//    else
//        final_result_angle = 999.0f;

    return final_result_angle;
}




int NanoDet::detect(const cv::Mat& rgb, std::vector<Object>& objects)
{
    int width = rgb.cols;
    int height = rgb.rows;

    // pad to multiple of 32
    int w = width;
    int h = height;
    float scale = 1.f;
    if (w > h)
    {
        scale = (float)target_size / w;
        w = target_size;
        h = h * scale;
    }
    else
    {
        scale = (float)target_size / h;
        h = target_size;
        w = w * scale;
    }

    ncnn::Mat in = ncnn::Mat::from_pixels_resize(rgb.data, ncnn::Mat::PIXEL_RGB2BGR, width, height, w, h);

    // pad to target_size rectangle
    int wpad = (w + 31) / 32 * 32 - w;
    int hpad = (h + 31) / 32 * 32 - h;
    ncnn::Mat in_pad;
    ncnn::copy_make_border(in, in_pad, hpad / 2, hpad - hpad / 2, wpad / 2, wpad - wpad / 2, ncnn::BORDER_CONSTANT, 0.f);

    //in_pad.substract_mean_normalize(mean_vals, norm_vals);
    {
        const float prob_threshold = 0.25f; //0.25f; 大量的密密麻麻的图像
        const float nms_threshold = 0.45f; //0.45f;

        //0.8 改成0.8 少了，但在上面 nms不起作用啊
        const float norm_vals[3] = {1 / 255.f, 1 / 255.f, 1 / 255.f};
        in_pad.substract_mean_normalize(0, norm_vals);


        ncnn::Extractor ex = nanodet.create_extractor();

        ex.input("images", in_pad);

        std::vector<Object> proposals;

        // stride 8
        {
            ncnn::Mat out;
            ex.extract("output", out);

            ncnn::Mat anchors(6);
            anchors[0] = 10.f;
            anchors[1] = 13.f;
            anchors[2] = 16.f;
            anchors[3] = 30.f;
            anchors[4] = 33.f;
            anchors[5] = 23.f;

            std::vector<Object> objects8;
            generate_proposals(anchors, 8, in_pad, out, prob_threshold, objects8);

            proposals.insert(proposals.end(), objects8.begin(), objects8.end());
        }

        // stride 16
        {
            ncnn::Mat out;
            ex.extract("417", out);  //781  417

            ncnn::Mat anchors(6);
            anchors[0] = 30.f;
            anchors[1] = 61.f;
            anchors[2] = 62.f;
            anchors[3] = 45.f;
            anchors[4] = 59.f;
            anchors[5] = 119.f;

            std::vector<Object> objects16;
            generate_proposals(anchors, 16, in_pad, out, prob_threshold, objects16);

            proposals.insert(proposals.end(), objects16.begin(), objects16.end());
        }

        // stride 32
        {
            ncnn::Mat out;
            ex.extract("437", out);//801

            ncnn::Mat anchors(6);
            anchors[0] = 116.f;
            anchors[1] = 90.f;
            anchors[2] = 156.f;
            anchors[3] = 198.f;
            anchors[4] = 373.f;
            anchors[5] = 326.f;

            std::vector<Object> objects32;
            generate_proposals(anchors, 32, in_pad, out, prob_threshold, objects32);

            proposals.insert(proposals.end(), objects32.begin(), objects32.end());
        }

        // sort all proposals by score from highest to lowest
        qsort_descent_inplace(proposals);

        // apply nms with nms_threshold
        std::vector<int> picked;
        nms_sorted_bboxes(proposals, picked, nms_threshold);

        int count = picked.size();

        objects.resize(count);
        for (int i = 0; i < count; i++) {
            objects[i] = proposals[picked[i]];

            // adjust offset to original unpadded
            float x0 = (objects[i].x - (wpad / 2)) / scale;
            float y0 = (objects[i].y - (hpad / 2)) / scale;
            float x1 = (objects[i].x + objects[i].w - (wpad / 2)) / scale;
            float y1 = (objects[i].y + objects[i].h - (hpad / 2)) / scale;

            // clip
            x0 = std::max(std::min(x0, (float) (width - 1)), 0.f);
            y0 = std::max(std::min(y0, (float) (height - 1)), 0.f);
            x1 = std::max(std::min(x1, (float) (width - 1)), 0.f);
            y1 = std::max(std::min(y1, (float) (height - 1)), 0.f);

            objects[i].x = x0;
            objects[i].y = y0;
            objects[i].w = x1 - x0;
            objects[i].h = y1 - y0;
        }
    }
    // sort objects by area
//    struct
//    {
//        bool operator()(const Object& a, const Object& b) const
//        {
//            return a.rect.area() > b.rect.area();
//        }
//    } objects_area_greater;
//    std::sort(objects.begin(), objects.end(), objects_area_greater);

    return 0;
}


void NanoDet::maxtwosqens(std::vector<int> &s,int& sta,int& end){


    //std::vector<int> s2=s;
    int j = 0 ;
    int max = 0 ;
    int qi=0;
    int zhong=0;

    for(int i = 0;i<s.size();i++){
        if(s[i] == 0){
            j++;
            zhong=i;
        }
        else{
            if(j>max){
                max = j;
                zhong = i-1;
            }
            j = 0 ;
        }
    }

    if(j>max)max = j ;

    int qirow = zhong-max;
    int j1 = 0 ;
    int max1 = 0 ;
    int qi1=0;
    int zhong1=0;

    for(int i = 0;i<s.size();i++){
        if(s[i] == 0){
            j1++;
            zhong1=i;
        }
        else{
            if(j1>max1){
                max1 = j1;
                zhong1 = i-1;
            }
            j1 = 0 ;
        }
    }

    if(j1>max1)max1 = j1 ;


    int qirow1 = zhong1-max1;
    int startke=0;
    int endke =0;

    if (qirow1>qirow){
        startke=zhong1-max1;
        endke =zhong;
    }
    else{
        startke=zhong+1;
        endke =zhong1-max1;
    }


    sta =startke;
    end = endke;
}

float NanoDet::detectthree(cv::Mat& rgb,const Object& obj,const float min_value,const float max_value) {
    cv::Mat midd_img;
    // 方法一
    midd_img = rgb(cv::Rect(cv::Point(obj.x, obj.y), cv::Size(obj.w, obj.h)));
    cv::cvtColor(midd_img, midd_img, cv::COLOR_BGR2RGB);
    cv::imwrite("/storage/emulated/0/DCIM/111midd_img.jpg", midd_img);
    cv::Mat gray_img;
    cv::cvtColor(midd_img, gray_img, CV_RGB2GRAY);
    cv::Mat imageSobel;
    cv::Sobel(gray_img, imageSobel, CV_16U, 1, 1);
    //图像的平均灰度,清晰度，越高约清晰
    double meanValue = 0.0;
    meanValue = mean(imageSobel)[0];
    if (meanValue<3.0)
        return float(10086.111f);


    int wight = midd_img.cols;
    int height = midd_img.rows;


    //cv::medianBlur(gray_img, gray_img, 5);
    std::vector <cv::Vec3f> circles;

    cv::HoughCircles(gray_img, circles, cv::HOUGH_GRADIENT, 1, 120, 100, 50, int(height * 0.35),
                     int(height * 0.48));

    float circle_x = 0;
    float circle_y = 0;
    float circle_r = 0;
    float reference_zero_angle = 20;
    float reference_end_angle = 340;
    float min_angle = 90;
    float max_angle = 270;
    float zhenpos = 0;
    int qipos = 0;
    int zhipos = 0;

    int b = circles.size();
    if (b == 0) {

        return float(10086.111f);
    } else {
        cv::Vec3d xyr = this->avg_circles(circles, b);

        circle_x = xyr[0];
        circle_y = xyr[1];
        circle_r = xyr[2];


        int thresh = 120;

        int maxValue = 255;
        cv::Mat midd_img2;
        cv::threshold(gray_img, midd_img2, thresh, maxValue, CV_THRESH_BINARY_INV);

        cv::Mat lin_polar_img;
        cv::linearPolar(midd_img2, lin_polar_img, cv::Point2f(circle_x, circle_y), circle_r,
                        CV_WARP_FILL_OUTLIERS + CV_INTER_LINEAR);
        cv::imwrite("/storage/emulated/0/DCIM/222lin_polar_img.jpg", lin_polar_img);


        int pwight = lin_polar_img.cols;
        int pheight = lin_polar_img.rows;

        cv::Mat sumcol;
        cv::reduce(lin_polar_img, sumcol, 1, CV_REDUCE_SUM,CV_32SC1);

        std::vector<float> sumcolvec;


        for (int i = 0; i < sumcol.rows; i++) {
            //要不要加0
            float p = sumcol.at<float>(i, 0);
            sumcolvec.push_back(p);
            //float p1 = sumcol.at<float>(0,i);
            //sumcolvec1.push_back(p);
        }

        //min_angle = *(max_element(sumcolvec.begin(), sumcolvec.end())+1);
        std::vector<float>::iterator maxPosition = max_element(sumcolvec.begin(), sumcolvec.end());
        int posi = maxPosition - sumcolvec.begin();
        // float sumsum = std::accumulate(sumcolvec.begin(), sumcolvec.end(), 0);
        //if(posi==0 || sumsum==0.0)
        //  return float(10086.111f);
        //min_angle = frth_angle_[maxPosition - frth_sub.begin()+1];
        cv::cvtColor(lin_polar_img,lin_polar_img,cv::COLOR_GRAY2BGR);
        cv::line(lin_polar_img, cv::Point( int(0),int(posi)), cv::Point(int(pwight),int(posi)),(255, 0,255 ), 3);

        cv::imwrite("/storage/emulated/0/DCIM/222444lin_polar_img.jpg", lin_polar_img);


        zhenpos = float(posi);



        float separation = 10.0;
        int interval = int(360 / separation);

        std::vector<cv::Point> pts;

        for (int i = 0; i < interval; i++) {
            cv::Point pp;
            for (int j = 0; j < 2; j++) {
                if (j % 2 == 0)
                    pp.x = circle_x + 1.0 * circle_r * cos(separation * i * CV_PI / 180);
                else
                    pp.y = circle_y + 1.0 * circle_r * sin(separation * i * CV_PI / 180);
            }
            pts.push_back(pp);
        }

        cv::Mat canny;
        cv::Canny(gray_img, canny, 200, 20);
        //Mat region_of_interest_vertices= p3;
        //imwrite("canny.jpg", canny);

        std::vector<std::vector<cv::Point>> region_of_interest_vertices;
        region_of_interest_vertices.push_back(pts);

        cv::Mat cropped_image = this->region_of_interest(canny, region_of_interest_vertices);


        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        //findContours(cropped_image,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());
        cv::findContours(cropped_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        //Mat imageContours=Mat::zeros(image.size(),CV_8UC1);
        //Mat Contours=Mat::zeros(image.size(),CV_8UC1);  //绘制
        //std::vector<int> int_cnt;
        std::vector<std::vector<cv::Point> > int_cnt;

        for (int i = 0; i < contours.size(); i++) {
            float area = cv::contourArea(contours[i]);
            cv::Rect prect = cv::boundingRect(contours[i]);

            float cpd = this->dist_2_pts(prect.x + prect.width / 2, prect.y + prect.height / 2, circle_x,circle_y);

            if ((area < 500) && (cpd < circle_r * 4 / 4) && (cpd > circle_r * 2 / 4)) {
                //drawContours(contours3, vector<vector<Point> >(1, contours[i]), -1,Scalar(255, 0, 0), 3);
                int_cnt.push_back(contours[i]);
            }
        }
        //imwrite("contours3.jpg", contours3);
        if (int_cnt.size() == 0)
            return float(10086.111f);

        std::vector<int> frth_quad_index;
        std::vector<int> thrd_quad_index;
        std::vector<float> frth_quad_angle;
        std::vector<float> thrd_quad_angle;

        for (int i = 0; i < int_cnt.size(); i++) {
            //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
            std::vector<cv::Point> conPoints;
            float x1, y1;
            float sx1 = 0, sy1 = 0;
            for (int j = 0; j < contours[i].size(); j++) {
                //绘制出contours向量内所有的像素点
                //Point P=Point(contours[i][j].x,contours[i][j].y);
                //conPoints.push_back(P);
                sx1 += contours[i][j].x;
                sy1 += contours[i][j].y;
            }
            x1 = sx1 / contours[i].size();
            y1 = sy1 / contours[i].size();

            float xlen = x1 - circle_x;
            float ylen = circle_y - y1;

            //double res = atan2(float(ylen), float(xlen));
            //res = res * 180.0 / M_PI;

            if ((xlen < 0) && (ylen < 0)) {
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI;
                float final_start_angle = 90 - res;

                frth_quad_index.push_back(i);
                frth_quad_angle.push_back(final_start_angle);

                if (final_start_angle > reference_zero_angle)
                    if (final_start_angle < min_angle)
                        min_angle = final_start_angle;

            }

            if ((xlen > 0) && (ylen < 0)) {
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI;
                float final_end_angle = 270 + res;

                thrd_quad_index.push_back(i);
                thrd_quad_angle.push_back(final_end_angle);

                if (final_end_angle < reference_end_angle)
                    if (final_end_angle > max_angle)
                        max_angle = final_end_angle;
            }

        }


        std::vector<float> frth_angle_(frth_quad_angle);
        std::vector<float> thrd_angle_(thrd_quad_angle);

        //升序 ，，降序
        std::sort(frth_angle_.begin(), frth_angle_.end(), std::less<float>());
        std::sort(thrd_angle_.begin(), thrd_angle_.end(), std::greater<float>());

        std::vector<float> frth_sub;
        std::vector<float> thrd_sub;
        for (int i = 0; i < frth_angle_.size() - 1; i++)
            frth_sub.push_back(frth_angle_[i + 1] - frth_angle_[i]);
        for (int i = 0; i < thrd_angle_.size() - 1; i++)
            thrd_sub.push_back(thrd_angle_[i + 1] - thrd_angle_[i]);


        std::vector<float>::iterator maxPosition1 = max_element(frth_sub.begin(), frth_sub.end());
        min_angle = frth_angle_[maxPosition1 - frth_sub.begin() + 1];
        //min_angle = *(max_element(frth_sub.begin(), frth_sub.end())+1);

        std::vector<float>::iterator minPosition = min_element(thrd_sub.begin(), thrd_sub.end());
        max_angle = thrd_angle_[minPosition - thrd_sub.begin() + 1];

    }
    float old_min = float(min_angle) ;
    float old_max = float(max_angle) ;



    float new_min = float( min_value);
    float new_max = float( max_value);

    float old_value = 270.0-zhenpos;

    float old_range = (old_max - old_min);
    float new_range = (new_max - new_min);

    float final_value = float(((old_value - old_min) * new_range) / float(old_range)) + float(new_min);

    if ((final_value< float(min_value) ) || (final_value> float(max_value)))
        return float(10086.111f);

    return final_value;






}

float NanoDet::polardetect(cv::Mat& rgb,const Object& obj,const float min_value,const float max_value) {

    cv::Mat midd_img;
    // 方法一
    midd_img = rgb(cv::Rect(cv::Point(obj.x, obj.y), cv::Size(obj.w, obj.h)));
    cv::cvtColor(midd_img, midd_img, cv::COLOR_BGR2RGB);
    cv::imwrite("/storage/emulated/0/DCIM/111midd_img.jpg", midd_img);
    cv::Mat gray_img;
    cv::cvtColor(midd_img, gray_img, CV_RGB2GRAY);
    cv::Mat imageSobel;
    cv::Sobel(gray_img, imageSobel, CV_16U, 1, 1);
    //图像的平均灰度,清晰度，越高约清晰
    double meanValue = 0.0;
    meanValue = mean(imageSobel)[0];
    if (meanValue<3.0)
        return float(10086.111f);


    int wight = midd_img.cols;
    int height = midd_img.rows;


    //cv::medianBlur(gray_img, gray_img, 5);

    std::vector <cv::Vec3f> circles;


    cv::HoughCircles(gray_img, circles, cv::HOUGH_GRADIENT, 1, 120, 100, 50, int(height * 0.35),
                     int(height * 0.48));

    float circle_x = 0;
    float circle_y = 0;
    float circle_r = 0;
    float reference_zero_angle = 20;
    float reference_end_angle = 340;
    float min_angle = 90;
    float max_angle = 270;


    int b = circles.size();
    if (b == 0) {

        return float(10086.111f);
    } else {
        cv::Vec3d xyr = this->avg_circles(circles, b);

        circle_x = xyr[0];
        circle_y = xyr[1];
        circle_r = xyr[2];


        int thresh = 120;

        int maxValue = 255;
        cv::Mat midd_img2;
        cv::threshold(gray_img, midd_img2, thresh, maxValue, CV_THRESH_BINARY_INV);

        cv::Mat lin_polar_img;
        cv::linearPolar(midd_img2, lin_polar_img, cv::Point2f(circle_x, circle_y), circle_r,
                        CV_WARP_FILL_OUTLIERS + CV_INTER_LINEAR);
        cv::imwrite("/storage/emulated/0/DCIM/222lin_polar_img.jpg", lin_polar_img);


        int pwight = lin_polar_img.cols;
        int pheight = lin_polar_img.rows;
//        try{
            //什么类型，float int
            cv::Mat sumcol;
            cv::reduce(lin_polar_img, sumcol, 1, CV_REDUCE_SUM,CV_32SC1);

            std::vector<float> sumcolvec;

//        }
//        catch(std::exception& e)
//            {
//                std::cout << e.what() << std::endl;
//            }



        for (int i = 0; i < sumcol.rows; i++) {
            //要不要加0
            float p = sumcol.at<float>(i, 0);
            sumcolvec.push_back(p);
            //float p1 = sumcol.at<float>(0,i);
            //sumcolvec1.push_back(p);
        }

        //min_angle = *(max_element(sumcolvec.begin(), sumcolvec.end())+1);
        std::vector<float>::iterator maxPosition = max_element(sumcolvec.begin(), sumcolvec.end());
        int posi = maxPosition - sumcolvec.begin();
        // float sumsum = std::accumulate(sumcolvec.begin(), sumcolvec.end(), 0);
        //if(posi==0 || sumsum==0.0)
          //  return float(10086.111f);
        //min_angle = frth_angle_[maxPosition - frth_sub.begin()+1];
        cv::cvtColor(lin_polar_img,lin_polar_img,cv::COLOR_GRAY2BGR);
        cv::line(lin_polar_img, cv::Point( int(0),int(posi)), cv::Point(int(pwight),int(posi)),(255, 0,255 ), 3);

        cv::imwrite("/storage/emulated/0/DCIM/222444lin_polar_img.jpg", lin_polar_img);

        int zhenpos = 0;
        int qipos = 0;
        int zhipos = 0;
        zhenpos = int(posi);


        float separation = 10.0;
        int interval = int(360 / separation);

        cv::Mat p3 = cv::Mat::zeros(cv::Size(interval, 2), CV_32FC1);
        std::vector <cv::Point> pts;

        for (int i = 0; i < interval; i++) {
            cv::Point pp;
            for (int j = 0; j < 2; j++) {
                if (j % 2 == 0)
                    pp.x = circle_x + 1.0 * circle_r * cos(separation * i * CV_PI / 180);
                else
                    pp.y = circle_y + 1.0 * circle_r * sin(separation * i * CV_PI / 180);
            }
            pts.push_back(pp);
        }

        //图像对比度增强
      //  cv::Mat imageEnhance;
      //   cv::Mat kernel = (cv::Mat_<float>(3, 3) << 0, -1, 0, 0, 1, 0, 0, -1, 0);
      //  cv::filter2D(gray_img, imageEnhance, CV_8UC1, kernel);
      //  cv::imwrite("/storage/emulated/0/DCIM/333imageEnhance.jpg", imageEnhance);

        //对img进行限制对比度自适应直方图均衡化
        cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE(3,cv::Size(8,8));
        cv::Mat CLAHEimg;
        clahe->apply(gray_img, CLAHEimg);

        cv::Mat canny;
        cv::Canny(CLAHEimg, canny, 180, 40);
        //Mat region_of_interest_vertices= p3;
        cv::imwrite("/storage/emulated/0/DCIM/333canny.jpg", canny);

        std::vector <std::vector<cv::Point>> region_of_interest_vertices;
        region_of_interest_vertices.push_back(pts);


        cv::Mat cropped_image = region_of_interest(canny, region_of_interest_vertices);
        cv::imwrite("/storage/emulated/0/DCIM/333cropped_image.jpg", cropped_image);

        cv::Mat maskpl = cv::Mat::zeros(cropped_image.size(), CV_8UC1);
        cv::Mat contours3 = midd_img.clone();

        std::vector <std::vector<cv::Point>> contours;
        std::vector <cv::Vec4i> hierarchy;
        cv::findContours(cropped_image, contours, hierarchy, cv::RETR_EXTERNAL,
                         cv::CHAIN_APPROX_NONE);

        std::vector <std::vector<cv::Point>> int_cnt;
        for (int i = 0; i < contours.size(); i++) {
            float area = cv::contourArea(contours[i]);
            cv::Rect prect = boundingRect(contours[i]);

            float cpd = dist_2_pts(prect.x + prect.width / 2, prect.y + prect.height / 2, circle_x,
                                   circle_y);

            if ((area < 500) && (cpd < circle_r * 4 / 4) && (cpd > circle_r * 2 / 4)) {

                //cv::drawContours(contours3, vector<vector<Point> >(1,contours[i]), -1, Scalar(255,0,0), 3);
                cv::drawContours(maskpl, std::vector < std::vector < cv::Point > > (1, contours[i]),
                                 -1, 255, 3);

                int_cnt.push_back(contours[i]);
            }
        }
        cv::imwrite("/storage/emulated/0/DCIM/444maskpl.jpg", maskpl);

        if(int_cnt.size()==0)
            return float(10086.111f);

        cv::Mat polar6;
        cv::linearPolar(maskpl, polar6, cv::Point2f(circle_x, circle_y), circle_r,
                        CV_WARP_FILL_OUTLIERS + CV_INTER_LINEAR);
        cv::imwrite("/storage/emulated/0/DCIM/444555polar63.jpg", polar6);


        std::vector<int> sum6;
        std::vector <std::vector<int>> sum6tt;

        for (int col = 0; col < pwight; col++) {

            cv::Mat col1 = polar6.colRange(col, col + 1);
            std::vector<int> sum6t;
            for (int i = 0; i < pheight - 1;) {
                int p1 = (int)col1.at<uchar>(i, 0);
                int p2 = (int)col1.at<uchar>(i+1, 0);
                int p21 = abs(p2 - p1);
                i += 2;
                //__android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn sum6t ", "%f value",p21 );
                sum6t.push_back(p21);
            }
            sum6tt.push_back(sum6t);

            float suum = accumulate(sum6t.begin(), sum6t.end(), 0.0);
            sum6.push_back(suum);
        }

        std::vector<int>::iterator maxPosition2 = max_element(sum6.begin(), sum6.end());
        int posi2 = maxPosition2 - sum6.begin();

        for (auto val : sum6tt[posi2])
            __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn sum6tt ", "%d value",val );

        //std::vector<int> kedulist ;
        //std::copy(kedulist.begin(),kedulist.end(),  std::back_inserter(sum6tt[posi2]));

        if (sum6tt[posi2].size()==0)
            return float(10086.111f);

        cv::Mat polar63;
        cv::cvtColor(polar6, polar63, cv::COLOR_GRAY2BGR);

        //cv::line(polar63, cv::Point(int(posi2-3), int(0)), cv::Point(int(posi2-3),int(pwight)),cv::Scalar(255, 0,255 ), 3);

        int possum = 0;

        int startke = 0;
        int endke = 0;
        this->maxtwosqens(sum6tt[posi2], startke, endke);
        startke= startke*2;
        endke =endke* 2;
        __android_log_print(ANDROID_LOG_DEBUG, "YoloV5Ncnn sum6tt ", "%d  %d %d value",startke,endke,posi );
        cv::line(polar63, cv::Point( int(0),int(endke)), cv::Point(int(pwight),int(endke)),cv::Scalar(255, 0,255 ), 3);
        cv::line(polar63, cv::Point( int(0),int(startke)),cv::Point(int(pwight),int(startke)),cv::Scalar(255, 0,255 ), 3);
        cv::imwrite("/storage/emulated/0/DCIM/444polar63.jpg", polar63);


        float new_min = float(min_value);
        float new_max = float(max_value);

        qipos= startke;
        zhipos= endke;
        float final_value =
                float(zhenpos - qipos) / float(pheight - qipos + zhipos) * float(new_max - new_min);

        if ((final_value < float(new_min)) || (final_value > float(new_max)))
            return float(10086.111f);

        return final_value;
    }

}

float NanoDet::detectvalue(cv::Mat& rgb,const Object& obj,const float min_value,const float max_value) {

    cv::Mat midd_img;
    // 方法一
    midd_img = rgb(cv::Rect(cv::Point(obj.x, obj.y), cv::Size(obj.w, obj.h)));
    cv::cvtColor(midd_img, midd_img, cv::COLOR_BGR2RGB);
    //cv::Mat midd_img = rgb.clone();
    //cv::Mat midd_img = rgb;
    int wight = midd_img.rows;
    int height = midd_img.cols;

    cv::Mat gray_img;
    cv::cvtColor(midd_img, gray_img, cv::COLOR_RGB2GRAY);
    //cv::medianBlur(gray_img, gray_img, 5);

    cv::Mat imageSobel;
    cv::Sobel(gray_img, imageSobel, CV_16U, 1, 1);
    //图像的平均灰度,清晰度，越高约清晰
    double meanValue = 0.0;
    meanValue = mean(imageSobel)[0];
    if (meanValue<3.0)
        return float(10086.111f);


    std::vector<cv::Vec3f> circles;


    cv::HoughCircles(gray_img, circles, cv::HOUGH_GRADIENT, 1, 120, 100, 50, int(height * 0.35),int(height * 0.48));

    float circle_x = 0;
    float circle_y = 0;
    float circle_r = 0;
    float reference_zero_angle = 20;
    float reference_end_angle = 340;
    float min_angle = 90;
    float max_angle = 270;


    int b = circles.size();
    if (b == 0) {
        cv::Mat a;
        return float(10086.111f);
    } else {
        cv::Vec3d xyr = this->avg_circles(circles, b);

        circle_x = xyr[0];
        circle_y = xyr[1];
        circle_r = xyr[2];


        float separation = 10.0;
        int interval = int(360 / separation);

        std::vector<cv::Point> pts;

        for (int i = 0; i < interval; i++) {
            cv::Point pp;
            for (int j = 0; j < 2; j++) {
                if (j % 2 == 0)
                    pp.x = circle_x + 1.0 * circle_r * cos(separation * i * CV_PI / 180);
                else
                    pp.y = circle_y + 1.0 * circle_r * sin(separation * i * CV_PI / 180);
            }
            pts.push_back(pp);
        }

        cv::Mat canny;
        cv::Canny(gray_img, canny, 200, 20);
        //Mat region_of_interest_vertices= p3;
        //imwrite("canny.jpg", canny);

        std::vector<std::vector<cv::Point>> region_of_interest_vertices;
        region_of_interest_vertices.push_back(pts);

        cv::Mat cropped_image = this->region_of_interest(canny, region_of_interest_vertices);


        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        //findContours(cropped_image,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());
        cv::findContours(cropped_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        //Mat imageContours=Mat::zeros(image.size(),CV_8UC1);
        //Mat Contours=Mat::zeros(image.size(),CV_8UC1);  //绘制
        //std::vector<int> int_cnt;
        std::vector<std::vector<cv::Point> > int_cnt;

        for (int i = 0; i < contours.size(); i++) {
            float area = cv::contourArea(contours[i]);
            cv::Rect prect = cv::boundingRect(contours[i]);

            float cpd = this->dist_2_pts(prect.x + prect.width / 2, prect.y + prect.height / 2, circle_x,circle_y);

            if ((area < 500) && (cpd < circle_r * 4 / 4) && (cpd > circle_r * 2 / 4)) {
                //drawContours(contours3, vector<vector<Point> >(1, contours[i]), -1,Scalar(255, 0, 0), 3);
                int_cnt.push_back(contours[i]);
            }
        }
        //imwrite("contours3.jpg", contours3);
        if (int_cnt.size() == 0)
            return float(10086.111f);

        //10 350


        std::vector<int> frth_quad_index;
        std::vector<int> thrd_quad_index;
        std::vector<float> frth_quad_angle;
        std::vector<float> thrd_quad_angle;

        for (int i = 0; i < int_cnt.size(); i++) {
            //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
            std::vector<cv::Point> conPoints;
            float x1, y1;
            float sx1 = 0, sy1 = 0;
            for (int j = 0; j < contours[i].size(); j++) {
                //绘制出contours向量内所有的像素点
                //Point P=Point(contours[i][j].x,contours[i][j].y);
                //conPoints.push_back(P);
                sx1 += contours[i][j].x;
                sy1 += contours[i][j].y;
            }
            x1 = sx1 / contours[i].size();
            y1 = sy1 / contours[i].size();

            float xlen = x1 - circle_x;
            float ylen = circle_y - y1;

            //double res = atan2(float(ylen), float(xlen));
            //res = res * 180.0 / M_PI;

            if ((xlen < 0) && (ylen < 0)) {
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI;
                float final_start_angle = 90 - res;

                frth_quad_index.push_back(i);
                frth_quad_angle.push_back(final_start_angle);

                if (final_start_angle > reference_zero_angle)
                    if (final_start_angle < min_angle)
                        min_angle = final_start_angle;

            }

            if ((xlen > 0) && (ylen < 0)) {
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI;
                float final_end_angle = 270 + res;

                thrd_quad_index.push_back(i);
                thrd_quad_angle.push_back(final_end_angle);

                if (final_end_angle < reference_end_angle)
                    if (final_end_angle > max_angle)
                        max_angle = final_end_angle;
            }

        }


        std::vector<float> frth_angle_(frth_quad_angle);
        std::vector<float> thrd_angle_(thrd_quad_angle);

        //升序 ，，降序
        std::sort(frth_angle_.begin(), frth_angle_.end(), std::less<float>());
        std::sort(thrd_angle_.begin(), thrd_angle_.end(), std::greater<float>());

        std::vector<float> frth_sub;
        std::vector<float> thrd_sub;
        for (int i = 0; i < frth_angle_.size() - 1; i++)
            frth_sub.push_back(frth_angle_[i + 1] - frth_angle_[i]);
        for (int i = 0; i < thrd_angle_.size() - 1; i++)
            thrd_sub.push_back(thrd_angle_[i + 1] - thrd_angle_[i]);


        std::vector<float>::iterator maxPosition = max_element(frth_sub.begin(), frth_sub.end());
        min_angle = frth_angle_[maxPosition - frth_sub.begin() + 1];
        //min_angle = *(max_element(frth_sub.begin(), frth_sub.end())+1);

        std::vector<float>::iterator minPosition = min_element(thrd_sub.begin(), thrd_sub.end());
        max_angle = thrd_angle_[minPosition - thrd_sub.begin() + 1];

    }


    //检测线

    //50cm 模糊3像素
    //cv::Ptr<cv::CLAHE> clahe = createCLAHE(40.0, Size(8, 8));
    //Mat dstcle;
    //限制对比度的自适应阈值
    //clahe->apply(gray_img, dstcle);
    //原图一定屏蔽掉，模糊的要添加，原图添加，识别不了， 模糊的 不添加 识别不了
    //gray2 =dst

    int thresh = 166;
    int maxValue = 255;
    cv::Mat midd_img2;

    std::vector<cv::Vec4i> mylines;
    int g_nthreshold = 39;


    cv::threshold(gray_img, midd_img2, thresh, maxValue, CV_THRESH_BINARY_INV);

    cv::HoughLinesP(midd_img2, mylines, 3, CV_PI / 180, 100, 10, 0);

    if (mylines.empty()) return float(10086.111f);
    cv::Point circle_center = cv::Point2f(circle_x, circle_y);
    float circle_radius = circle_r;


    float diff1LowerBound = 0.05;
    float diff1UpperBound = 0.25;
    float diff2LowerBound = 0.05;
    float diff2UpperBound = 1.0;


    std::vector<cv::Vec4i> final_line_list;
    std::vector<float> distance_list;
    std::vector<float> line_length_list;
    cv::Mat midd_img6 = midd_img.clone();

    for (size_t i = 0; i < mylines.size(); i++) {
        cv::Vec4i l = mylines[i];
        float diff1 = this->dist_2_pts(circle_center.x, circle_center.y, l[0], l[1]);
        float diff2 = this->dist_2_pts(circle_center.x, circle_center.y, l[2], l[3]);
        if (diff1 > diff2) {
            float temp = diff1;
            diff1 = diff2;
            diff2 = temp;
        }

        if (((diff1 < diff1UpperBound * circle_radius) && (diff1 > diff1LowerBound * circle_radius)) &&
            ((diff2 < diff2UpperBound * circle_radius) && (diff2 > diff2LowerBound * circle_radius))) {

            float line_length = this->dist_2_pts(l[0], l[1], l[2], l[3]);
            float distance = this->getDist_P2L(cv::Point2f(circle_center.x, circle_center.y), cv::Point2f(l[0], l[1]),
                                               cv::Point2f(l[2], l[3]));

            if ((line_length>0.1*circle_radius)  && (distance>-20) && (distance <10)){
                final_line_list.push_back(cv::Vec4i(l[0], l[1], l[2], l[3]));
                distance_list.push_back(distance);
                line_length_list.push_back(line_length);
                //cv::line(midd_img6, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(23, 180, 55), 2,CV_AA);
            }

        }

    };
    //imwrite("midd_img6.jpg", midd_img6);
    if (final_line_list.empty()) return float(10086.111f);
    //输出第一个线，点到直线的距离，点到两个端点的距离，线的长度；最短距离的位置，线最长的位置
    std::vector<float>::iterator maxPosition = max_element(line_length_list.begin(),line_length_list.end());

    std::vector<float>::iterator minPosition = min_element(distance_list.begin(),distance_list.end());

    cv::Vec4i final_line;

    final_line = final_line_list[maxPosition - line_length_list.begin() + 1];


    float x1 = final_line[0];
    float y1 = final_line[1];
    float x2 = final_line[2];
    float y2 = final_line[3];


    //find the farthest point from the center to be what is used to determine the angle
    float dist_pt_0 = this->dist_2_pts(circle_center.x, circle_center.y, x1, y1);
    float dist_pt_1 = this->dist_2_pts(circle_center.x, circle_center.y, x2, y2);

    float x_angle = 0.0;
    float y_angle = 0.0;
    if (dist_pt_0 > dist_pt_1) {
        x_angle = x1 - circle_center.x;
        y_angle = circle_center.y - y1;
    } else {
        x_angle = x2 - circle_center.x;
        y_angle = circle_center.y - y2;
    }

    x_angle = (x1 + x2) / 2 - circle_center.x;
    y_angle = circle_center.y - (y1 + y2) / 2;


    double res = atan2(float(y_angle), float(x_angle));

    //these were determined by trial and error
    res = res * 180.0 / M_PI;

    float final_angle = 0.0;

    if ((x_angle > 0) && (y_angle > 0))//in quadrant I
        final_angle = 270 - res;
    if (x_angle < 0 && y_angle > 0) //in quadrant II
        final_angle = 90 - res;
    if (x_angle < 0 && y_angle < 0)  //in quadrant III
        final_angle = 90 - res;
    if (x_angle > 0 && y_angle < 0)  //in quadrant IV
        final_angle = 270 - res;


    //vector<float> final_value_list;
    //for (int i = 0; i < 10; i++) {
    min_angle=50;
    max_angle=320;
    float old_min = float(min_angle) ;
    float old_max = float(max_angle) ;



    float new_min = float( min_value);
    float new_max = float( max_value);

    float old_value = final_angle;

    float old_range = (old_max - old_min);
    float new_range = (new_max - new_min);
    float final_value = (((old_value - old_min) * new_range) / old_range) + new_min;

    if ((final_value< float(min_value) ) || (final_value> float(max_value)))
        return float(10086.111f);

    return final_value;


}

float NanoDet::detectfour(cv::Mat& rgb,const Object& obj,const float min_value,const float max_value) {

    cv::Mat midd_img;
    // 方法一
    midd_img = rgb(cv::Rect(cv::Point(obj.x, obj.y), cv::Size(obj.w, obj.h)));
    cv::cvtColor(midd_img, midd_img, cv::COLOR_BGR2RGB);
    //cv::Mat midd_img = rgb.clone();
    //cv::Mat midd_img = rgb;
    int wight = midd_img.rows;
    int height = midd_img.cols;

    cv::Mat gray_img;
    cv::cvtColor(midd_img, gray_img, cv::COLOR_RGB2GRAY);
    //cv::medianBlur(gray_img, gray_img, 5);

    cv::Mat imageSobel;
    cv::Sobel(gray_img, imageSobel, CV_16U, 1, 1);
    //图像的平均灰度,清晰度，越高约清晰
    double meanValue = 0.0;
    meanValue = mean(imageSobel)[0];
    if (meanValue<3.0)
        return float(10086.111f);


    std::vector<cv::Vec3f> circles;


    cv::HoughCircles(gray_img, circles, cv::HOUGH_GRADIENT, 1, 120, 100, 50, int(height * 0.35),int(height * 0.48));

    float circle_x = 0;
    float circle_y = 0;
    float circle_r = 0;
    float reference_zero_angle = 20;
    float reference_end_angle = 340;
    float min_angle = 90;
    float max_angle = 270;


    int b = circles.size();
    if (b == 0) {
        cv::Mat a;
        return float(10086.111f);
    } else {
        cv::Vec3d xyr = this->avg_circles(circles, b);

        circle_x = xyr[0];
        circle_y = xyr[1];
        circle_r = xyr[2];


        float separation = 10.0;
        int interval = int(360 / separation);

        std::vector<cv::Point> pts;

        for (int i = 0; i < interval; i++) {
            cv::Point pp;
            for (int j = 0; j < 2; j++) {
                if (j % 2 == 0)
                    pp.x = circle_x + 1.0 * circle_r * cos(separation * i * CV_PI / 180);
                else
                    pp.y = circle_y + 1.0 * circle_r * sin(separation * i * CV_PI / 180);
            }
            pts.push_back(pp);
        }

        cv::Mat canny;
        cv::Canny(gray_img, canny, 200, 20);
        //Mat region_of_interest_vertices= p3;
        //imwrite("canny.jpg", canny);

        std::vector<std::vector<cv::Point>> region_of_interest_vertices;
        region_of_interest_vertices.push_back(pts);

        cv::Mat cropped_image = this->region_of_interest(canny, region_of_interest_vertices);


        std::vector<std::vector<cv::Point>> contours;
        std::vector<cv::Vec4i> hierarchy;

        //findContours(cropped_image,contours,hierarchy,RETR_TREE,CHAIN_APPROX_SIMPLE,Point());
        cv::findContours(cropped_image, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        //Mat imageContours=Mat::zeros(image.size(),CV_8UC1);
        //Mat Contours=Mat::zeros(image.size(),CV_8UC1);  //绘制
        //std::vector<int> int_cnt;
        std::vector<std::vector<cv::Point> > int_cnt;

        for (int i = 0; i < contours.size(); i++) {
            float area = cv::contourArea(contours[i]);
            cv::Rect prect = cv::boundingRect(contours[i]);

            float cpd = this->dist_2_pts(prect.x + prect.width / 2, prect.y + prect.height / 2, circle_x,circle_y);

            if ((area < 500) && (cpd < circle_r * 4 / 4) && (cpd > circle_r * 2 / 4)) {
                //drawContours(contours3, vector<vector<Point> >(1, contours[i]), -1,Scalar(255, 0, 0), 3);
                int_cnt.push_back(contours[i]);
            }
        }
        //imwrite("contours3.jpg", contours3);
        if (int_cnt.size() == 0)
            return float(10086.111f);

        //10 350


        std::vector<int> frth_quad_index;
        std::vector<int> thrd_quad_index;
        std::vector<float> frth_quad_angle;
        std::vector<float> thrd_quad_angle;

        for (int i = 0; i < int_cnt.size(); i++) {
            //contours[i]代表的是第i个轮廓，contours[i].size()代表的是第i个轮廓上所有的像素点数
            std::vector<cv::Point> conPoints;
            float x1, y1;
            float sx1 = 0, sy1 = 0;
            for (int j = 0; j < contours[i].size(); j++) {
                //绘制出contours向量内所有的像素点
                //Point P=Point(contours[i][j].x,contours[i][j].y);
                //conPoints.push_back(P);
                sx1 += contours[i][j].x;
                sy1 += contours[i][j].y;
            }
            x1 = sx1 / contours[i].size();
            y1 = sy1 / contours[i].size();

            float xlen = x1 - circle_x;
            float ylen = circle_y - y1;

            //double res = atan2(float(ylen), float(xlen));
            //res = res * 180.0 / M_PI;

            if ((xlen < 0) && (ylen < 0)) {
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI;
                float final_start_angle = 90 - res;

                frth_quad_index.push_back(i);
                frth_quad_angle.push_back(final_start_angle);

                if (final_start_angle > reference_zero_angle)
                    if (final_start_angle < min_angle)
                        min_angle = final_start_angle;

            }

            if ((xlen > 0) && (ylen < 0)) {
                double res = atan2(abs(float(ylen)), abs(float(xlen)));
                res = res * 180.0 / M_PI;
                float final_end_angle = 270 + res;

                thrd_quad_index.push_back(i);
                thrd_quad_angle.push_back(final_end_angle);

                if (final_end_angle < reference_end_angle)
                    if (final_end_angle > max_angle)
                        max_angle = final_end_angle;
            }

        }


        std::vector<float> frth_angle_(frth_quad_angle);
        std::vector<float> thrd_angle_(thrd_quad_angle);

        //升序 ，，降序
        std::sort(frth_angle_.begin(), frth_angle_.end(), std::less<float>());
        std::sort(thrd_angle_.begin(), thrd_angle_.end(), std::greater<float>());

        std::vector<float> frth_sub;
        std::vector<float> thrd_sub;
        for (int i = 0; i < frth_angle_.size() - 1; i++)
            frth_sub.push_back(frth_angle_[i + 1] - frth_angle_[i]);
        for (int i = 0; i < thrd_angle_.size() - 1; i++)
            thrd_sub.push_back(thrd_angle_[i + 1] - thrd_angle_[i]);


        std::vector<float>::iterator maxPosition = max_element(frth_sub.begin(), frth_sub.end());
        min_angle = frth_angle_[maxPosition - frth_sub.begin() + 1];
        //min_angle = *(max_element(frth_sub.begin(), frth_sub.end())+1);

        std::vector<float>::iterator minPosition = min_element(thrd_sub.begin(), thrd_sub.end());
        max_angle = thrd_angle_[minPosition - thrd_sub.begin() + 1];

    }


    //检测线

    //50cm 模糊3像素
    //cv::Ptr<cv::CLAHE> clahe = createCLAHE(40.0, Size(8, 8));
    //Mat dstcle;
    //限制对比度的自适应阈值
    //clahe->apply(gray_img, dstcle);
    //原图一定屏蔽掉，模糊的要添加，原图添加，识别不了， 模糊的 不添加 识别不了
    //gray2 =dst

    int thresh = 166;
    int maxValue = 255;
    cv::Mat midd_img2;

    std::vector<cv::Vec4i> mylines;
    int g_nthreshold = 39;


    cv::threshold(gray_img, midd_img2, thresh, maxValue, CV_THRESH_BINARY_INV);

    cv::HoughLinesP(midd_img2, mylines, 3, CV_PI / 180, 100, 10, 0);

    if (mylines.empty()) return float(10086.111f);
    cv::Point circle_center = cv::Point2f(circle_x, circle_y);
    float circle_radius = circle_r;


    float diff1LowerBound = 0.05;
    float diff1UpperBound = 0.25;
    float diff2LowerBound = 0.05;
    float diff2UpperBound = 1.0;


    std::vector<cv::Vec4i> final_line_list;
    std::vector<float> distance_list;
    std::vector<float> line_length_list;
    cv::Mat midd_img6 = midd_img.clone();

    for (size_t i = 0; i < mylines.size(); i++) {
        cv::Vec4i l = mylines[i];
        float diff1 = this->dist_2_pts(circle_center.x, circle_center.y, l[0], l[1]);
        float diff2 = this->dist_2_pts(circle_center.x, circle_center.y, l[2], l[3]);
        if (diff1 > diff2) {
            float temp = diff1;
            diff1 = diff2;
            diff2 = temp;
        }

        if (((diff1 < diff1UpperBound * circle_radius) && (diff1 > diff1LowerBound * circle_radius)) &&
            ((diff2 < diff2UpperBound * circle_radius) && (diff2 > diff2LowerBound * circle_radius))) {

            float line_length = this->dist_2_pts(l[0], l[1], l[2], l[3]);
            float distance = this->getDist_P2L(cv::Point2f(circle_center.x, circle_center.y), cv::Point2f(l[0], l[1]),
                                               cv::Point2f(l[2], l[3]));

            if ((line_length>0.1*circle_radius)  && (distance>-20) && (distance <10)){
                final_line_list.push_back(cv::Vec4i(l[0], l[1], l[2], l[3]));
                distance_list.push_back(distance);
                line_length_list.push_back(line_length);
                //cv::line(midd_img6, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(23, 180, 55), 2,CV_AA);
            }

        }

    };
    //imwrite("midd_img6.jpg", midd_img6);
    if (final_line_list.empty()) return float(10086.111f);
    //输出第一个线，点到直线的距离，点到两个端点的距离，线的长度；最短距离的位置，线最长的位置
    std::vector<float>::iterator maxPosition = max_element(line_length_list.begin(),line_length_list.end());

    std::vector<float>::iterator minPosition = min_element(distance_list.begin(),distance_list.end());

    cv::Vec4i final_line;

    final_line = final_line_list[maxPosition - line_length_list.begin() + 1];


    float x1 = final_line[0];
    float y1 = final_line[1];
    float x2 = final_line[2];
    float y2 = final_line[3];


    //find the farthest point from the center to be what is used to determine the angle
    float dist_pt_0 = this->dist_2_pts(circle_center.x, circle_center.y, x1, y1);
    float dist_pt_1 = this->dist_2_pts(circle_center.x, circle_center.y, x2, y2);

    float x_angle = 0.0;
    float y_angle = 0.0;
    if (dist_pt_0 > dist_pt_1) {
        x_angle = x1 - circle_center.x;
        y_angle = circle_center.y - y1;
    } else {
        x_angle = x2 - circle_center.x;
        y_angle = circle_center.y - y2;
    }

    x_angle = (x1 + x2) / 2 - circle_center.x;
    y_angle = circle_center.y - (y1 + y2) / 2;


    double res = atan2(float(y_angle), float(x_angle));

    //these were determined by trial and error
    res = res * 180.0 / M_PI;

    float final_angle = 0.0;

    if ((x_angle > 0) && (y_angle > 0))//in quadrant I
        final_angle = 270 - res;
    if (x_angle < 0 && y_angle > 0) //in quadrant II
        final_angle = 90 - res;
    if (x_angle < 0 && y_angle < 0)  //in quadrant III
        final_angle = 90 - res;
    if (x_angle > 0 && y_angle < 0)  //in quadrant IV
        final_angle = 270 - res;


    //vector<float> final_value_list;
    //for (int i = 0; i < 10; i++) {
    min_angle=50;
    max_angle=320;
    float old_min = float(min_angle) ;
    float old_max = float(max_angle) ;



    float new_min = float( min_value);
    float new_max = float( max_value);

    float old_value = final_angle;

    float old_range = (old_max - old_min);
    float new_range = (new_max - new_min);
    float final_value = (((old_value - old_min) * new_range) / old_range) + new_min;

    if ((final_value< float(min_value) ) || (final_value> float(max_value)))
        return float(10086.111f);

    return final_value;


}

int NanoDet::draw(cv::Mat& rgb, const std::vector<Object>& objects)
{
    static const char* class_names[] = {
            "zhizhen","yejing","fangxing"
    };

    static const unsigned char colors[19][3] = {
            { 54,  67, 244},
            { 99,  30, 233},
            {176,  39, 156},
            {183,  58, 103},
            {181,  81,  63},
            {243, 150,  33},
            {244, 169,   3},
            {212, 188,   0},
            {136, 150,   0},
            { 80, 175,  76},
            { 74, 195, 139},
            { 57, 220, 205},
            { 59, 235, 255},
            {  7, 193, 255},
            {  0, 152, 255},
            { 34,  87, 255},
            { 72,  85, 121},
            {158, 158, 158},
            {139, 125,  96}
    };

    int color_index = 0;

    for (size_t i = 0; i < objects.size(); i++)
    {
        const Object& obj = objects[i];

//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, cv::Rect(cv::Point(obj.x, obj.y), cv::Size(obj.w, obj.h)), cc, 2);

        char text[256];
        sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.x;
        int y = obj.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);
    }

    return 0;
}


int NanoDet::draw(cv::Mat& rgb, const Object & obj,float value)
{
    static const char* class_names[] = {
            "zhizhen","yejing","fangxing"
    };

    static const unsigned char colors[19][3] = {
        { 54,  67, 244},
        { 99,  30, 233},
        {176,  39, 156},
        {183,  58, 103},
        {181,  81,  63},
        {243, 150,  33},
        {244, 169,   3},
        {212, 188,   0},
        {136, 150,   0},
        { 80, 175,  76},
        { 74, 195, 139},
        { 57, 220, 205},
        { 59, 235, 255},
        {  7, 193, 255},
        {  0, 152, 255},
        { 34,  87, 255},
        { 72,  85, 121},
        {158, 158, 158},
        {139, 125,  96}
    };

    int color_index = 0;



//        if (obj.y/rgb.cols <0.7 && obj.y/rgb.cols >1.4)
//            continue;




        //float resuze = this->dushu(imageROI);
        float resuze = value;


//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

        const unsigned char* color = colors[color_index % 19];
        color_index++;

        cv::Scalar cc(color[0], color[1], color[2]);

        cv::rectangle(rgb, cv::Rect(cv::Point(obj.x, obj.y), cv::Size(obj.w, obj.h)), cc, 2);

        char text[256];

        //char ch[50];
        //memset(ch, '\0', sizeof ch);
        //sprintf(text, "value = %.3f, confidence = %.1f%%", resuze, obj.prob * 100);
        sprintf(text, "value = %.3f", resuze);

        //sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
        //sprintf(text, "%s %.1f%%", class_names[obj.label], resuze);

        int baseLine = 0;
        cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

        int x = obj.x;
        int y = obj.y - label_size.height - baseLine;
        if (y < 0)
            y = 0;
        if (x + label_size.width > rgb.cols)
            x = rgb.cols - label_size.width;

        cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

        cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

        cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);

    return 0;
}


int NanoDet::draw(cv::Mat& rgb, const Object & obj)
{
    static const char* class_names[] = {
            "zhizhen","yejing","fangxing"
    };

    static const unsigned char colors[19][3] = {
            { 54,  67, 244},
            { 99,  30, 233},
            {176,  39, 156},
            {183,  58, 103},
            {181,  81,  63},
            {243, 150,  33},
            {244, 169,   3},
            {212, 188,   0},
            {136, 150,   0},
            { 80, 175,  76},
            { 74, 195, 139},
            { 57, 220, 205},
            { 59, 235, 255},
            {  7, 193, 255},
            {  0, 152, 255},
            { 34,  87, 255},
            { 72,  85, 121},
            {158, 158, 158},
            {139, 125,  96}
    };

    int color_index = 0;



//        if (obj.y/rgb.cols <0.7 && obj.y/rgb.cols >1.4)
//            continue;




    //float resuze = this->dushu(imageROI);
    //float resuze = value;


//         fprintf(stderr, "%d = %.5f at %.2f %.2f %.2f x %.2f\n", obj.label, obj.prob,
//                 obj.rect.x, obj.rect.y, obj.rect.width, obj.rect.height);

    const unsigned char* color = colors[color_index % 19];
    color_index++;

    cv::Scalar cc(color[0], color[1], color[2]);

    cv::rectangle(rgb, cv::Rect(cv::Point(obj.x, obj.y), cv::Size(obj.w, obj.h)), cc, 2);

    char text[256];

    //char ch[50];
    //memset(ch, '\0', sizeof ch);
    //sprintf(text, "value = %.3f, confidence = %.1f%%", resuze, obj.prob * 100);

    //sprintf(text, "%s, confidence = %.1f%%", obj.label, obj.prob * 100);

    //sprintf(text, "%s %.1f%%", class_names[obj.label], obj.prob * 100);
    //sprintf(text, "%s %.1f%%", class_names[obj.label], resuze);

    int baseLine = 0;
    cv::Size label_size = cv::getTextSize(text, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

    int x = obj.x;
    int y = obj.y - label_size.height - baseLine;
    if (y < 0)
        y = 0;
    if (x + label_size.width > rgb.cols)
        x = rgb.cols - label_size.width;

    cv::rectangle(rgb, cv::Rect(cv::Point(x, y), cv::Size(label_size.width, label_size.height + baseLine)), cc, -1);

    cv::Scalar textcc = (color[0] + color[1] + color[2] >= 381) ? cv::Scalar(0, 0, 0) : cv::Scalar(255, 255, 255);

    cv::putText(rgb, text, cv::Point(x, y + label_size.height), cv::FONT_HERSHEY_SIMPLEX, 0.5, textcc, 1);

    return 0;
}


cv::Vec3d NanoDet::avg_circles(std::vector<cv::Vec3f> circles, int b){
    int avg_x=0;
    int avg_y=0;
    int avg_r=0;
    for (int i=0;  i< b; i++ )
    {
        //平均圆心 半径
        avg_x = avg_x + circles[i][0];
        avg_y = avg_y + circles[i][1];
        avg_r = avg_r + circles[i][2];
    }
    //半径为啥int
    avg_x = int(avg_x/(b));
    avg_y = int(avg_y/(b));
    avg_r = int(avg_r/(b));

    cv::Vec3d xyr = cv::Vec3d(avg_x, avg_y, avg_r);
    return xyr;

}

float NanoDet::getDist_P2L(cv::Point2f pointP, cv::Point2f pointA, cv::Point2f pointB)
{
    float A = 0, B = 0, C = 0;
    A = pointA.y - pointB.y;
    B = pointB.x - pointA.x;
    C = pointA.x*pointB.y - pointA.y*pointB.x;

    float distance = 0.0;
    distance = ((float)abs(A*pointP.x + B*pointP.y + C)) / ((float)sqrtf(A*A + B*B));
    return distance;
}


float NanoDet::dist_2_pts(int x1, int y1, int x2, int y2){
    int pp = pow(x2-x1,2)+pow(y2-y1,2);
    float tmp = sqrt(pp);
    return tmp;
}
cv::Mat NanoDet::region_of_interest(cv::Mat &img,std::vector<std::vector<cv::Point>> &vertices){
    cv::Mat mask = cv::Mat::zeros(img.size(), img.type());
    int match_mask_color= 255;

    cv::fillPoly(mask, vertices, cv::Scalar(match_mask_color));

    cv::Mat masked_image;
    cv::bitwise_and(img, mask,masked_image);

    return masked_image.clone();;


}