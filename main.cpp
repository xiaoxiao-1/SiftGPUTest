// SiftGPU模块
#include <SiftGPU.h>

// 标准C++
#include <iostream>
#include <vector>

// OpenCV图像
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

// boost库中计时函数
#include <boost/timer.hpp>

// OpenGL
#include <GL/gl.h>

using namespace std;

int main(int argc, char **argv)
{
    // 声明SiftGPU并初始化
    SiftGPU sift;
    char *myargv[4] = {"-fo", "-1", "-v", "1"};
    sift.ParseParam(4, myargv);

    // 检查硬件是否支持SiftGPU
    int support = sift.CreateContextGL();
    if (support != SiftGPU::SIFTGPU_FULL_SUPPORTED)
    {
        cerr << "SiftGPU is not supported!" << endl;
        return 2;
    }

    // 测试直接读取一张图像
    cout << "running sift" << endl;
    boost::timer timer;

    // 先用OpenCV读取一个图像，然后调用SiftGPU提取特征
    cv::Mat color_img = cv::imread("../data/rgb1.png");
    cv::Mat img = cv::imread("../data/rgb1.png", 0);

    cv::Mat color_img_2 = cv::imread("../data/rgb2.png");
    cv::Mat img_2 = cv::imread("../data/rgb2.png", 0);

    int width = img.cols;
    int height = img.rows;
    timer.restart();
    // 注意我们处理的是灰度图，故照如下设置
    sift.RunSIFT("../data/rgb1.png");
    cout << "siftgpu::runSIFT() cost time=" << timer.elapsed() << endl;

    // 获取特征点数目
    int feature_num = sift.GetFeatureNum();
    cout << "keypoint number of the image is:" << feature_num << std::endl;

    // 获取特征点和特征点描述子
    std::vector<SiftGPU::SiftKeypoint> kpv_1;
    std::vector<float> descv_1;
    kpv_1.resize(feature_num);
    descv_1.resize(128 * feature_num);
    sift.GetFeatureVector(&kpv_1[0], &descv_1[0]);

    // 将SiftGPU格式的特征点和描述子变为opencv格式
    std::vector<cv::KeyPoint> cv_kpv_1;
    // cv::Mat descripotrs(feature_num, 128, CV_32F, &descv[0]);
    for (size_t i = 0; i < feature_num; i++)
    {
        cv::KeyPoint kp(kpv_1[i].x, kpv_1[i].y, kpv_1[i].s);
        cv_kpv_1.push_back(kp);
    }

    if (cv_kpv_1.size() != feature_num)
    {
        std::cerr << "cv_kpv size don't equal to feature_num!" << std::endl;
    }
    // 显示特征点
    cv::Mat draw_image;
    cv::drawKeypoints(color_img, cv_kpv_1, draw_image, cv::Scalar(0, 0, 255));
    cv::imshow("keypoints", draw_image);
    cv::waitKey(0);

    // 开始处理第二章图片
    //  注意我们处理的是灰度图，故照如下设置
    sift.RunSIFT("../data/rgb2.png");
    cout << "siftgpu::runSIFT() cost time=" << timer.elapsed() << endl;

    // 获取特征点数目
    int feature_num_2 = sift.GetFeatureNum();
    cout << "keypoint number of the image is:" << feature_num_2 << std::endl;

    // 获取特征点和特征点描述子
    std::vector<SiftGPU::SiftKeypoint> kpv_2;
    std::vector<float> descv_2;
    kpv_2.resize(feature_num_2);
    descv_2.resize(128 * feature_num_2);
    sift.GetFeatureVector(&kpv_2[0], &descv_2[0]);

    // 将SiftGPU格式的特征点和描述子变为opencv格式
    std::vector<cv::KeyPoint> cv_kpv_2;
    // cv::Mat descripotrs(feature_num, 128, CV_32F, &descv[0]);
    for (size_t i = 0; i < feature_num_2; i++)
    {
        cv::KeyPoint kp(kpv_2[i].x, kpv_2[i].y, kpv_2[i].s);
        cv_kpv_2.push_back(kp);
    }

    if (cv_kpv_2.size() != feature_num_2)
    {
        std::cerr << "cv_kpv size don't equal to feature_num!" << std::endl;
    }
    // 显示特征点
    cv::Mat draw_image_2;
    cv::drawKeypoints(color_img_2, cv_kpv_2, draw_image_2, cv::Scalar(0, 0, 255));
    cv::imshow("keypoints", draw_image_2);
    cv::waitKey(0);

    // 开始进行特征点匹配
    SiftMatchGPU matcher;
    matcher.VerifyContextGL();
    // 设置需要匹配的描述符
    matcher.SetDescriptors(0, feature_num, &descv_1[0]);
    matcher.SetDescriptors(1, feature_num_2, &descv_2[0]);
    //设置用于存储匹配结果的buff
    int (*match_buf)[2]=new int[feature_num][2];
    int num_match=matcher.GetSiftMatch(feature_num,match_buf);

    //设置opencv的match对象
    std::vector<cv::DMatch> matchs;
    matchs.resize(num_match);
    for(int i=0;i<num_match;i++){
        cv::DMatch match;
        match.queryIdx=match_buf[i][0];
        match.trainIdx=match_buf[i][1];
        matchs[i]=match;
    }
    cv::Mat match_image;
    cv::drawMatches(color_img,cv_kpv_1,color_img_2,cv_kpv_2,matchs,match_image);
    cv::imshow("match",match_image);
    cv::waitKey(0);

    return 0;
}