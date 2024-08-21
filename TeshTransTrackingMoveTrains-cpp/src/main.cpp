#include <random>

#include <filesystem>
#include "nn/onnx_model_base.h"
#include "nn/autobackend.h"
#include <opencv2/opencv.hpp>
#include <vector>
#include <algorithm> 
#include <numeric> 
#include "utils/augment.h"
#include "constants.h"
#include "utils/common.h"
#include "utils/ops.h"
namespace fs = std::filesystem;
using namespace std;

namespace fs = std::filesystem;

void initializeOpenCV() {
    cv::setUseOptimized(true);
    cv::setNumThreads(cv::getNumberOfCPUs());
}

cv::VideoCapture openVideo(const std::string& video_path) {
    cv::VideoCapture cap(video_path);
    if (!cap.isOpened()) {
        std::cerr << "Error: Unable to open video file" << std::endl;
        exit(1);
    }
    return cap;
}

cv::VideoWriter prepareOutputVideo(const cv::VideoCapture& cap, const std::string& video_path, double scaleFactor) {
    int frame_width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
    int frame_height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
    double fps = cap.get(cv::CAP_PROP_FPS);

    fs::path videoFilePath(video_path);
    fs::path newFilePath = videoFilePath.stem();
    newFilePath += "-cpp-yolo";
    newFilePath += videoFilePath.extension();
    assert(newFilePath != videoFilePath);

    cv::Size newSize(static_cast<int>(frame_width * scaleFactor), static_cast<int>(frame_height * scaleFactor));
    return cv::VideoWriter(newFilePath.string(), cv::VideoWriter::fourcc('M', 'P', '4', 'V'), 24, cv::Size(newSize.width, newSize.height * 3));
}

std::vector<cv::Scalar> generateColors(AutoBackendOnnx& model) {
    return generateRandomColors(model.getNc(), model.getCh());
}

void processFrame(cv::Mat& prevFrame, cv::Mat& currentFrame, AutoBackendOnnx& model, const std::vector<cv::Scalar>& colors,
    std::unordered_map<int, std::string>& names, float conf_threshold, float iou_threshold, float mask_threshold,
    int conversion_code, cv::VideoWriter& outputVideo, double scaleFactor, bool b_gtx) {

    cv::Mat bboxFrame = currentFrame.clone();
    std::vector<YoloResults> results = model.predict_once(currentFrame, conf_threshold, iou_threshold, mask_threshold, conversion_code);
    cv::Size show_shape = currentFrame.size();

    plot_results(bboxFrame, results, colors, names, show_shape);

    cv::Mat grayPrev, grayCurrent;
    cv::cvtColor(prevFrame, grayPrev, cv::COLOR_BGR2GRAY);
    cv::cvtColor(currentFrame, grayCurrent, cv::COLOR_BGR2GRAY);

    cv::Mat flow;
    cv::calcOpticalFlowFarneback(grayPrev, grayCurrent, flow, 0.3, 3, 15, 3, 5, 1.1, 0);

    cv::Mat objectFlowX, objectFlowY;
    std::vector<cv::Mat> flowPlanes(2);
    cv::split(flow, flowPlanes);
    cv::Mat flowX = flowPlanes[0];
    cv::Mat flowY = flowPlanes[1];

    cv::Mat compensatedFlow = cv::Mat::zeros(currentFrame.size(), CV_32FC3);


    for (const YoloResults& result : results) {
        if (names.find(result.class_idx) != names.end()) {
            cv::Mat mask = result.mask;

            cv::Mat mask8U;
            mask.convertTo(mask8U, CV_8UC1);

            cv::Mat full_mask = cv::Mat::zeros(flowX.size(), CV_8UC1);
            cv::Rect roi(result.bbox.x, result.bbox.y, mask.cols, mask.rows);

            mask8U.copyTo(full_mask(roi));

            cv::Mat objectFlowX, objectFlowY;
            flowX.copyTo(objectFlowX, full_mask);
            flowY.copyTo(objectFlowY, full_mask);

            cv::Mat backgroundMask;
            cv::bitwise_not(full_mask, backgroundMask);
            cv::Mat backgroundFlowX, backgroundFlowY;

            flowX.copyTo(backgroundFlowX, backgroundMask);
            flowY.copyTo(backgroundFlowY, backgroundMask);

            double medianBackgroundFlowX = calculateMedianNonZero(backgroundFlowX);
            double medianBackgroundFlowY = calculateMedianNonZero(backgroundFlowY);

            cv::Mat compensatedFlowX = objectFlowX; //- medianBackgroundFlowX;
            cv::Mat compensatedFlowY = objectFlowY;//- medianBackgroundFlowY;

            cv::Mat compensatedFlowX_Float, compensatedFlowY_Float;
            compensatedFlowX.convertTo(compensatedFlowX_Float, CV_32F);
            compensatedFlowY.convertTo(compensatedFlowY_Float, CV_32F);

            std::vector<cv::Mat> compensatedFlowChannels = { compensatedFlowX_Float, compensatedFlowY_Float, cv::Mat::zeros(compensatedFlowX_Float.size(), CV_32F) };
            cv::Mat mergedFlow;
            cv::merge(compensatedFlowChannels, mergedFlow);

            compensatedFlow += mergedFlow;

            double meanFlowX = calculateMeanNonZero(compensatedFlowX) - medianBackgroundFlowX;
            double meanFlowY = calculateMeanNonZero(compensatedFlowY) - medianBackgroundFlowY;

            cv::Moments M = cv::moments(full_mask, true);
            if (M.m00 != 0) {
                int centerX = static_cast<int>(M.m10 / M.m00);
                int centerY = static_cast<int>(M.m01 / M.m00);
                cv::arrowedLine(bboxFrame, cv::Point(centerX, centerY),
                    cv::Point(centerX + static_cast<int>(meanFlowX * 12),
                        centerY + static_cast<int>(meanFlowY * 12)),
                    cv::Scalar(100, 0, 0), 4, cv::LINE_AA, 0, 0.5);
            }
        }
    }

    cv::Mat mag, ang;
    cv::cartToPolar(flowX, flowY, mag, ang, false);
    cv::Mat hsv(currentFrame.size(), CV_32FC3);

    std::vector<cv::Mat> hsvPlanes(3);
    hsvPlanes[0] = ang * (180.0 / CV_PI / 2.0);
    hsvPlanes[1] = cv::Mat::ones(ang.size(), CV_32F) * 255.0;
    cv::normalize(mag, hsvPlanes[2], 0, 255, cv::NORM_MINMAX);

    cv::merge(hsvPlanes, hsv);

    cv::Mat hsv8;
    hsv.convertTo(hsv8, CV_8UC3);
    cv::Mat opticalFlowColor;
    cv::cvtColor(hsv8, opticalFlowColor, cv::COLOR_HSV2BGR);

    cv::Mat compensatedFlowDisplay;
    cv::normalize(compensatedFlow, compensatedFlowDisplay, 0, 255, cv::NORM_MINMAX);
    compensatedFlowDisplay.convertTo(compensatedFlowDisplay, CV_8UC3);

    cv::Mat resizedBBoxFrame, resizedOpticalFlowColor, resizedCompensatedFlowDisplay;
    cv::resize(bboxFrame, resizedBBoxFrame, cv::Size(), scaleFactor, scaleFactor);
    cv::resize(opticalFlowColor, resizedOpticalFlowColor, cv::Size(), scaleFactor, scaleFactor);
    cv::resize(compensatedFlowDisplay, resizedCompensatedFlowDisplay, cv::Size(), scaleFactor, scaleFactor);

    std::vector<cv::Mat> images = { resizedBBoxFrame, resizedOpticalFlowColor, resizedCompensatedFlowDisplay };
    cv::Mat combinedVertical;
    cv::vconcat(images, combinedVertical);

    outputVideo.write(combinedVertical);
    if (b_gtx) {
        cv::imshow("Combined View", combinedVertical);
    }
}

int main(int argc, char* argv[]) {
    initializeOpenCV();
    if (argc < 5) {
        std::cerr << "Usage: program <video_path> <model_path> <conf> <boolean gtx>" << std::endl;
        return -1;
    }

    std::string video_path = argv[1];
    const std::string& modelPath = argv[2];
    float conf_threshold = std::stof(argv[3]);  // Преобразование строки в float
    bool b_gtx = std::stoi(argv[4]);  // Преобразование строки в int и конвертация в bool

    cout << "video_path: " << video_path << endl;
    cout << "modelPath: " << modelPath << endl;
    cout << "b_gtx: " << b_gtx << endl;
    cout << "conf_threshold: " << conf_threshold << endl;

    const std::string& onnx_provider = OnnxProviders::CUDA;
    const std::string& onnx_logid = "yolov8_inference2";
    float mask_threshold = 0.5f;

    float iou_threshold = 0.45f;
    int conversion_code = cv::COLOR_BGR2RGB;
    double scaleFactor = 0.5;

    cv::VideoCapture cap = openVideo(video_path);
    cv::VideoWriter outputVideo = prepareOutputVideo(cap, video_path, scaleFactor);

    AutoBackendOnnx model(modelPath.c_str(), onnx_logid.c_str(), onnx_provider.c_str());
    std::vector<cv::Scalar> colors = generateColors(model);
    std::unordered_map<int, std::string> names = { {6, "train"} };

    cv::Mat prevFrame, currentFrame;
    cap >> prevFrame;

    while (cap.read(currentFrame)) {
        if (currentFrame.empty()) {
            std::cerr << "Error: Empty frame detected" << std::endl;
            break;
        }

        processFrame(prevFrame, currentFrame, model, colors, names, conf_threshold, iou_threshold, mask_threshold, conversion_code, outputVideo, scaleFactor, b_gtx);

        prevFrame = currentFrame.clone();

        if (b_gtx && cv::waitKey(1) == 'q') {
            break;
        }
    }

    cap.release();
    outputVideo.release();
    if (b_gtx) {
        cv::destroyAllWindows();
    }
    return 0;
}