#pragma once

#include <opencv2/opencv.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/imgproc.hpp>
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
#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/core.hpp>
#include <vector>
#include <nn/autobackend.h>

// Define the skeleton and color mappings
std::vector<std::vector<int>> skeleton = { {16, 14}, {14, 12}, {17, 15}, {15, 13}, {12, 13}, {6, 12}, {7, 13}, {6, 7},
                                          {6, 8}, {7, 9}, {8, 10}, {9, 11}, {2, 3}, {1, 2}, {1, 3}, {2, 4}, {3, 5}, {4, 6}, {5, 7} };

std::vector<cv::Scalar> posePalette = {
        cv::Scalar(255, 128, 0), cv::Scalar(255, 153, 51), cv::Scalar(255, 178, 102), cv::Scalar(230, 230, 0), cv::Scalar(255, 153, 255),
        cv::Scalar(153, 204, 255), cv::Scalar(255, 102, 255), cv::Scalar(255, 51, 255), cv::Scalar(102, 178, 255), cv::Scalar(51, 153, 255),
        cv::Scalar(255, 153, 153), cv::Scalar(255, 102, 102), cv::Scalar(255, 51, 51), cv::Scalar(153, 255, 153), cv::Scalar(102, 255, 102),
        cv::Scalar(51, 255, 51), cv::Scalar(0, 255, 0), cv::Scalar(0, 0, 255), cv::Scalar(255, 0, 0), cv::Scalar(255, 255, 255)
};

std::vector<int> limbColorIndices = { 9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16 };
std::vector<int> kptColorIndices = { 16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9 };

cv::Scalar generateRandomColor(int numChannels) {
    if (numChannels < 1 || numChannels > 3) {
        throw std::invalid_argument("Invalid number of channels. Must be between 1 and 3.");
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dis(0, 255);

    cv::Scalar color;
    for (int i = 0; i < numChannels; i++) {
        color[i] = dis(gen); // for each channel separately generate value
    }

    return color;
}

std::vector<cv::Scalar> generateRandomColors(int class_names_num, int numChannels) {
    std::vector<cv::Scalar> colors;
    for (int i = 0; i < class_names_num; i++) {
        cv::Scalar color = generateRandomColor(numChannels);
        colors.push_back(color);
    }
    return colors;
}

// Функция для вычисления медианного значения из ненулевых элементов
double calculateMedianNonZero(const cv::Mat& flow) {
    std::vector<double> values;

    // Проверяем тип и размеры матрицы
    if (flow.type() != CV_32F && flow.type() != CV_64F) {
        std::cerr << "Error: Flow matrix must be of type CV_32F or CV_64F!" << std::endl;
        return 0.0;
    }

    // Извлекаем ненулевые значения из матрицы
    for (int y = 0; y < flow.rows; ++y) {
        for (int x = 0; x < flow.cols; ++x) {
            double value = flow.at<float>(y, x);
            if (value != 0.0) {
                values.push_back(value);
            }
        }
    }

    if (values.empty()) {
        return 0.0; // Возвращаем 0, если нет ненулевых значений
    }

    // Сортируем значения и находим медиану
    std::sort(values.begin(), values.end());
    size_t size = values.size();
    return (size % 2 == 0) ? (values[size / 2 - 1] + values[size / 2]) / 2.0 : values[size / 2];
}

double calculateMeanNonZero(const cv::Mat& mat) {
    std::vector<float> nonZeroValues;
    for (int i = 0; i < mat.rows; ++i) {
        for (int j = 0; j < mat.cols; ++j) {
            float value = mat.at<float>(i, j);
            if (value != 0) {
                nonZeroValues.push_back(value);
            }
        }
    }
    if (nonZeroValues.empty()) return 0;
    double sum = std::accumulate(nonZeroValues.begin(), nonZeroValues.end(), 0.0);
    return sum / nonZeroValues.size();
}

//void plot_keypoints(cv::Mat& image, const std::vector<std::vector<float>>& keypoints, const cv::Size& shape) {
void plot_keypoints(cv::Mat& image, const std::vector<YoloResults>& results, const cv::Size& shape) {

    int radius = 5;
    bool drawLines = true;

    if (results.empty()) {
        return;
    }

    std::vector<cv::Scalar> limbColorPalette;
    std::vector<cv::Scalar> kptColorPalette;

    for (int index : limbColorIndices) {
        limbColorPalette.push_back(posePalette[index]);
    }

    for (int index : kptColorIndices) {
        kptColorPalette.push_back(posePalette[index]);
    }

    for (const auto& res : results) {
        auto keypoint = res.keypoints;
        bool isPose = keypoint.size() == 51;  // numKeypoints == 17 && keypoints[0].size() == 3;
        drawLines &= isPose;

        // draw points
        for (int i = 0; i < 17; i++) {
            int idx = i * 3;
            int x_coord = static_cast<int>(keypoint[idx]);
            int y_coord = static_cast<int>(keypoint[idx + 1]);

            if (x_coord % shape.width != 0 && y_coord % shape.height != 0) {
                if (keypoint.size() == 3) {
                    float conf = keypoint[2];
                    if (conf < 0.5) {
                        continue;
                    }
                }
                cv::Scalar color_k = isPose ? kptColorPalette[i] : cv::Scalar(0, 0,
                    255);  // Default to red if not in pose mode
                cv::circle(image, cv::Point(x_coord, y_coord), radius, color_k, -1, cv::LINE_AA);
            }
        }
        // draw lines
        if (drawLines) {
            for (int i = 0; i < skeleton.size(); i++) {
                const std::vector<int>& sk = skeleton[i];
                int idx1 = sk[0] - 1;
                int idx2 = sk[1] - 1;

                int idx1_x_pos = idx1 * 3;
                int idx2_x_pos = idx2 * 3;

                int x1 = static_cast<int>(keypoint[idx1_x_pos]);
                int y1 = static_cast<int>(keypoint[idx1_x_pos + 1]);
                int x2 = static_cast<int>(keypoint[idx2_x_pos]);
                int y2 = static_cast<int>(keypoint[idx2_x_pos + 1]);

                float conf1 = keypoint[idx1_x_pos + 2];
                float conf2 = keypoint[idx2_x_pos + 2];

                // Check confidence thresholds
                if (conf1 < 0.5 || conf2 < 0.5) {
                    continue;
                }

                // Check if positions are within bounds
                if (x1 % shape.width == 0 || y1 % shape.height == 0 || x1 < 0 || y1 < 0 ||
                    x2 % shape.width == 0 || y2 % shape.height == 0 || x2 < 0 || y2 < 0) {
                    continue;
                }

                // Draw a line between keypoints
                cv::Scalar color_limb = limbColorPalette[i];
                cv::line(image, cv::Point(x1, y1), cv::Point(x2, y2), color_limb, 2, cv::LINE_AA);
            }
        }
    }
}

void plot_results(cv::Mat img, std::vector<YoloResults>& results,
    std::vector<cv::Scalar> color, std::unordered_map<int, std::string>& names,
    const cv::Size& shape
) {

    cv::Mat mask = img.clone();

    int radius = 5;
    bool drawLines = true;

    auto raw_image_shape = img.size();
    std::vector<cv::Scalar> limbColorPalette;
    std::vector<cv::Scalar> kptColorPalette;

    for (int index : limbColorIndices) {
        limbColorPalette.push_back(posePalette[index]);
    }

    for (int index : kptColorIndices) {
        kptColorPalette.push_back(posePalette[index]);
    }

    for (const auto& res : results) {
        float left = res.bbox.x;
        float top = res.bbox.y;
        int color_num = res.class_idx;
        if (names.find(res.class_idx) != names.end()) {
            // Draw bounding box
            rectangle(img, res.bbox, color[res.class_idx], 2);

            // Try to get the class name corresponding to the given class_idx
            std::string class_name;
            auto it = names.find(res.class_idx);
            if (it != names.end()) {
                class_name = it->second;
            }
            else {
                std::cerr << "Warning: class_idx not found in names for class_idx = " << res.class_idx << std::endl;
                // Then convert it to a string anyway
                class_name = std::to_string(res.class_idx);

            }

            // Draw mask if available
            if (res.mask.rows && res.mask.cols > 0) {
                mask(res.bbox).setTo(color[res.class_idx], res.mask);
            }

            // Create label
            std::stringstream labelStream;
            labelStream << class_name << " " << std::fixed << std::setprecision(2) << res.conf;
            std::string label = labelStream.str();

            cv::Size text_size = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.6, 2, nullptr);
            cv::Rect rect_to_fill(left - 1, top - text_size.height - 5, text_size.width + 2, text_size.height + 5);
            cv::Scalar text_color = cv::Scalar(255.0, 255.0, 255.0);
            rectangle(img, rect_to_fill, color[res.class_idx], -1);
            putText(img, label, cv::Point(left - 1.5, top - 2.5), cv::FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2);
        }
    }

    // Combine the image and mask
    addWeighted(img, 0.6, mask, 0.4, 0, img);
    //    resize(img, img, img.size());
    //    resize(img, img, shape);
    //    // Show the image
    //    imshow("img", img);
    //    cv::waitKey();
}

void clip_boxes(cv::Rect& box, const cv::Size& shape) {
    box.x = std::max(0, std::min(box.x, shape.width));
    box.y = std::max(0, std::min(box.y, shape.height));
    box.width = std::max(0, std::min(box.width, shape.width - box.x));
    box.height = std::max(0, std::min(box.height, shape.height - box.y));
}

void clip_boxes(cv::Rect_<float>& box, const cv::Size& shape) {
    box.x = std::max(0.0f, std::min(box.x, static_cast<float>(shape.width)));
    box.y = std::max(0.0f, std::min(box.y, static_cast<float>(shape.height)));
    box.width = std::max(0.0f, std::min(box.width, static_cast<float>(shape.width - box.x)));
    box.height = std::max(0.0f, std::min(box.height, static_cast<float>(shape.height - box.y)));
}


void clip_boxes(std::vector<cv::Rect>& boxes, const cv::Size& shape) {
    for (cv::Rect& box : boxes) {
        clip_boxes(box, shape);
    }
}

void clip_boxes(std::vector<cv::Rect_<float>>& boxes, const cv::Size& shape) {
    for (cv::Rect_<float>& box : boxes) {
        clip_boxes(box, shape);
    }
}

// source: ultralytics/utils/ops.py scale_boxes lines 99+ (ultralytics==8.0.160)
cv::Rect_<float> scale_boxes(const cv::Size& img1_shape, cv::Rect_<float>& box, const cv::Size& img0_shape,
    std::pair<float, cv::Point2f> ratio_pad = std::make_pair(-1.0f, cv::Point2f(-1.0f, -1.0f)), bool padding = true) {

    float gain, pad_x, pad_y;

    if (ratio_pad.first < 0.0f) {
        gain = std::min(static_cast<float>(img1_shape.height) / static_cast<float>(img0_shape.height),
            static_cast<float>(img1_shape.width) / static_cast<float>(img0_shape.width));
        pad_x = roundf((img1_shape.width - img0_shape.width * gain) / 2.0f - 0.1f);
        pad_y = roundf((img1_shape.height - img0_shape.height * gain) / 2.0f - 0.1f);
    }
    else {
        gain = ratio_pad.first;
        pad_x = ratio_pad.second.x;
        pad_y = ratio_pad.second.y;
    }

    //cv::Rect scaledCoords(box);
    cv::Rect_<float> scaledCoords(box);

    if (padding) {
        scaledCoords.x -= pad_x;
        scaledCoords.y -= pad_y;
    }

    scaledCoords.x /= gain;
    scaledCoords.y /= gain;
    scaledCoords.width /= gain;
    scaledCoords.height /= gain;

    // Clip the box to the bounds of the image
    clip_boxes(scaledCoords, img0_shape);

    return scaledCoords;
}


//void clip_coords(cv::Mat& coords, const cv::Size& shape) {
//    // Clip x-coordinates to the image width
//    cv::Mat xCoords = coords.col(0);
//    cv::Mat yCoords = coords.col(1);
//
//    for (int i = 0; i < coords.rows; ++i) {
//        xCoords.at<float>(i) = std::max(std::min(xCoords.at<float>(i), static_cast<float>(shape.width - 1)), 0.0f);
//        yCoords.at<float>(i) = std::max(std::min(yCoords.at<float>(i), static_cast<float>(shape.height - 1)), 0.0f);
//    }
//}

void clip_coords(std::vector<float>& coords, const cv::Size& shape) {
    // Assuming coords are of shape [1, 17, 3]
    for (int i = 0; i < coords.size(); i += 3) {
        coords[i] = std::min(std::max(coords[i], 0.0f), static_cast<float>(shape.width - 1));  // x
        coords[i + 1] = std::min(std::max(coords[i + 1], 0.0f), static_cast<float>(shape.height - 1));  // y
    }
}

// source: ultralytics/utils/ops.py scale_coords lines 753+ (ultralytics==8.0.160)
//cv::Mat scale_coords(const cv::Size& img1_shape, cv::Mat& coords, const cv::Size& img0_shape)
//cv::Mat scale_coords(const cv::Size& img1_shape, std::vector<float> coords, const cv::Size& img0_shape)
std::vector<float> scale_coords(const cv::Size& img1_shape, std::vector<float>& coords, const cv::Size& img0_shape)
{
//    cv::Mat scaledCoords = coords.clone();
    std::vector<float> scaledCoords = coords;

    // Calculate gain and padding
    double gain = std::min(static_cast<double>(img1_shape.width) / img0_shape.width, static_cast<double>(img1_shape.height) / img0_shape.height);
    cv::Point2d pad((img1_shape.width - img0_shape.width * gain) / 2, (img1_shape.height - img0_shape.height * gain) / 2);

    // Apply padding
//    scaledCoords.col(0) = (scaledCoords.col(0) - pad.x);
//    scaledCoords.col(1) = (scaledCoords.col(1) - pad.y);
    // Assuming coords are of shape [1, 17, 3]
    for (int i = 0; i < scaledCoords.size(); i += 3) {
        scaledCoords[i] -= pad.x;  // x padding
        scaledCoords[i + 1] -= pad.y;  // y padding
    }

    // Scale coordinates
//    scaledCoords.col(0) /= gain;
//    scaledCoords.col(1) /= gain;
    // Assuming coords are of shape [1, 17, 3]
    for (int i = 0; i < scaledCoords.size(); i += 3) {
        scaledCoords[i] /= gain;
        scaledCoords[i + 1] /= gain;
    }

    clip_coords(scaledCoords, img0_shape);
    return scaledCoords;
}


cv::Mat crop_mask(const cv::Mat& mask, const cv::Rect& box) {
    int h = mask.rows;
    int w = mask.cols;

    int x1 = box.x;
    int y1 = box.y;
    int x2 = box.x + box.width;
    int y2 = box.y + box.height;

    cv::Mat cropped_mask = cv::Mat::zeros(h, w, mask.type());

    for (int r = 0; r < h; ++r) {
        for (int c = 0; c < w; ++c) {
            if (r >= y1 && r < y2 && c >= x1 && c < x2) {
                cropped_mask.at<float>(r, c) = mask.at<float>(r, c);
            }
        }
    }

    return cropped_mask;
}

//std::tuple<std::vector<cv::Rect_<float>>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
std::tuple<std::vector<cv::Rect>, std::vector<float>, std::vector<int>, std::vector<std::vector<float>>>
non_max_suppression(const cv::Mat& output0, int class_names_num, int data_width, double conf_threshold,
                    float iou_threshold) {

    std::vector<int> class_ids;
    std::vector<float> confidences;
//    std::vector<cv::Rect_<float>> boxes;
    std::vector<cv::Rect> boxes;
    std::vector<std::vector<float>> rest;

    int rest_start_pos = class_names_num + 4;
    int rest_features = data_width - rest_start_pos;
//    int data_width = rest_start_pos + total_features_num;

    int rows = output0.rows;
    float* pdata = (float*) output0.data;

    for (int r = 0; r < rows; ++r) {
        cv::Mat scores(1, class_names_num, CV_32FC1, pdata + 4);
        cv::Point class_id;
        double max_conf;
        minMaxLoc(scores, nullptr, &max_conf, nullptr, &class_id);

        if (max_conf > conf_threshold) {
            std::vector<float> mask_data(pdata + 4 + class_names_num, pdata + data_width);
            class_ids.push_back(class_id.x);
            confidences.push_back((float) max_conf);

            float out_w = pdata[2];
            float out_h = pdata[3];
            float out_left = MAX((pdata[0] - 0.5 * out_w + 0.5), 0);
            float out_top = MAX((pdata[1] - 0.5 * out_h + 0.5), 0);
            cv::Rect_<float> bbox(out_left, out_top, (out_w + 0.5), (out_h + 0.5));
            boxes.push_back(bbox);
            if (rest_features > 0) {
                std::vector<float> rest_data(pdata + rest_start_pos, pdata + data_width);
                rest.push_back(rest_data);
            }
        }
        pdata += data_width; // next prediction
    }

    //
    //float masks_threshold = 0.50;
    //int top_k = 500;
    //const float& nmsde_eta = 1.0f;
    std::vector<int> nms_result;
    cv::dnn::NMSBoxes(boxes, confidences, conf_threshold, iou_threshold, nms_result); // , nms_eta, top_k);
//    cv::dnn::NMSBoxes(boxes, confidences, );
    std::vector<int> nms_class_ids;
    std::vector<float> nms_confidences;
//    std::vector<cv::Rect_<float>> boxes;
    std::vector<cv::Rect> nms_boxes;
    std::vector<std::vector<float>> nms_rest;
    for (int idx: nms_result) {
        nms_class_ids.push_back(class_ids[idx]);
        nms_confidences.push_back(confidences[idx]);
        nms_boxes.push_back(boxes[idx]);
        nms_rest.push_back(rest[idx]);
    }
    return std::make_tuple(nms_boxes, nms_confidences, nms_class_ids, nms_rest);
}
