#ifndef SIMPLE_FIR_HPP
#define SIMPLE_FIR_HPP

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "opencv2/imgproc/imgproc.hpp"

cv::Mat opencv_denoise_image(cv::Mat image, cv::Size kernel_size);
cv::Mat crop_opencv_denoised_image(cv::Mat image, cv::Size kernel_size);
cv::Mat add_noise_to_image(cv::Mat image, float noise_strength);
void show_image(std::string window_name, cv::Mat image);

#endif
