#ifndef SIMPLE_FIR_HPP
#define SIMPLE_FIR_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

cv::Mat naive_denoise_image(cv::Mat image, cv::Size kernel_size);
cv::Mat naive_separable_denoise_image(cv::Mat image, cv::Size kernel_size);
cv::Mat naive_filter_image(cv::Mat image, cv::Mat kernel);
cv::Mat opencv_denoise_image(cv::Mat image, cv::Size kernel_size);
cv::Mat crop_image_by_kernel(cv::Mat image, cv::Size kernel_size);
cv::Mat add_noise_to_image(cv::Mat image, float noise_strength);
void show_image(std::string window_name, cv::Mat image, bool normalize=true);

#endif
