
#include <iostream>
#include <getopt.h>
#include <chrono>

#include "simple_fir.hpp"

using namespace std;

int main(int argc, char **argv)
{
  // Arguments handling
  // ###########################################################################
  static struct option long_options[] = {
  {"input_image", required_argument, 0, 'i'},
  {"noise_strength", required_argument, 0, 'n'},
  {"kernel_size", required_argument, 0, 'k'},
  {0, 0, 0, 0}};

  string input_image;
  float noise_strength;
  int k_size;
  cv::Size kernel_size;

  int option_index = 0;
  int opt = 0;
  while ((opt = getopt_long(argc, argv, "i:n:k:", long_options, &option_index)) != -1)
  {
    switch (opt)
    {
      case 'i':
        input_image = optarg;
        cout << "Input image: " << optarg << endl;
        break;

      case 'n':
        noise_strength = stof(optarg);
        cout << "Noise strength: " << optarg << endl;
        break;

      case 'k':
        k_size = stoi(optarg);
        kernel_size = cv::Size(k_size, k_size);
        cout << "Kenel size: " << optarg << endl;
        break;

      default:
        break;
    }
  }

  if (input_image.empty() || noise_strength < 0.0 || noise_strength > 1.0 ||
      k_size % 2 == 0 || k_size < 0)
  {
    cout << endl;
    cout << "No input image or noise strength below zero/above one or even/negative kernel size." << endl;
    return -1;
  }

  cv::Mat image = imread(input_image, cv::IMREAD_GRAYSCALE);

  if(!image.data)
  {
    cout <<  "Could not open or find the image" << endl ;
    return -1;
  }
  cout << endl;
  // ###########################################################################


  // Add noise to image
  // ###########################################################################
  cv::Mat noised_image = add_noise_to_image(image, noise_strength);
  // ###########################################################################


  // OpenCV denoise
  // ###########################################################################
  auto opencv_denoise_begin = chrono::high_resolution_clock::now();

  cv::Mat opencv_denoised_image = opencv_denoise_image(noised_image, kernel_size);
  opencv_denoised_image = crop_image_by_kernel(opencv_denoised_image, kernel_size);

  auto opencv_denoise_end = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> opencv_denoise_time = opencv_denoise_end - opencv_denoise_begin;
  cout << "OpenCV denoise in sec: " << (opencv_denoise_time.count() / 1000.0) << endl;
  // ###########################################################################


  // Naive denoise
  // ###########################################################################
  auto naive_denoise_begin = chrono::high_resolution_clock::now();

  cv::Mat naive_denoised_image = naive_denoise_image(noised_image, kernel_size);

  auto naive_denoise_end = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> naive_denoise_time = naive_denoise_end - naive_denoise_begin;
  cout << "Naive denoise in sec: " << (naive_denoise_time.count() / 1000.0) << endl;
  // ###########################################################################


  // Naive separable denoise
  // ###########################################################################
  auto naive_separable_denoise_begin = chrono::high_resolution_clock::now();

  cv::Mat naive_separable_denoised_image = naive_separable_denoise_image(noised_image, kernel_size);

  auto naive_separable_denoise_end = chrono::high_resolution_clock::now();
  chrono::duration<double, std::milli> naive_separable_denoise_time = naive_separable_denoise_end - naive_separable_denoise_begin;
  cout << "Naive separable denoise in sec: " << (naive_separable_denoise_time.count() / 1000.0) << endl;
  // ###########################################################################


  // Absolute difference between OpenCV and Naive denoised images
  // ###########################################################################
  cv::Mat diff_OpenCV_naive;
  cv::absdiff(opencv_denoised_image, naive_denoised_image, diff_OpenCV_naive);
  cv::Mat diff_OpenCV_naive_separable;
  cv::absdiff(opencv_denoised_image, naive_separable_denoised_image, diff_OpenCV_naive_separable);

  float sum_of_diff = cv::sum(diff_OpenCV_naive)[0];
  float sum_of_diff_separable = cv::sum(diff_OpenCV_naive_separable)[0];
  int number_of_pixels = opencv_denoised_image.rows * opencv_denoised_image.cols;
  cout << endl;
  cout << "Difference between OpenCV and naive approach (mean per pixel): " << sum_of_diff / number_of_pixels << endl;
  cout << "Difference between OpenCV and naive separable approach (mean per pixel): " << sum_of_diff_separable / number_of_pixels << endl;
  // ###########################################################################


  // Show output images
  // ###########################################################################
  show_image("Input Image", image);
  show_image("Noised Image", noised_image);
  show_image("OpenCV Denoised Image", opencv_denoised_image);
  show_image("Naive Denoised Image", naive_denoised_image);
  show_image("Naive Separable Denoised Image", naive_denoised_image);
  show_image("Difference between OpenCV and naive approach", diff_OpenCV_naive, false);
  show_image("Difference between OpenCV and naive separable approach", diff_OpenCV_naive_separable, false);
  // ###########################################################################

  cv::waitKey(0);

  return 0;
}


cv::Mat naive_denoise_image(cv::Mat image, cv::Size kernel_size)
{
  cv::Mat gaussian_kernel = cv::getGaussianKernel(kernel_size.height, 0, CV_32F);
  cv::Mat gaussian_kernel_2D = gaussian_kernel * gaussian_kernel.t();
  cv::Mat denoised_image = naive_filter_image(image, gaussian_kernel_2D);
  return denoised_image;
}


cv::Mat naive_separable_denoise_image(cv::Mat image, cv::Size kernel_size)
{
  cv::Mat gaussian_kernel = cv::getGaussianKernel(kernel_size.height, 0, CV_32F);
  cv::Mat denoised_image = naive_filter_image(image, gaussian_kernel);
  denoised_image = naive_filter_image(denoised_image, gaussian_kernel.t());
  return denoised_image;
}


cv::Mat naive_filter_image(cv::Mat image, cv::Mat kernel)
{
  cv::Mat output_image(image.rows, image.cols, CV_32F);
  output_image = crop_image_by_kernel(output_image, kernel.size());
  for (int x_img = 0; x_img < image.cols - kernel.cols; ++x_img)
  {
    for (int y_img = 0; y_img < image.rows - kernel.rows; ++y_img)
    {
      float kernel_sum = 0;
      for (int x = 0; x < kernel.cols; ++x)
      {
        for (int y = 0; y < kernel.rows; ++y)
        {
          kernel_sum += image.at<float>(y_img + y, x_img + x) * kernel.at<float>(y, x);
        }
      }
      output_image.at<float>(y_img, x_img) = kernel_sum;
    }
  }
  return output_image;
}


// use OpenCV to denoise image
cv::Mat opencv_denoise_image(cv::Mat image, cv::Size kernel_size)
{
  cv::Mat denoised_image;
  GaussianBlur(image, denoised_image, kernel_size, 0, 0, cv::BORDER_DEFAULT);
  return denoised_image;
}


// crop image by kernel so it has the same size as image filtered by the naive approach
cv::Mat crop_image_by_kernel(cv::Mat image, cv::Size kernel_size)
{
  cv::Point top_left_point(floor(kernel_size.width / 2.0), floor(kernel_size.height / 2.0));
  cv::Size roi_size(image.size().width - 2 * floor(kernel_size.width / 2.0),
                    image.size().height - 2 * floor(kernel_size.height / 2.0));
  cv::Rect roi(top_left_point, roi_size);
  cv::Mat cropped_image = image(roi);
  return cropped_image;
}


// noise strength is between 0 (no noise) and 1 (maximum noise)
// returns float image with values between 0 and 1
cv::Mat add_noise_to_image(cv::Mat image, float noise_strength)
{
  cv::Mat noise(image.size(), CV_32F);
  cv::randn(noise, 0, noise_strength);
  cv::Mat noised_image;
  cv::normalize(image, noised_image, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
  noised_image = noised_image + noise;
  cv::normalize(noised_image, noised_image, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
  return noised_image;
}


void show_image(string window_name, cv::Mat image, bool normalize)
{
  //resize image to fit the screen
  cv::resize(image, image, cv::Size(600, 400), cv::INTER_AREA);

  // normalize image for good contrast
  if (normalize)
  {
    cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);
  }

  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::imshow(window_name, image);
}
