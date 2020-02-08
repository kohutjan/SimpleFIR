
#include <getopt.h>

#include "simple_fir.hpp"

using namespace std;

int main(int argc, char **argv)
{
  static struct option long_options[] = {
  {"input_image", required_argument, 0, 'i'},
  {"noise_strength", required_argument, 0, 'n'},
  {"kernel_size", required_argument, 0, 'k'},
  {0, 0, 0}};

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

  cv::Mat noised_image = add_noise_to_image(image, noise_strength);

  cv::Mat opencv_denoised_image = opencv_denoise_image(noised_image, kernel_size);
  opencv_denoised_image = crop_opencv_denoised_image(opencv_denoised_image, kernel_size);

  show_image("Input Image", image);
  show_image("Noised Image", noised_image);
  show_image("OpenCV Denoised Image", opencv_denoised_image);

  cv::waitKey(0);

  return 0;
}


// use OpenCV to denoise image
cv::Mat opencv_denoise_image(cv::Mat image, cv::Size kernel_size)
{
  cv::Mat denoised_image;
  GaussianBlur(image, denoised_image, kernel_size, 0, 0, cv::BORDER_DEFAULT);
  return denoised_image;
}


// crop OpenCV denoised image so it has the same size as image filtered by naive approach
cv::Mat crop_opencv_denoised_image(cv::Mat image, cv::Size kernel_size)
{
  int half_kernel_size = floor(kernel_size.height / 2.0);
  cv::Point top_left_point(half_kernel_size, half_kernel_size);
  cv::Size roi_size(image.size().width - 2 * half_kernel_size,
                    image.size().height - 2 * half_kernel_size);
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


void show_image(string window_name, cv::Mat image)
{
  //resize image to fit the screen
  cv::resize(image, image, cv::Size(600, 400), cv::INTER_AREA);

  // normalize image for good contrast
  cv::normalize(image, image, 0.0, 1.0, cv::NORM_MINMAX, CV_32F);

  cv::namedWindow(window_name, cv::WINDOW_AUTOSIZE);
  cv::imshow(window_name, image);
}
