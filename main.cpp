#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>  
#include <iostream>
#include <vector>

const std::string windowName = "window1";

const float EPS = 1e-4;

// I/O
// 得到的都是浮点表示的图

cv::Mat getInput(const std::string& picPath);
void outputImage(const cv::Mat& mat, const std::string& picPath);

// Processing

std::vector<cv::Mat> processImage(const cv::Mat& mat);

// Transform functions

inline cv::Vec3f bgrToYmc(const cv::Vec3f& vec)
{
	return cv::Vec3f(1.0f - vec[0], 1.0f - vec[1], 1.0f - vec[2]);
}
inline cv::Vec3f ymcToBgr(const cv::Vec3f& vec)
{
	return cv::Vec3f(1.0f - vec[0], 1.0f - vec[1], 1.0f - vec[2]);
}
inline cv::Vec3f bgrToHsi(const cv::Vec3f& vec)
{
	float R = vec[2], G = vec[1], B = vec[0];
	float RG = (R - G), RB = (R - B);
	float tmp = sqrt(RG * RG + RB * (G - B)); // 分母不能为0！！！
	float theta = tmp != 0 ? acos(0.5f * (RG + RB) / tmp) : 0;
	theta = B > G + EPS ? CV_2PI - theta : theta;
	return cv::Vec3f(
		theta / CV_2PI,
		1.0f - 3.0f * std::min(std::min(R, G), B) / (R + G + B),
		(R + G + B) / 3.0f
	);
}
inline cv::Vec3f hsiToBgr(const cv::Vec3f& vec)
{
	float H = vec[0] * CV_2PI, S = vec[1], I = vec[2];
	float H2 = fmod(H, CV_2PI / 3);
	float v1 = I * (1.0f - S);
	float v2 = I * (1.0f + S * cos(H2) / cos(CV_PI / 3 - H2));
	float v3 = 3 * I - v1 - v2;
	if (H + EPS < CV_2PI / 3.0f)
	{
		return cv::Vec3f(v1, v3, v2);
	}
	else if (H + EPS < 2.0f * CV_2PI / 3.0f)
	{
		return cv::Vec3f(v3, v2, v1);
	}
	else
	{
		return cv::Vec3f(v2, v1, v3);
	}
}

int main()
{
	cv::Mat mat = getInput("0.jpg");

	auto matArr = processImage(mat);

	outputImage(matArr[0], "1RGB.jpg");
	outputImage(matArr[1], "2CMY.jpg");
	outputImage(matArr[2], "3HSI.jpg");

	cv::imshow("w1", matArr[0]);
	cv::imshow("w2", matArr[1]);
	cv::imshow("w3", matArr[2]);

	cv::waitKey();

	return 0;
}

cv::Mat getInput(const std::string& picPath)
{
	cv::Mat mat;
	cv::imread(picPath).convertTo(mat, CV_32FC3, 1 / 255.0);
	return mat;
}
void outputImage(const cv::Mat& mat, const std::string& picPath)
{
	cv::Mat tmpMat;
	mat.convertTo(tmpMat, CV_8UC3, 255.0f);
	imwrite(picPath, tmpMat);
}

// 用一个二次函数来表示颜色分量变化函数
// 假设二次函数过(0, 0), (1, 1)
// scale的值是x = 0.5时f(x)的值
float colorComponentFunction(float x, float scale = 0.5f)
{
	float a = 2 - 4 * scale;
	return a * x * x + (1 - a) * x;
}

// 对色调的转换要用一个特殊的函数
float hFunction(float x)
{
	if (x <= 1.0f / 2)
	{
		return x * 0.55;
	}
	else
	{
		return x * 1.2;
	}
}

std::vector<cv::Mat> processImage(const cv::Mat& mat)
{
	cv::Mat matRGB, matCMY, matHSI;
	mat.copyTo(matRGB);
	mat.copyTo(matCMY);
	mat.copyTo(matHSI);

	for (int i = 0; i < mat.rows; i++)
	{
		for (int j = 0; j < mat.cols; j++)
		{
			// tranformation
			matCMY.at<cv::Vec3f>(i, j) = bgrToYmc(matCMY.at<cv::Vec3f>(i, j));
			matHSI.at<cv::Vec3f>(i, j) = bgrToHsi(matHSI.at<cv::Vec3f>(i, j));

			// tonal function

			matRGB.at<cv::Vec3f>(i, j)[2] = colorComponentFunction(matRGB.at<cv::Vec3f>(i, j)[2], 0.8);
			matCMY.at<cv::Vec3f>(i, j)[2] = colorComponentFunction(matCMY.at<cv::Vec3f>(i, j)[2], 0.2);

			matHSI.at<cv::Vec3f>(i, j)[2] = colorComponentFunction(matHSI.at<cv::Vec3f>(i, j)[2], 0.60); // I
			matHSI.at<cv::Vec3f>(i, j)[1] = colorComponentFunction(matHSI.at<cv::Vec3f>(i, j)[1], 0.35); // S
			matHSI.at<cv::Vec3f>(i, j)[0] = hFunction(matHSI.at<cv::Vec3f>(i, j)[0]); // H

			// inverse transformation
			matCMY.at<cv::Vec3f>(i, j) = ymcToBgr(matCMY.at<cv::Vec3f>(i, j));
			matHSI.at<cv::Vec3f>(i, j) = hsiToBgr(matHSI.at<cv::Vec3f>(i, j));
		}
	}

	std::vector<cv::Mat> res;
	res.push_back(matRGB);
	res.push_back(matCMY);
	res.push_back(matHSI);

	return res;
}