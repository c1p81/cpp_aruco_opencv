#include <iostream>
#include <string>
#include <ctime>
#include <cmath>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <opencv2/aruco.hpp>
#include <opencv2/core/version.hpp>
#include <opencv2/objdetect/aruco_detector.hpp>
#include <opencv2/core/utils/logger.hpp>

#include <filesystem>

namespace fs = std::filesystem;
float x_ref = 0.0f;
float y_ref = 0.0f;
float z_ref = 0.0f;




std::vector<float> movingAverage(const std::vector<float>& data, int windowSize) {
    std::vector<float> result;
    int n = data.size();

    if (windowSize <= 0 || windowSize > n) {
        std::cerr << "Dimensione finestra non valida." << std::endl;
        return result;
    }

    float sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum += data[i];
        if (i >= windowSize)
            sum -= data[i - windowSize];
        if (i >= windowSize - 1)
            result.push_back(sum / windowSize);
    }

    return result;
}



bool isRotationMatrix(cv::Mat &R) {
    cv::Mat Rt;
    cv::transpose(R, Rt);
    cv::Mat shouldBeIdentity = Rt * R;
    cv::Mat I = cv::Mat::eye(3, 3, shouldBeIdentity.type());
    return cv::norm(I, shouldBeIdentity) < 1e-6;
}

float distanza_da_reference(float x, float y, float z){
 return std::sqrt(std::pow(x_ref - x, 2) +
                     std::pow(y_ref - y, 2) +
                     std::pow(z_ref - z, 2));
}

int converti_tempo(std::string timestamp)
{
    std::tm tm = {};
    tm.tm_year = std::stoi(timestamp.substr(0, 4)) - 1900; // years since 1900
    tm.tm_mon  = std::stoi(timestamp.substr(4, 2)) - 1;     // months since January
    tm.tm_mday = std::stoi(timestamp.substr(6, 2));
    tm.tm_hour = std::stoi(timestamp.substr(8, 2));
    tm.tm_min  = std::stoi(timestamp.substr(10, 2));
    tm.tm_sec  = std::stoi(timestamp.substr(12, 2));
    return timegm(&tm);
}

cv::Vec3d rotationMatrixToEulerAngles(cv::Mat &R) {
    assert(isRotationMatrix(R));
    double sy = std::sqrt(R.at<double>(0, 0) * R.at<double>(0, 0) + 
                          R.at<double>(1, 0) * R.at<double>(1, 0));
    bool singular = sy < 1e-6;
    double x, y, z;
    if (!singular) {
        x = std::atan2(R.at<double>(2, 1), R.at<double>(2, 2));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = std::atan2(R.at<double>(1, 0), R.at<double>(0, 0));
    } else {
        x = std::atan2(-R.at<double>(1, 2), R.at<double>(1, 1));
        y = std::atan2(-R.at<double>(2, 0), sy);
        z = 0;
    }
    return cv::Vec3d(x, y, z);
}



int main() {
    //std::cout << "OpenCV version: " << CV_VERSION << std::endl;
    cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
    freopen("/dev/null", "w", stderr);
    float arucoDim = 0.25f; // meters

    std::vector<cv::Point3f> objPoints = {
    {-arucoDim/2.f,  arucoDim/2.f, 0},
    { arucoDim/2.f,  arucoDim/2.f, 0},
    { arucoDim/2.f, -arucoDim/2.f, 0},
    {-arucoDim/2.f, -arucoDim/2.f, 0}
};

    std::ofstream tag15("tag15.csv");
    std::ofstream tag20("tag20.csv");
    std::ofstream tag25("tag25.csv");
    std::ofstream tag40("tag40.csv");

    cv::Mat K = cv::Mat::eye(3, 3, CV_64F); 
    cv::Mat D = cv::Mat::zeros(5, 1, CV_64F); 

    K = (cv::Mat_<double>(3,3) <<
                    2694.30873, 0.0,   1793.8734,
                    0.0,    2704.6156, 1105.87691,
                    0,      0,     1);

    D.at<double>(0, 0) = -0.41168919;  // k1
    D.at<double>(1, 0) = 0.26635946;  // k2
    D.at<double>(2, 0) = -0.0017025;   // p1
    D.at<double>(3, 0) = 0.00744752;   // p2
    D.at<double>(4, 0) = -0.16179997;   // k3                

    std::string folderPath = "../Tags";

    if (!fs::exists(folderPath) || !fs::is_directory(folderPath)) {
        std::cerr << "Directory does not exist: " << folderPath << std::endl;
        return 1;
    }

    for (const auto& entry : fs::directory_iterator(folderPath)) {
        // dalle dimensione del file evita le foto notturne
        if (fs::is_regular_file(entry) && (fs::file_size(entry) > 2000000)) {
            std::string test = entry.path().filename().string();
            std::string result = test.substr(9, test.length() - 9 - 16);
            std::time_t utime = converti_tempo(result);
            
            cv::Mat img = cv::imread(entry.path(), cv::IMREAD_GRAYSCALE);
            if (img.empty()) {
                    std::cerr << "Could not read the image\n";
                    return 1;
                }

            cv::Ptr<cv::CLAHE> clahe = cv::createCLAHE();
            clahe->apply(img, img);
    
            std::vector<int> markerIds;
            std::vector<std::vector<cv::Point2f>> markerCorners, rejectedCandidates;
            cv::aruco::DetectorParameters detectorParams = cv::aruco::DetectorParameters();
            cv::aruco::Dictionary dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
            cv::aruco::ArucoDetector detector(dictionary, detectorParams);
            detector.detectMarkers(img, markerCorners, markerIds, rejectedCandidates);   
            
            if (!markerIds.empty()) {
                std::vector<cv::Vec3d> rvecs, tvecs;
                //subpixel refine
                for (auto& corners : markerCorners) {
                    cv::cornerSubPix(img, corners, cv::Size(3, 3), cv::Size(-1, -1),
                        cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::MAX_ITER, 30, 0.1)
                    );
                }

                rvecs.clear();
                tvecs.clear();

                for (size_t i = 0; i < markerCorners.size(); ++i) {
                    cv::Vec3d rvec, tvec;
                    cv::solvePnP(objPoints, markerCorners[i], K, D, rvec, tvec);
                    rvecs.push_back(rvec);
                    tvecs.push_back(tvec);
                }

                //cv::aruco::estimatePoseSingleMarkers(markerCorners, arucoDim, K, D, rvecs, tvecs);
                for (size_t i = 0; i < markerIds.size(); ++i) {
                    //std::cout << markerIds[i] << ";"<< tvecs[i][0] << ";" << tvecs[i][1] << ";" << tvecs[i][2] << "\n";
                    if (markerIds[i] == 11){
                        x_ref = tvecs[i][0];
                        y_ref = tvecs[i][1];
                        z_ref = tvecs[i][2];
                    }
                }

                for (size_t i = 0; i < markerIds.size(); ++i) {
                    cv::Mat R;
                    cv::Rodrigues(rvecs[i], R);
                    R = R.t(); // transpose
                    cv::Mat R_flip = (cv::Mat_<double>(3, 3) <<
                                    1,  0,  0,
                                    0, -1,  0,
                                    0,  0, -1);
                    R = R_flip * R;

                    cv::Vec3d angles = rotationMatrixToEulerAngles(R);
                    //double dist = cv::norm(tvecs[i]);
                    /*std::cout << markerIds[i] << ";"
                    << tvecs[i][0] << ";" << tvecs[i][1] << ";" << tvecs[i][2] << ";"
                    << utime << ";"
                    << result << ";"
                    << angles[0] * 180.0 / CV_PI << ";" 
                    << angles[1] * 180.0 / CV_PI << ";" 
                    << angles[2] * 180.0 / CV_PI << ";"
                    << dist << "\n";*/
                    double dist = distanza_da_reference( tvecs[i][0], tvecs[i][1], tvecs[i][2]);
                    //std::cout << markerIds[i] << ";" << utime << ";" << dist << "\n"; 
                    switch (markerIds[i])
                    // lo spazio nel csv e' il delimitatore di default per gnuplot
                    //gnuplot -e "plot 'tag25.csv' using 1;2 with points; pause -1"
                    {
                    case 15:
                        if ((dist < 2.5) && (dist > 2.3)){
                            tag15 << utime << " " << dist << "\n";
                            }                           

                        break;
                    case 20:
                        if ((dist < 5.5) && (dist > 5.1)){
                            tag20 << utime << " " << dist << "\n";
                            }                           
                        break;
                    case 25:
                        if ((dist < 3.05) && (dist > 2.90)){
                            tag25 << utime << " " << dist << "\n";
                            }                           
                        break;
                    case 40:
                    if ((dist < 3.45) && (dist > 3.2)){
                            tag40 << utime << " " << dist << "\n";
                            }   
                        break;
                    default:
                        break;
                    }
                }
            }
        }
    }
    tag15.close();
    tag20.close();
    tag25.close();
    tag40.close();
    return 0;
}