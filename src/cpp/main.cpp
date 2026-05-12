#include "QRscanner.hpp"

#include <opencv2/imgcodecs.hpp>

#include <iostream>
#include <string>
#include <filesystem>

int main(int argc, char** argv)
{
    std::string image_path;
    std::string dir_path;
    qrscanner::ScanOptions options;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--seed" && i + 1 < argc) {
            options.random_seed = static_cast<unsigned int>(std::stoul(argv[++i]));
        } else if (arg == "--debug-output" && i + 1 < argc) {
            options.debug_output_dir = argv[++i];
        } else if (arg == "--directory" && i + 1 < argc) {
            dir_path = argv[++i];
        } else if (image_path.empty()) {
            image_path = arg;
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return 1;
        }
    }

    if (!dir_path.empty()) {
        namespace fs = std::filesystem;
        if (!fs::exists(dir_path) || !fs::is_directory(dir_path)) {
            std::cerr << "Invalid directory: " << dir_path << "\n";
            return 3;
        }
        // while (true) {
        for (const auto& entry : fs::recursive_directory_iterator(dir_path)) {
            if (entry.is_regular_file() && entry.path().extension() == ".png") {
                std::string current_image_path = entry.path().string();
                cv::Mat img = cv::imread(current_image_path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
                if (img.empty()) {
                    std::cerr << "Failed to read image: " << current_image_path << "\n";
                    continue;
                }
                auto result = qrscanner::detectAndDecode(img, options);
                std::cout << current_image_path << ": ";
                if (result.status == qrscanner::ScanStatus::Decoded)
                    std::cout << result.text << "\n";
                else if (result.status == qrscanner::ScanStatus::Ctf)
                    std::cout << "CTF\n";
                else
                    std::cout << "None\n";
            }
        }
        // }
    } else if (!image_path.empty()) {
        cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
        if (img.empty()) {
            std::cerr << "Failed to read image: " << image_path << "\n";
            return 2;
        }
        auto result = qrscanner::detectAndDecode(img, options);
        if (result.status == qrscanner::ScanStatus::Decoded)
            std::cout << result.text << "\n";
        else if (result.status == qrscanner::ScanStatus::Ctf)
            std::cout << "CTF\n";
        else
            std::cout << "None\n";
    } else {
        std::cerr << "Usage: qrscanner_cli <image> [--directory DIR] [--seed N] [--debug-output DIR]\n";
        return 1;
    }

    return 0;
}
