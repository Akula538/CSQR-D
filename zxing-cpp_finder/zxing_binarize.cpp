#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#include <iostream>
#include <vector>
#include <memory>

#include "ImageView.h"
#include "HybridBinarizer.h"
#include "BitMatrix.h"
#include "ConcentricFinder.h"
#include "qrcode/QRDetector.h"

using namespace ZXing;
using namespace ZXing::QRCode;

// Convert RGB/RGBA to grayscale (Luminance)
std::unique_ptr<uint8_t[]> ConvertToGray(uint8_t* src, int w, int h, int channels)
{
    auto gray = std::make_unique<uint8_t[]>(w * h);

    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            uint8_t* p = src + (y * w + x) * channels;

            uint8_t r = p[0];
            uint8_t g = (channels >= 2 ? p[1] : p[0]);
            uint8_t b = (channels >= 3 ? p[2] : p[0]);

            // ZXing luminance formula
            uint8_t lum = static_cast<uint8_t>((306 * r + 601 * g + 117 * b + 0x200) >> 10);

            gray[y * w + x] = lum;
        }
    }

    return gray;
}

// Save BitMatrix as binary PNG
void SaveBitMatrixAsPNG(const BitMatrix& matrix, const char* filename)
{
    int width = matrix.width();
    int height = matrix.height();

    std::unique_ptr<uint8_t[]> image(new uint8_t[width * height]);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            image[y * width + x] = matrix.get(x, y) ? 0 : 255; // black=0, white=255
        }
    }

    if (!stbi_write_png(filename, width, height, 1, image.get(), width)) {
        std::cerr << "Failed to save binary image: " << filename << "\n";
    } else {
        // std::cout << "Saved binary image as " << filename << "\n";
    }
}

int main(int argc, char** argv)
{
    if (argc < 2) {
        std::cerr << "Usage: qr_finder.exe <image>\n";
        return 1;
    }

    const char* path = argv[1];

    int w, h, channels;
    uint8_t* data = stbi_load(path, &w, &h, &channels, 0);
    if (!data) {
        std::cerr << "Failed to load image: " << path << "\n";
        return 2;
    }

    if (channels != 1 && channels != 3 && channels != 4) {
        std::cerr << "Unsupported channels: " << channels << "\n";
        stbi_image_free(data);
        return 3;
    }

    auto gray = ConvertToGray(data, w, h, channels);

    ImageView imgView(
        gray.get(),
        w * h,              // buffer size
        w,
        h,
        ImageFormat::Lum,   // grayscale
        w,                  // rowStride
        1                    // pixStride
    );

    HybridBinarizer bin(imgView);
    auto matrixPtr = bin.getBlackMatrix();
    const BitMatrix& matrix = *matrixPtr;

    // Сохраняем бинаризованное изображение
    SaveBitMatrixAsPNG(matrix, "zxing-cpp_finder/output.png");
}
