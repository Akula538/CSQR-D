#if defined(_WIN32) || defined(_WIN64)
  #include <io.h>
  #include <fcntl.h>
#endif

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
    std::vector<unsigned char> fileData;
    unsigned char* data = nullptr;
    int w = 0, h = 0, channels = 0;

    if (argc >= 2) {
        // Старая логика — загрузка из файла, если передали путь
        const char* path = argv[1];
        data = stbi_load(path, &w, &h, &channels, 0);
        if (!data) {
            std::cerr << "Failed to load image: " << path << "\n";
            return 2;
        }
    } else {
    #if defined(_WIN32) || defined(_WIN64)
        // на Windows поставим stdin в бинарный режим (чтобы CR/LF не портили байты)
        _setmode(_fileno(stdin), _O_BINARY);
#endif
        // читаем все байты из stdin в vector
        std::istreambuf_iterator<char> it(std::cin.rdbuf());
        std::istreambuf_iterator<char> end;
        fileData.assign(it, end);

        if (fileData.empty()) {
            std::cerr << "No input received on stdin\n";
            return 4;
        }

        // декодируем изображение из памяти
        data = stbi_load_from_memory(fileData.data(), static_cast<int>(fileData.size()), &w, &h, &channels, 0);
        if (!data) {
            std::cerr << "Failed to decode image from stdin (unsupported format or corrupted data)\n";
            return 5;
        }
    }

    if (channels != 1 && channels != 3 && channels != 4) {
        std::cerr << "Unsupported channels: " << channels << "\n";
        stbi_image_free(data);
        return 3;
    }

    auto gray = ConvertToGray(data, w, h, channels);

    ImageView imgView(
        gray.get(),
        w * h,
        w,
        h,
        ImageFormat::Lum,
        w,
        1
    );

    HybridBinarizer bin(imgView);
    auto matrixPtr = bin.getBlackMatrix();
    const BitMatrix& matrix = *matrixPtr;

    // Сохраняем бинаризованное изображение (можно оставить/удалить)
    SaveBitMatrixAsPNG(matrix, "zxing-cpp_finder/output.png");

    auto patterns = FindFinderPatterns(matrix, /*tryHarder=*/true);

    for (size_t i = 0; i < patterns.size(); ++i) {
        const auto& cp = patterns[i];
        auto quad = FindConcentricPatternCorners(matrix, cp, cp.size, 2);

        std::cout << cp.x << "," << cp.y << " " << cp.size << " ";

        if (!quad) {
            std::cout << '\n';
            continue;
        }

        for (int j = 0; j < 4; j++) {
            auto p = (*quad)[j];
            std::cout << p.x << "," << p.y << " ";
        }
        std::cout << '\n';
    }

    stbi_image_free(data);
    return 0;
}