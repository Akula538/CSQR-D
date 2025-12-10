cmake_minimum_required(VERSION 3.10)
project(test)

include_directories(zxing-cpp/core/src)

add_library(finder_candidates finder_candidates.cpp)
g++ finder_candidates.cpp -I zxing-cpp/core/src -I zxing-cpp/core/src/qrcode -std=c++17 -c


g++ main.cpp finder_candidates.cpp zxing-cpp/core/src/BitMatrix.cpp zxing-cpp/core/src/qrcode/QRDetector.cpp zxing-cpp/core/src/ConcentricFinder.cpp -I zxing-cpp/core/src -I zxing-cpp/core/src/qrcode -std=c++17 -o qr_finder


g++ main.cpp finder_candidates.cpp zxing-cpp/core/src/*.cpp zxing-cpp/core/src/qrcode/*.cpp -I zxing-cpp/core/src -I zxing-cpp/core/src/qrcode -I zxing-cpp/core -std=c++17 -o qr_finder.exe 


./qr_finder.exe "C:\Users\user\Documents\Projects\QRscanner\data\v2\H\big\img0001_v2_H_big_d2.png"


g++ -std=c++17 main.cpp -I zxing-cpp/core/src -I zxing-cpp/core/src/zxing -I zxing-cpp/core/src/qrcode zxing-cpp/build/libZXing.a zxing-cpp/build/core/libZXingCore.a -lpng -lz -ljpeg -o finder

g++ main.cpp -std=c++17 -o qr_finder.exe -I zxing-cpp/core/src -I zxing-cpp/core/src/qrcode zxing-cpp/build/core/libZXing.a -lpng -lz -ljpeg


c++ -O3 -Wall -shared -std=c++17 -fPIC zxing_binarize -Ipybind11/include -Izxing-cpp/core/src -Izxing-cpp/core/src/qrcode -o pybinarize`python3-config --extension-suffix`

c++ -O3 -Wall -shared -std=c++17 -fPIC zxing_binarize.cpp -I"C:\Users\user\AppData\Local\Packages\PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0\LocalCache\local-packages\Python312\site-packages\pybind11\include" -Izxing-cpp/core/src -Izxing-cpp/core/src/qrcode -IC:/Python310/include -LC:/Python310/libs -lpython310 -o pybinarize.pyd


c++ -O3 -Wall -shared -std=c++17 -fPIC zxing_binarize.cpp -I"C:/Users/user/AppData/Local/Packages/PythonSoftwareFoundation.Python.3.12_qbz5n2kfra8p0/LocalCache/local-packages/Python312/site-packages/pybind11/include" -I"C:/Users/user/AppData/Local/Programs/Python/Python312/include" -Izxing-cpp/core/src -Izxing-cpp/core/src/qrcode -L"C:/Users/user/AppData/Local/Programs/Python/Python312/libs" -I. -I zxing-cpp/core/src -I zxing-cpp/core/src/qrcode zxing-cpp/build/core/libZXing.a -lpython312 -o pybinarize.pyd
