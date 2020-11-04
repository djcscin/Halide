@echo off

if not exist bin md bin
if not exist images_output md images_output

cd bin && ^
cmake -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=%VCPKG_CMAKE_FILE% -DHalide_DIR=%HALIDE_ROOT%\lib\cmake\Halide .. && ^
cmake --build . && cd .. && ^
bin\gamma1 ..\images\final2016.jpg 0.5 2.2 ..\images\final2017.jpg 0.5 2.2 images_output\final2020.jpg && ^
bin\gamma1 ..\images\final2016.jpg 0.4 0.9 ..\images\final2017.jpg 0.6 0.9 images_output\final2021.jpg && ^
bin\gamma1 ..\images\final2016.jpg 0.5 1.3 ..\images\final2017.jpg 0.5 1.3 images_output\final2022.jpg && ^
bin\gamma0 ..\images\final2016.jpg 1.1 2.2 images_output\final2023.jpg && ^
bin\gamma0 ..\images\final2016.jpg 1.1 0.9 images_output\final2024.jpg && ^
bin\gamma0 ..\images\final2016.jpg 1.1 1.3 images_output\final2025.jpg
