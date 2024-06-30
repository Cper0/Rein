cmake_minimum_required(VERSION 3.12)
project(REIN VERSION 1.0 LANGUAGES CXX)

find_package(LibVNCServer REQUIRED)
find_package(Armadillo REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} /usr/local/libtorch")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 20)


add_executable(REIN
	src/main.cpp
	src/monitor.cpp
	src/keymap.cpp
	src/util.cpp
	src/mnist.cpp

	src/nn/layer_base.cpp
	src/nn/dropout.cpp
	src/nn/normalizer.cpp
	src/nn/flatten.cpp
	src/nn/affine.cpp
	src/nn/sigmoid.cpp
	src/nn/relu.cpp
	src/nn/conv.cpp
	src/nn/backconv.cpp
	src/nn/rectifier.cpp

	src/nn/simerror.cpp
	src/nn/softmaxwithloss.cpp
	src/nn/sme.cpp
	src/nn/optimizer.cpp
)

target_include_directories(REIN PUBLIC
	include
	include/nn
	${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(REIN LibVNCServer::vncclient)
target_link_libraries(REIN LibVNCServer::vncserver)
target_link_libraries(REIN ${ARMADILLO_LIBRARIES})
target_link_libraries(REIN ${OpenCV_LIBS})
target_link_libraries(REIN ${TORCH_LIBRARIES})
