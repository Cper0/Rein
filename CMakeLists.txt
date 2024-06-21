cmake_minimum_required(VERSION 3.12)
project(REIN)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 23)

add_executable(REIN
	src/main.cpp
	src/monitor.cpp
	src/keymap.cpp
	src/util.cpp
	src/mnist.cpp

	src/nn/layer_base.cpp
	src/nn/conv.cpp
	src/nn/affine.cpp
	src/nn/sigmoid.cpp
	src/nn/relu.cpp
	src/nn/softmaxwithloss.cpp
	src/nn/sme.cpp
	src/nn/optimizer.cpp
)

target_include_directories(REIN PUBLIC
	include
)



find_package(OpenCV REQUIRED)
find_package(LibVNCServer REQUIRED)
find_package(Armadillo REQUIRED)

target_link_libraries(REIN ${OpenCV_LIBS})
target_link_libraries(REIN LibVNCServer::vncclient)
target_link_libraries(REIN LibVNCServer::vncserver)
target_link_libraries(REIN ${ARMADILLO_LIBRARIES})
