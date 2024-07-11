cmake_minimum_required(VERSION 3.12)
project(REIN VERSION 1.0 LANGUAGES CXX)

find_package(LibVNCServer REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Torch REQUIRED)

set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH} /usr/local/libtorch")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
set(CMAKE_CXX_STANDARD 20)


add_executable(REIN
	src/main.cpp
	src/monitor.cpp
	src/keymap.cpp
	src/agent.cpp

	src/vae.cpp
	src/vae_encoder.cpp
	src/vae_decoder.cpp
	src/action_encoder.cpp

	src/image_holder.cpp
)

target_include_directories(REIN PUBLIC
	include
	${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

target_link_libraries(REIN LibVNCServer::vncclient)
target_link_libraries(REIN LibVNCServer::vncserver)
target_link_libraries(REIN ${OpenCV_LIBS})
target_link_libraries(REIN ${TORCH_LIBRARIES})
