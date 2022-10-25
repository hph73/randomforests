cmake_minimum_required(VERSION 3.10)
project(RandomForest)

# set(THREADS_PREFER_PTHREAD_FLAG ON)
# find_package(Threads REQUIRED)
# set(LIBS Threads::Threads)

set(CMAKE_CXX_STANDARD 11)

file(GLOB SRC src/* src/tinyxml2/*)

message("SRC:" ${SRC})

add_executable(RandomForestTrain demo/rf_train_hao.cpp ${SRC})

add_executable(RandomForestTest demo/rf_test_hao.cpp ${SRC})

# target_link_libraries(RandomForest  ${LIBS})
