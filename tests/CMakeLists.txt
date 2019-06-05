include_directories(${CMAKE_SOURCE_DIR}/src)

add_executable(test_link_layer test_link_layer.cpp)
target_link_libraries(test_link_layer neuralfield)
add_test(link_layer test_link_layer)
