cmake_minimum_required(VERSION 3.26)

project(triangle)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Vulkan REQUIRED)

add_compile_definitions(VULKAN_HPP_NO_TO_STRING)

add_subdirectory(deps/glfw)

set(GLM_TEST_ENABLE OFF CACHE BOOL "GLM Build unit tests")
add_subdirectory(deps/glm EXCLUDE_FROM_ALL)

add_compile_definitions(GLM_FORCE_RADIANS)
add_compile_definitions(GLM_FORCE_DEPTH_ZERO_TO_ONE)
add_compile_definitions(GLM_ENABLE_EXPERIMENTAL)

add_executable(${PROJECT_NAME} 
	main.cpp
	src/ModelViewer.cpp
	src/stb_Impl.cpp
	src/tiny_obj_loader_Impl.cpp)

target_include_directories(${PROJECT_NAME} PRIVATE src)

target_include_directories(${PROJECT_NAME} PRIVATE deps/stb)
target_include_directories(${PROJECT_NAME} PRIVATE deps/tinyobjloader)

target_link_libraries(${PROJECT_NAME} glfw ${GLFW_LIBRARIES} Vulkan::Vulkan)

# Добавление команды для трансляции шейдеров
add_custom_command(
    OUTPUT ${CMAKE_SOURCE_DIR}/shaders/vert.spv
           ${CMAKE_SOURCE_DIR}/shaders/frag.spv
    COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.vert -o ${CMAKE_SOURCE_DIR}/shaders/vert.spv
    COMMAND glslc ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.frag -o ${CMAKE_SOURCE_DIR}/shaders/frag.spv
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.vert
            ${CMAKE_CURRENT_SOURCE_DIR}/shaders/shader.frag
    COMMENT "Compiling shaders"
)

add_custom_target(
    ShadersTarget ALL
    DEPENDS ${CMAKE_SOURCE_DIR}/shaders/vert.spv
            ${CMAKE_SOURCE_DIR}/shaders/frag.spv
)

add_dependencies(${PROJECT_NAME} ShadersTarget)