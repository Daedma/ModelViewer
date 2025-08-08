list(APPEND TESTS_LIST 
  "VulkanContext"
)

include(GoogleTest)

enable_testing()

foreach(test_name IN LISTS TESTS_LIST)
  set(test_target "test_${test_name}")
  set(test_source "${CMAKE_CURRENT_SOURCE_DIR}/test_${test_name}.cpp")

  add_executable(${test_target} ${test_source})

  target_include_directories(${test_target} PRIVATE
    ${PROJECT_SOURCE_DIR}/src
  )

  target_link_libraries(${test_target} PRIVATE 
    ${PROJECT_NAME}::lib 
    GTest::gtest_main
  )

  gtest_discover_tests(${test_target})
endforeach()
