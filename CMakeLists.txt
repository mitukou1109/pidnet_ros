cmake_minimum_required(VERSION 3.8)
project(pidnet_ros)

find_package(ament_cmake_auto REQUIRED)

find_program(UV_EXECUTABLE uv REQUIRED)

ament_auto_find_build_dependencies()

set(VENV_DIR ${CMAKE_CURRENT_BINARY_DIR}/.venv)

add_custom_target(${PROJECT_NAME}_venv ALL
  COMMAND ${CMAKE_COMMAND}
          -DUV_EXECUTABLE=${UV_EXECUTABLE}
          -DVENV_DIR=${VENV_DIR}
          -P ${CMAKE_CURRENT_SOURCE_DIR}/cmake/create_venv.cmake
  COMMAND ${CMAKE_COMMAND} -E env
          UV_PROJECT_ENVIRONMENT=${VENV_DIR}
          ${UV_EXECUTABLE} sync
  WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

install(
  DIRECTORY ${VENV_DIR}/
  DESTINATION share/${PROJECT_NAME}/.venv
  USE_SOURCE_PERMISSIONS
)

install(
  PROGRAMS ${VENV_DIR}/bin/segmentation_node
  DESTINATION lib/${PROJECT_NAME}
)

ament_auto_package()