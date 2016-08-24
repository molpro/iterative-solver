project(IterativeSolver)
cmake_minimum_required(VERSION 2.8)
add_subdirectory(Eigen)
aux_source_directory(. SRC_LIST)
add_executable(${PROJECT_NAME}-test ${SRC_LIST})
add_library (${PROJECT_NAME} ${SRC_LIST})
include_directories(Eigen )

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/Modules/")
find_package(MKL)
if (MKL_FOUND)
include_directories(${MKL_INCLUDE_DIRS})
message("MKL link libraries are ${MKL_LIBRARIES}")
target_link_libraries(${PROJECT_NAME}-test ${MKL_LIBRARIES})
add_definitions(-DEIGEN_USE_MKL_ALL)
else (MKL_FOUND)
find_package(BLAS REQUIRED)
if (BLAS_FOUND)
include_directories(${BLAS_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}-test ${BLAS_LIBRARY} -lm -ldl)
message("Blas link library is ${BLAS_LIBRARIES}")
else (BLAS_FOUND)
target_link_libraries(${PROJECT_NAME}-test -lm -ldl)
endif (BLAS_FOUND)
endif (MKL_FOUND)


find_package(Doxygen)
if(DOXYGEN_FOUND)
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile @ONLY)
add_custom_target(${PROJECT_NAME}-doc ALL
${DOXYGEN_EXECUTABLE} ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile
WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
COMMENT "Generating API documentation with Doxygen" VERBATIM
)
endif(DOXYGEN_FOUND)