cmake_minimum_required(VERSION 2.8)
project(image-filtering)

set(ROOT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR})
set(SRC_DIRECTORY ${ROOT_DIRECTORY}/src)
set(INCLUDE_DIRECTORY ${ROOT_DIRECTORY}/include)
include_directories(${INCLUDE_DIRECTORY})
add_executable(filtering
        ${SRC_DIRECTORY}/main.c
        ${SRC_DIRECTORY}/gif.c
        ${SRC_DIRECTORY}/main.c
        ${SRC_DIRECTORY}/mpi_util.c
        ${SRC_DIRECTORY}/dgif_lib.c
        ${SRC_DIRECTORY}/egif_lib.c
        ${SRC_DIRECTORY}/gifalloc.c
        ${SRC_DIRECTORY}/gif_err.c
        ${SRC_DIRECTORY}/gif_font.c
        ${SRC_DIRECTORY}/gif_hash.c
        ${SRC_DIRECTORY}/openbsd-reallocarray.c
        ${SRC_DIRECTORY}/quantize.c
        )

# OpenMP support
find_package(OpenMP)
set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS} -O3")

# MPI support
find_package(MPI)
include_directories(${MPI_INCLUDE_PATH})
target_link_libraries(filtering ${MPI_C_LIBRARIES})
target_link_libraries(filtering m)
