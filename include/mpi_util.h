#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#include <gif.h>
#include <mpi.h>

extern int mpi_rank;
extern int mpi_size;
extern int mpi_thread_level_provided;

void mpi_util_init(animated_gif *image);
void dungeon_master_to_masters(animated_gif *image);
void masters_to_slaves(animated_gif *image);
void slaves_to_masters(animated_gif *image);
void masters_to_dungeon_master(animated_gif *image);
void mpi_apply_gray_filter( animated_gif * image );
void mpi_apply_gray_line( animated_gif * image );
void mpi_apply_sobel_filter( animated_gif * image );
void mpi_apply_blur_filter( animated_gif * image, int size, int threshold );

#endif // MPI_UTIL_H
