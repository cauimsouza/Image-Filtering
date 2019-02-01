#ifndef MPI_UTIL_H
#define MPI_UTIL_H

#include <gif.h>
#include <mpi.h>

extern int rank;
extern int size;

void bcast_image(animated_gif *image);
void gather_image(animated_gif *image);
void mpi_apply_gray_filter( animated_gif * image );
void mpi_apply_gray_line( animated_gif * image );
void mpi_apply_sobel_filter( animated_gif * image );
void mpi_apply_blur_filter( animated_gif * image, int size, int threshold );

#endif // MPI_UTIL_H
