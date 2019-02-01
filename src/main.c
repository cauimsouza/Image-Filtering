/*
 * INF560
 *
 * Image Filtering Project
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include <mpi.h>

#include <mpi_util.h>
#include <gif.h>

#define SOBELF_DEBUG 0

/* MPI global variables */
int size, rank;

int main( int argc, char ** argv )
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);
	
    char * input_filename ; 
    char * output_filename ;
    animated_gif * image ;
    struct timeval t1, t2;
    double duration ;

    if ( argc < 3 )
    {
        fprintf( stderr, "Usage: %s input.gif output.gif \n", argv[0] ) ;
        return 1 ;
    }

    input_filename = argv[1] ;
    output_filename = argv[2] ;

    /* IMPORT Timer start */
    gettimeofday(&t1, NULL);

    /* Load file and store the pixels in array */
    if (rank == 0) {
        image = load_pixels( input_filename ) ;
        if ( image == NULL ) { return 1 ; }

        /* IMPORT Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
          input_filename, image->n_images, duration ) ;

        /* FILTER Timer start */
        gettimeofday(&t1, NULL);

        /* Convert the pixels into grayscale */
        /* apply_gray_filter( image ) ; */

        /* Apply blur filter with convergence value */
        /* apply_blur_filter( image, 5, 20 ) ; */
    } else {
        image = (animated_gif*) malloc(sizeof(animated_gif));
    }

    bcast_image(image);

    /* Apply sobel filter on pixels */
    mpi_apply_sobel_filter( image ) ;

    gather_image(image);

    if (rank == 0) {
        /* FILTER Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "SOBEL done in %lf s\n", duration ) ;

        /* EXPORT Timer start */
        gettimeofday(&t1, NULL);

        /* Store file from array of pixels to GIF file */
        if ( !store_pixels( output_filename, image ) ) { return 1 ; }

        /* EXPORT Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "Export done in %lf s in file %s\n", duration, output_filename ) ;
    }

    MPI_Finalize();

    return 0 ;
}
