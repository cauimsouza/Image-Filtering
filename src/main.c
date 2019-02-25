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
int mpi_size, mpi_rank;

int main( int argc, char ** argv )
{
	MPI_Init(&argc, &argv);
	MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
	MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);
	
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
    if (mpi_rank == 0) {
        image = load_pixels( input_filename ) ;
        if ( image == NULL ) { return 1 ; }

        /* IMPORT Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        printf( "GIF loaded from file %s with %d image(s) in %lf s\n", 
          input_filename, image->n_images, duration ) ;

        /* FILTER Timer start */
        gettimeofday(&t1, NULL);
    } else {
        image = (animated_gif*) malloc(sizeof(animated_gif));
        image->p = NULL;
        image->width = NULL;
        image->height = NULL;
        image->g = NULL;
    }

	mpi_util_init();

	/***********************************************************/
	dungeon_master_to_masters(image);
	masters_to_slaves(image);
	
	/* /\* Convert the pixels into grayscale *\/		   */
	double tik, tak;
	if (mpi_rank == 0) tik = MPI_Wtime();
	mpi_apply_gray_filter( image ) ;
	if (mpi_rank == 0) tak = MPI_Wtime();
	if (mpi_rank == 0) printf("Gray filter: %f s\n", tak - tik);
	/* slaves_to_masters(image); */
	
	/* Apply blur filter with convergence value */
	if (mpi_rank == 0) tik = MPI_Wtime();
	mpi_apply_blur_filter(image, 5, 20) ;
	if (mpi_rank == 0) tak = MPI_Wtime();
	if (mpi_rank == 0) printf("Blur filter: %f s\n", tak - tik);
	
	masters_to_slaves(image);
	
	/* Apply sobel filter on pixels */
	if (mpi_rank == 0) tik = MPI_Wtime();
	mpi_apply_sobel_filter( image ) ;
	if (mpi_rank == 0) tak = MPI_Wtime();
	if (mpi_rank == 0) printf("Sobel filter: %f s\n", tak - tik);
	
    slaves_to_masters(image);
	masters_to_dungeon_master(image);
	
    if (mpi_rank == 0) {
        /* FILTER Timer stop */
        gettimeofday(&t2, NULL);

        duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);

        /* printf( "SOBEL done in %lf s\n", duration ) ; */

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
