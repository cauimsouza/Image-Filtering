#include <stdlib.h>
#include <math.h>
#include <mpi.h>

#include <mpi_util.h>

#define CONV(l,c,nb_c) \
    (l)*(nb_c)+(c)

static MPI_Datatype dt_pixel;
static int dt_pixel_created = 0;

static void create_dt_pixel()
{
	if (dt_pixel_created) return;
	dt_pixel_created = 1;

	MPI_Type_contiguous(3, MPI_INT, &dt_pixel);
	MPI_Type_commit(&dt_pixel);
}

void bcast_image(animated_gif *image)
{
	create_dt_pixel();

	MPI_Bcast(&image->n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

	if (mpi_rank != 0) {
		image->width = (int*) malloc(image->n_images * sizeof(int));
		image->height = (int*) malloc(image->n_images * sizeof(int));
	}
	MPI_Bcast(image->width, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
	MPI_Bcast(image->height, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);

    int i;
	if (mpi_rank != 0) {
		image->p = (pixel**) malloc(image->n_images * sizeof(pixel*));
		for (i = 0; i < image->n_images; i++)
			image->p[i] = (pixel*) malloc(image->width[i] * image->height[i] * sizeof(pixel));
	}

    for (i = 0; i < image->n_images; i++)
        MPI_Bcast(image->p[i], image->width[i] * image->height[i], dt_pixel, 0, MPI_COMM_WORLD);
}

void gather_image(animated_gif *image)
{
	create_dt_pixel();

    int i, j;
    if (mpi_rank == 0) {
        MPI_Status status;
        for (i = 1; i < mpi_size; i++) {
            for (j = 0; j < image->n_images; j++) {
                int buf[2];
                MPI_Recv(buf, 2, MPI_INT, i, 0, MPI_COMM_WORLD, &status);

                int first_index = buf[0], n_pixels = buf[1];
                MPI_Recv(&image->p[j][first_index], n_pixels, dt_pixel, i, 0, MPI_COMM_WORLD, &status);
            }
        }
    } else {
        for (j = 0; j < image->n_images; j++) {
            int width = image->width[j];
            int height = image->height[j];
            int line_share = width / mpi_size;
            if (width % mpi_size) line_share++;
            int first_line = (mpi_rank == 0 ? 1 : mpi_rank * line_share);
            int last_line = (mpi_rank == mpi_size - 1 ? height - 2 : (mpi_rank + 1) * line_share - 1);

            int first_index = first_line * width;
            int n_pixels = (last_line - first_line + 1) * width;

            int buf[2] = {first_index, n_pixels};
            MPI_Send(buf, 2, MPI_INT, 0, 0, MPI_COMM_WORLD);
            MPI_Send(&image->p[j][first_index], n_pixels, dt_pixel, 0, 0,MPI_COMM_WORLD);
        }
    }
}

void
mpi_apply_gray_filter( animated_gif * image )
{
    int i, j ;
    pixel ** p ;

    p = image->p ;

    for ( i = 0 ; i < image->n_images ; i++ )
    {
        for ( j = 0 ; j < image->width[i] * image->height[i] ; j++ )
        {
            int moy ;

            // moy = p[i][j].r/4 + ( p[i][j].g * 3/4 ) ;
            moy = (p[i][j].r + p[i][j].g + p[i][j].b)/3 ;
            if ( moy < 0 ) moy = 0 ;
            if ( moy > 255 ) moy = 255 ;

            p[i][j].r = moy ;
            p[i][j].g = moy ;
            p[i][j].b = moy ;
        }
    }
}

void
mpi_apply_blur_filter( animated_gif * image, int size, int threshold )
{
    int i, j, k ;
    int width, height ;
    int end = 0 ;
    int n_iter = 0 ;

    pixel ** p ;
    pixel * new ;

    /* Get the pixels of all images */
    p = image->p ;


    /* Process all images */
    for ( i = 0 ; i < image->n_images ; i++ )
    {
        n_iter = 0 ;
        width = image->width[i] ;
        height = image->height[i] ;

        /* Allocate array of new pixels */
        new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

        /* Perform at least one blur iteration */
        do
        {
            end = 1 ;
            n_iter++ ;

            /* Apply blur on top part of image (10%) */
            for(j=size; j<height/10-size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    int stencil_j, stencil_k ;
                    int t_r = 0 ;
                    int t_g = 0 ;
                    int t_b = 0 ;

                    for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                    {
                        for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                        {
                            t_r += p[i][CONV(j+stencil_j,k+stencil_k,width)].r ;
                            t_g += p[i][CONV(j+stencil_j,k+stencil_k,width)].g ;
                            t_b += p[i][CONV(j+stencil_j,k+stencil_k,width)].b ;
                        }
                    }

                    new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
                    new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                    new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
                }
            }

            /* Copy the middle part of the image */
            for(j=height/10-size; j<height*0.9+size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    new[CONV(j,k,width)].r = p[i][CONV(j,k,width)].r ; 
                    new[CONV(j,k,width)].g = p[i][CONV(j,k,width)].g ; 
                    new[CONV(j,k,width)].b = p[i][CONV(j,k,width)].b ; 
                }
            }

            /* Apply blur on the bottom part of the image (10%) */
            for(j=height*0.9+size; j<height-size; j++)
            {
                for(k=size; k<width-size; k++)
                {
                    int stencil_j, stencil_k ;
                    int t_r = 0 ;
                    int t_g = 0 ;
                    int t_b = 0 ;

                    for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
                    {
                        for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
                        {
                            t_r += p[i][CONV(j+stencil_j,k+stencil_k,width)].r ;
                            t_g += p[i][CONV(j+stencil_j,k+stencil_k,width)].g ;
                            t_b += p[i][CONV(j+stencil_j,k+stencil_k,width)].b ;
                        }
                    }

                    new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
                    new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
                    new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
                }
            }

            for(j=1; j<height-1; j++)
            {
                for(k=1; k<width-1; k++)
                {

                    float diff_r ;
                    float diff_g ;
                    float diff_b ;

                    diff_r = (new[CONV(j  ,k  ,width)].r - p[i][CONV(j  ,k  ,width)].r) ;
                    diff_g = (new[CONV(j  ,k  ,width)].g - p[i][CONV(j  ,k  ,width)].g) ;
                    diff_b = (new[CONV(j  ,k  ,width)].b - p[i][CONV(j  ,k  ,width)].b) ;

                    if ( diff_r > threshold || -diff_r > threshold 
                            ||
                             diff_g > threshold || -diff_g > threshold
                             ||
                              diff_b > threshold || -diff_b > threshold
                       ) {
                        end = 0 ;
                    }

                    p[i][CONV(j  ,k  ,width)].r = new[CONV(j  ,k  ,width)].r ;
                    p[i][CONV(j  ,k  ,width)].g = new[CONV(j  ,k  ,width)].g ;
                    p[i][CONV(j  ,k  ,width)].b = new[CONV(j  ,k  ,width)].b ;
                }
            }

        }
        while ( threshold > 0 && !end ) ;

        // printf( "Nb iter for image %d\n", n_iter ) ;

        free (new) ;
    }

}

void mpi_apply_sobel_filter( animated_gif * image )
{
    int i, j, k ;
    int width, height ;

    pixel ** p ;

    p = image->p ;


    for ( i = 0 ; i < image->n_images ; i++ )
    {
        width = image->width[i] ;
        height = image->height[i] ;

        int line_share = width / mpi_size;
        if (width % mpi_size) line_share++;
        int first_line = (mpi_rank == 0 ? 1 : mpi_rank * line_share);
        int last_line = (mpi_rank == mpi_size - 1 ? height - 2 : (mpi_rank + 1) * line_share - 1);

        pixel * sobel ;

        sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

        for(j = first_line; j <= last_line; j++)
        {
            for(k=1; k<width-1; k++)
            {
                int pixel_blue_no, pixel_blue_n, pixel_blue_ne;
                int pixel_blue_so, pixel_blue_s, pixel_blue_se;
                int pixel_blue_o , pixel_blue  , pixel_blue_e ;

                float deltaX_blue ;
                float deltaY_blue ;
                float val_blue;

                pixel_blue_no = p[i][CONV(j-1,k-1,width)].b ;
                pixel_blue_n  = p[i][CONV(j-1,k  ,width)].b ;
                pixel_blue_ne = p[i][CONV(j-1,k+1,width)].b ;
                pixel_blue_so = p[i][CONV(j+1,k-1,width)].b ;
                pixel_blue_s  = p[i][CONV(j+1,k  ,width)].b ;
                pixel_blue_se = p[i][CONV(j+1,k+1,width)].b ;
                pixel_blue_o  = p[i][CONV(j  ,k-1,width)].b ;
                pixel_blue    = p[i][CONV(j  ,k  ,width)].b ;
                pixel_blue_e  = p[i][CONV(j  ,k+1,width)].b ;

                deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;             

                deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;

                val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;


                if ( val_blue > 50 ) 
                {
                    sobel[CONV(j  ,k  ,width)].r = 255 ;
                    sobel[CONV(j  ,k  ,width)].g = 255 ;
                    sobel[CONV(j  ,k  ,width)].b = 255 ;
                } else
                {
                    sobel[CONV(j  ,k  ,width)].r = 0 ;
                    sobel[CONV(j  ,k  ,width)].g = 0 ;
                    sobel[CONV(j  ,k  ,width)].b = 0 ;
                }
            }
        }

        for(j=1; j<height-1; j++)
        {
            for(k=1; k<width-1; k++)
            {
                p[i][CONV(j  ,k  ,width)].r = sobel[CONV(j  ,k  ,width)].r ;
                p[i][CONV(j  ,k  ,width)].g = sobel[CONV(j  ,k  ,width)].g ;
                p[i][CONV(j  ,k  ,width)].b = sobel[CONV(j  ,k  ,width)].b ;
            }
        }

        free (sobel) ;
    }

}
