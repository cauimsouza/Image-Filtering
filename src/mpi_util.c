#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>

#include <mpi_util.h>
#include <stdio.h>

#define CONV(l,c,nb_c)				\
  (l)*(nb_c)+(c)

static MPI_Datatype dt_pixel;
static MPI_Comm gcomm;
static int grank, gsize;

static void create_dt_pixel()
{
  MPI_Type_contiguous(3, MPI_INT, &dt_pixel);
  MPI_Type_commit(&dt_pixel);
}

void mpi_util_init()
{
  create_dt_pixel();
}

static void create_comms(int n_comms)
{
  int color = mpi_rank % n_comms;
  MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank, &gcomm);
  MPI_Comm_rank(gcomm, &grank);
  MPI_Comm_size(gcomm, &gsize);
}

static int is_master(int n_comms) {
  return mpi_rank < n_comms;
}

static int get_npixels(animated_gif *image, int id) {
  return image->width[id] * image->height[id];
}

/* Broadcasts metadata */
static void bcast_meta(animated_gif *image)
{
  MPI_Bcast(&image->n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

  create_comms(image->n_images);

  if (mpi_rank != 0)
    {
      image->width = (int*) malloc(image->n_images * sizeof(int));
      image->height = (int*) malloc(image->n_images * sizeof(int));
    }

  MPI_Bcast(image->width, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(image->height, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
}

/* Each master distributes its image to all the members of its group */
void masters_to_slaves(animated_gif *image)
{
  if (mpi_size > image->n_images)
    {
      int image_id = mpi_rank % image->n_images;
      int npixels = get_npixels(image, image_id);

      if (!is_master(image->n_images) && !image->p)
	{
	  image->p = (pixel**) malloc(image->n_images * sizeof(pixel*));
	  image->p[image_id] = (pixel*) malloc(npixels * sizeof(pixel));
	}

      MPI_Bcast(image->p[image_id], npixels, dt_pixel, 0, gcomm);
    }
}

/* Dungeon master distributes gif to all masters */
void dungeon_master_to_masters(animated_gif *image)
{
  bcast_meta(image);

  /* Distribute images to masters */
  if (is_master(image->n_images) && mpi_rank > 0)
    {
      MPI_Status status;
      image->p = (pixel**) malloc(image->n_images * sizeof(pixel*));
      int i;
      for (i = mpi_rank; i < image->n_images; i += mpi_size)
	{
	  int npixels = get_npixels(image, i);
	  image->p[i] = (pixel*) malloc(npixels * sizeof(pixel));
	  MPI_Recv(image->p[i], npixels, dt_pixel, 0, 0, MPI_COMM_WORLD, &status);
        }
    } else if (mpi_rank == 0)
    {
      int i;
      for (i = 0; i < image->n_images; i++)
        {
	  if (i % mpi_size != 0)
            {
	      int npixels = get_npixels(image, i);
	      /* the master of each communicator is the one who has rank 0
	       * and it's the only one who receives images from the master of
	       * masters (0 of MPI_COMM_WORLD) */
	      MPI_Send(image->p[i], npixels, dt_pixel, i % mpi_size, 0, MPI_COMM_WORLD);
	    }
	}
    }

  //masters_to_slaves(image);
}

static void get_first_last_lines(animated_gif *image, int image_id, int *first_line, int *last_line)
{
  int width = image->width[image_id];
  int height = image->height[image_id];
  int chunk = height / gsize;
  if (height % gsize) chunk++;
  *first_line = chunk * grank;
  *last_line = (grank == gsize - 1 ? height - 1 : (grank + 1) * chunk - 1);
}

static int get_first_pixel(animated_gif *image, int image_id, int *npixels)
{
  int first_line, last_line;
  get_first_last_lines(image, image_id, &first_line, &last_line);

  int width = image->width[image_id];
  *npixels = (last_line - first_line + 1) * width;

  return first_line * width;
}

static void get_displs_array(int *displs, int *recvcounts)
{
  int i;
  int acc = 0;
  for (i = 0; i < gsize; i++)
    {
      displs[i] = acc;
      acc += recvcounts[i];
    }
}

/* Slaves send their work to their master */
void slaves_to_masters(animated_gif *image)
{
  /* In this case each image is treated by only one process */
  if (image->n_images >= mpi_size) return;

  /* In this case, each group treats only one image */
  int image_id = mpi_rank % image->n_images;
  int npixels;
  int first_pixel = get_first_pixel(image, image_id, &npixels); 

  int *recvcounts;
  if (grank == 0) recvcounts = (int*) malloc(gsize * sizeof(int));

  MPI_Gather(&npixels, 1, MPI_INT, recvcounts, 1, MPI_INT, 0, gcomm);

  int *displs;
  if (grank == 0) 
    {
      displs = (int*) malloc(gsize * sizeof(int));
      get_displs_array(displs, recvcounts);
    }

  MPI_Gatherv(&image->p[image_id][first_pixel], npixels,
	      dt_pixel, image->p[image_id], recvcounts,
	      displs, dt_pixel, 0, gcomm);
}

void masters_to_dungeon_master(animated_gif *image)
{
  if (mpi_rank == 0)
    {
      MPI_Status status;
      int i;
      for (i = 0; i < image->n_images; i++)
        {
	  int group_id = i % mpi_size;
	  if (group_id != 0)
            {
	      int npixels = get_npixels(image, i);
	      MPI_Recv(image->p[i], npixels, dt_pixel, group_id, 0, MPI_COMM_WORLD, &status);
	    }
        }
    } else if (is_master(image->n_images))
    {
      int i;
      for (i = mpi_rank; i < image->n_images; i += mpi_size)
        {
	  int npixels = get_npixels(image, i);
	  MPI_Send(image->p[i], npixels, dt_pixel, 0, 0, MPI_COMM_WORLD);
	}
    }
}

void
mpi_apply_gray_filter( animated_gif * image )
{
  int i, j ;
  pixel ** p ;

  p = image->p ;

  for ( i = mpi_rank % image->n_images ; i < image->n_images ; i += mpi_size)
    {
      int first_line, last_line;
      get_first_last_lines(image, i, &first_line, &last_line);

      int width = image->width[i];
      for ( j = first_line * width; j < (last_line + 1) * width ; j++ )

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
  if (!is_master(image->n_images)) return;

  int i, j, k ;
  int width, height ;
  int end = 0 ;
  int n_iter = 0 ;

  pixel ** p ;
  pixel * new ;

  /* Get the pixels of all images */
  p = image->p ;


  /* Process all images */
  for ( i = mpi_rank ; i < image->n_images ; i += mpi_size)
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

  for (i = mpi_rank % image->n_images ; i < image->n_images ; i += mpi_size)
    {
      width = image->width[i] ;
      height = image->height[i] ;

      pixel * sobel ;

      sobel = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

      int first_line, last_line;
      get_first_last_lines(image, i, &first_line, &last_line);
      if (first_line == 0) first_line++;
      if (last_line == height - 1) last_line--;

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
