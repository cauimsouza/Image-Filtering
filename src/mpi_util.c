#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <mpi.h>
#include <assert.h>
#include <omp.h>

#include <mpi_util.h>
#include <stdio.h>

#define CONV(l,c,nb_c)				\
  (l)*(nb_c)+(c)

#define ROW(offset, width) ((offset) / (width))
#define COL(offset, width) ((offset) % (width))

static MPI_Datatype dt_pixel;
static MPI_Comm gcomm, bcomm;
static int grank, gsize, brank, bsize;

static void create_dt_pixel()
{
  MPI_Type_contiguous(3, MPI_INT, &dt_pixel);
  MPI_Type_commit(&dt_pixel);
}

static void create_comms(int n_comms)
{
  int color = mpi_rank % n_comms;
  MPI_Comm_split(MPI_COMM_WORLD, color, mpi_rank, &gcomm);
  MPI_Comm_rank(gcomm, &grank);
  MPI_Comm_size(gcomm, &gsize);
}

/* Broadcasts metadata */
static void bcast_meta(animated_gif *image)
{
  MPI_Bcast(&image->n_images, 1, MPI_INT, 0, MPI_COMM_WORLD);

  if (mpi_rank != 0)
    {
      image->p = (pixel**) malloc(image->n_images * sizeof(pixel*));
      int i;
      for (i = 0; i < image->n_images; i++)
	image->p[i] = NULL;

      image->width = (int*) malloc(image->n_images * sizeof(int));
      image->height = (int*) malloc(image->n_images * sizeof(int));
    }

  MPI_Bcast(image->width, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
  MPI_Bcast(image->height, image->n_images, MPI_INT, 0, MPI_COMM_WORLD);
}

void mpi_util_init(animated_gif *image)
{
  create_dt_pixel();
  bcast_meta(image);
  create_comms(image->n_images);
}

static int is_master(int n_comms) {
  return grank == 0;
}

static int get_npixels(animated_gif *image, int id) {
  return image->width[id] * image->height[id];
}

/* Each master distributes its image to all the members of its group */
void masters_to_slaves(animated_gif *image)
{
  if (mpi_size > image->n_images)
    {
      int image_id = mpi_rank % image->n_images;
      int npixels = get_npixels(image, image_id);

      if (!is_master(image->n_images) && !image->p[image_id])
	image->p[image_id] = (pixel*) malloc(npixels * sizeof(pixel));

      MPI_Bcast(image->p[image_id], npixels, dt_pixel, 0, gcomm);
    }
}

/* Dungeon master distributes gif to all masters */
void dungeon_master_to_masters(animated_gif *image)
{
  /* Distribute images to masters */
  if (is_master(image->n_images) && mpi_rank > 0)
    {
      MPI_Status status;
      int i;
      for (i = mpi_rank; i < image->n_images; i += mpi_size)
	{
	  int npixels = get_npixels(image, i);
	  image->p[i] = (pixel*) malloc(npixels * sizeof(pixel));
	  MPI_Recv(image->p[i], npixels, dt_pixel, 0, 0, MPI_COMM_WORLD, &status);
        }
    }
  else if (mpi_rank == 0)
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

void mpi_apply_gray_filter(animated_gif * image)
{

#pragma omp parallel default(none) shared(image, mpi_rank, mpi_size)
  {
#pragma omp single
    {
      int i;
#pragma omp taskloop default(none) nogroup	\
  shared(image)
      for ( i = mpi_rank % image->n_images ; i < image->n_images ; i += mpi_size)
	{
	  int first_line, last_line;
	  get_first_last_lines(image, i, &first_line, &last_line);

	  pixel *p = image->p[i];
	  int width = image->width[i];
	  int j;
#pragma omp taskloop default(none) nogroup \
  firstprivate(p)
	  for ( j = first_line * width; j < (last_line + 1) * width ; j++ )

	    {
	      int moy ;

	      moy = (p[j].r + p[j].g + p[j].b)/3 ;
	      if ( moy < 0 ) moy = 0 ;
	      if ( moy > 255 ) moy = 255 ;

	      p[j].r = moy ;
	      p[j].g = moy ;
	      p[j].b = moy ;
	    }
	}
    }
  }
}

/* Create communicators for blur filter processing */
void create_blur_comm()
{
  int color = grank % 2;
  MPI_Comm_split(gcomm, color, grank, &bcomm);
  MPI_Comm_rank(bcomm, &brank);
  MPI_Comm_size(bcomm, &bsize);
}

/* Calculates region on which the process should work.
   first_pixel is an offset from the first pixel of the image,
   n_pixels is the number of pixels of the image for which the
   process is responsable.
*/
static void calculate_domain(int width, int height, int size,
			     int lower_bound, int upper_bound,
			     int *first_pixel, int *n_pixels)
{
  int bottom = grank % 2;
  if (bottom)
    {
      int n_pixels_total = (height - size - lower_bound) * width;
      int slice_size = n_pixels_total / bsize;
      *first_pixel = brank * slice_size + lower_bound * width;
      *n_pixels = slice_size;
      if (brank == bsize - 1)
	*n_pixels = n_pixels_total - (bsize - 1) * slice_size;
    }
  else
    {
      int n_pixels_total = (upper_bound - size) * width;
      int slice_size = n_pixels_total / bsize;
      *first_pixel = brank * slice_size + size * width;
      *n_pixels = slice_size;
      if (brank == bsize - 1)
	*n_pixels = n_pixels_total - (bsize - 1) * slice_size;
    }

  assert(*n_pixels > 0);
  assert(*first_pixel >= 0);
  assert(*first_pixel + *n_pixels <= width * height);
}

static void get_sendcounts(int n_pixels, int *result) {
  for (int i = 0; i < bsize; i++)
    result[i] = n_pixels;
}

static void get_sdispls(int *result) {
  for (int i = 0; i < bsize; i++)
    result[i] = 0;
}

static void get_recvcounts(int width, int height, int size,
			   int lower_bound, int upper_bound, int *recvcounts)
{
  int bottom = grank % 2;
  int n_pixels_total;
  if (bottom)
    n_pixels_total = (height - size - lower_bound) * width;
  else
    n_pixels_total = (upper_bound - size) * width;

  int slice_size = n_pixels_total / bsize;

  int i;
  for (i = 0; i < bsize - 1; i++)
    recvcounts[i] = slice_size;
  recvcounts[bsize - 1] = n_pixels_total - (bsize - 1) * slice_size;
}

static void get_rdispls(int width, int size, int lower_bound, int *recvcounts, int *rdispls)
{
  int bottom = grank % 2;
  int offset;
  if (bottom)
    offset = lower_bound * width;
  else
    offset = size * width;

  int i;
  for (i = 0; i < bsize - 1; i++)
    {
      rdispls[i] = offset;
      offset += recvcounts[i];
    }
  rdispls[bsize - 1] = offset;
}

static void blur_filter_parallel(animated_gif * image, int size, int threshold)
{
  create_blur_comm();
  int bottom = grank % 2;

  int image_id = mpi_rank % image->n_images;

  pixel *p = image->p[image_id];

  int width = image->width[image_id],
    height = image->height[image_id];
  pixel *new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

  int upper_bound = height / 10 - size; // this line is NOT treated
  int lower_bound = height * 0.9 + size; // this line IS treated

  int first_pixel, n_pixels;
  calculate_domain(width, height, size, lower_bound, upper_bound, &first_pixel, &n_pixels);

  int *sendcounts = (int*) malloc(bsize * sizeof(int));
  get_sendcounts(n_pixels, sendcounts);

  int *sdispls = (int*) malloc(bsize * sizeof(int));
  get_sdispls(sdispls);

  int *recvcounts = (int*) malloc(bsize * sizeof(int));
  get_recvcounts(width, height, size, lower_bound, upper_bound, recvcounts);

  int *rdispls = (int*) malloc(bsize * sizeof(int));
  get_rdispls(width, size, lower_bound, recvcounts, rdispls);

  int end = 0 ;
  do
    {
      end = 1 ;

      int i;
      for (i = 0; i < n_pixels; i++)
	{
	  int of = first_pixel + i;
	  int j = ROW(of, width);
	  int k = COL(of, width);
	  if (k < size || k >= width - size)
	    continue;

	  int stencil_j, stencil_k ;
	  int t_r = 0 ;
	  int t_g = 0 ;
	  int t_b = 0 ;

	  for ( stencil_j = -size ; stencil_j <= size ; stencil_j++ )
	    {
	      for ( stencil_k = -size ; stencil_k <= size ; stencil_k++ )
		{
		  t_r += p[CONV(j+stencil_j,k+stencil_k,width)].r ;
		  t_g += p[CONV(j+stencil_j,k+stencil_k,width)].g ;
		  t_b += p[CONV(j+stencil_j,k+stencil_k,width)].b ;
		}
	    }

	  new[CONV(j,k,width)].r = t_r / ( (2*size+1)*(2*size+1) ) ;
	  new[CONV(j,k,width)].g = t_g / ( (2*size+1)*(2*size+1) ) ;
	  new[CONV(j,k,width)].b = t_b / ( (2*size+1)*(2*size+1) ) ;
	}

      for (i = 0; i < n_pixels; i++)
	{
	  int of = first_pixel + i;
	  int j = ROW(of, width);
	  int k = COL(of, width);
	  if (k < size || k >= width - size)
	    continue;

	  float diff_r ;
	  float diff_g ;
	  float diff_b ;

	  diff_r = (new[CONV(j  ,k  ,width)].r - p[CONV(j  ,k  ,width)].r) ;
	  diff_g = (new[CONV(j  ,k  ,width)].g - p[CONV(j  ,k  ,width)].g) ;
	  diff_b = (new[CONV(j  ,k  ,width)].b - p[CONV(j  ,k  ,width)].b) ;

	  if ( diff_r > threshold || -diff_r > threshold 
	       ||
	       diff_g > threshold || -diff_g > threshold
	       ||
	       diff_b > threshold || -diff_b > threshold
	       ) {
	    end = 0 ;
	  }

	  p[CONV(j  ,k  ,width)].r = new[CONV(j  ,k  ,width)].r ;
	  p[CONV(j  ,k  ,width)].g = new[CONV(j  ,k  ,width)].g ;
	  p[CONV(j  ,k  ,width)].b = new[CONV(j  ,k  ,width)].b ;
	}

      MPI_Alltoallv(&p[first_pixel], sendcounts, sdispls, dt_pixel, p, recvcounts, rdispls, dt_pixel, bcomm);

      int my_end = end;
      MPI_Allreduce(&my_end, &end, 1, MPI_INT, MPI_LAND, gcomm);
    }
  while (threshold > 0 && !end) ;

  free (new) ;

  free(sendcounts);
  free(sdispls);
  free(recvcounts);
  free(rdispls);

  /* 
     fills the missing part of one group with the filtered part of the other group.
     The processes with grank 0 and 1 have necessarily treated different parts of
     the image.
  */
  MPI_Bcast(&p[size * width], (upper_bound - size) * width, dt_pixel, 0, gcomm);
  MPI_Bcast(&p[lower_bound * width], (height - size - lower_bound) * width, dt_pixel, 1, gcomm);
}

static void blur_filter_sequential(animated_gif * image, int size, int threshold)
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
  for ( i = mpi_rank ; i < image->n_images ; i += mpi_size)
    {
      n_iter = 0 ;
      width = image->width[i] ;
      height = image->height[i] ;

      /* Allocate array of new pixels */
      new = (pixel *)malloc(width * height * sizeof( pixel ) ) ;

      int upper_bound = height / 10 - size; // this line is NOT treated 
      int lower_bound = height * 0.9 + size; // this line IS treated

      /* Perform at least one blur iteration */
      do
        {
	  end = 1 ;
	  n_iter++ ;

	  /* Apply blur on top part of image (10%) */
	  for(j=size; j<upper_bound; j++)
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

	  /* Apply blur on the bottom part of the image (10%) */
	  for(j=lower_bound; j<height-size; j++)
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

	  for(j = size; j < upper_bound; j++)
            {
	      for(k = size; k < width - size; k++)
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

	  for(j = lower_bound; j < height - size; j++)
            {
	      for(k = size; k < width - size; k++)
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

      free (new) ;
    }
}

/* Applies blur filter on image. Supposes every MPI process
   has a copy of the images corresponding to its group.
*/
void
mpi_apply_blur_filter( animated_gif * image, int size, int threshold )
{
  if (gsize > 1)
    blur_filter_parallel(image, size, threshold);
  else
    blur_filter_sequential(image, size, threshold);
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
