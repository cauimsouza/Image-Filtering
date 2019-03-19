extern "C"{

#include <gif.h>
#include <cuda_util.cuh>

#include <stdio.h>
#include <cassert>
#include <sys/time.h>

#define FLATTEN(n_image, n_lin, n_col, width, height) n_image * (width * height) + n_lin * width + n_col
#define NILL 0

int get_leftmost_bit(int n)
{
    if (n == 0)
        return 0;

    int msb = 0;
    while (n != 0) {
        n = n / 2;
        msb++;
        }

    return (1 << msb);
}

__device__ void get_image_indices(int thread_index, int width, int height, int *i, int *j, int *k){
    *i = thread_index / (width * height);
    int in_image_index = thread_index % (width * height);
    *j = in_image_index / width;
    *k = in_image_index % width;
}

__global__ void kernel_sobel(pixel *d_image, pixel *d_sobels, int N, int width, int height, int window_size){
    int index = blockIdx.x *blockDim.x + threadIdx.x;;
    if (index < N * width * height){

        int i, j, k;
        get_image_indices(index, width, height, &i, &j, &k);

        if (j == 0 || j == height - 1 || k == 0 || k == width - 1){
            d_sobels[index] = d_image[index];
            return;
        }

        int offset = i * width * height;
        float deltaX_blue, deltaY_blue, val_blue;
        int pixel_blue_no = d_image[offset + (j-1) * width + k-1].b ;
        int pixel_blue_n  = d_image[offset + (j-1) * width + k  ].b ;
        int pixel_blue_ne = d_image[offset + (j-1) * width + k+1].b ;
        int pixel_blue_so = d_image[offset + (j+1) * width + k-1].b ;
        int pixel_blue_s  = d_image[offset + (j+1) * width + k  ].b ;
        int pixel_blue_se = d_image[offset + (j+1) * width + k+1].b ;
        int pixel_blue_o  = d_image[offset + (j  ) * width + k-1].b ;
        int pixel_blue_e  = d_image[offset + (j  ) * width + k+1].b ;

        deltaX_blue = -pixel_blue_no + pixel_blue_ne - 2*pixel_blue_o + 2*pixel_blue_e - pixel_blue_so + pixel_blue_se;

        deltaY_blue = pixel_blue_se + 2*pixel_blue_s + pixel_blue_so - pixel_blue_ne - 2*pixel_blue_n - pixel_blue_no;

        val_blue = sqrt(deltaX_blue * deltaX_blue + deltaY_blue * deltaY_blue)/4;
        // compare squares instead of sqrt
        d_sobels[index] = (val_blue > 50) ? (pixel){255,255,255} : (pixel){0,0,0};
    }
}

__global__ void kernel_gray(pixel *d_image, pixel *d_gray, int N, int width, int height, int window_size){
    int index = blockIdx.x *blockDim.x + threadIdx.x;;
    if (index < N * width * height){

        int i, j, k;
        get_image_indices(index, width, height, &i, &j, &k);

        int moy = (d_image[index].r + d_image[index].g + d_image[index].b)/3 ;
        if ( moy < 0 ) moy = 0 ;
        if ( moy > 255 ) moy = 255 ;
        d_gray[index] = (pixel) {moy, moy, moy};
    }
}

void
apply_kernel_block( animated_gif * image, void (*kernel_function)(pixel *, pixel *, int, int, int, int),
            int window_size, const char *filter_name = (const char*)"filter", int print_time=1)
{
    struct timeval t1, t2;
    if (print_time)
        gettimeofday(&t1, NULL);

    int i;

    if (!(image->n_images))
        return;

    int N = image->n_images;
    int width = image->width[0], height = image->height[0] ;

    pixel *d_sobels;
    pixel *d_image;
    cudaSetDevice(0);
    cudaMalloc((void**) &d_sobels, N * height * width * sizeof(pixel));
    cudaMalloc((void**) &d_image, N * height * width * sizeof(pixel));
    for (i = 0 ; i < N; i++)
        cudaMemcpy(d_image + i * width * height, image->p[i], width * height * sizeof(pixel), cudaMemcpyHostToDevice);

    int blocks = get_leftmost_bit((N * width * height) / 1024)<<1;
    assert( blocks < INT_MAX);
    kernel_function<<< blocks, 1024 >>>(d_image, d_sobels, N, width, height, window_size);

    for (i = 0 ; i < N; i++)
        cudaMemcpy(image->p[i], d_sobels + i * width * height, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);

    cudaFree(d_sobels);
    cudaFree(d_image);

    if (print_time){
        gettimeofday(&t2, NULL);
        double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        printf("%s done in %lf\n", filter_name, duration);
    }
}

void
apply_kernel_individual( animated_gif * image, void (*kernel_function)(pixel *, pixel *, int, int, int, int),
            int window_size, const char *filter_name = (const char*)"filter", int print_time=1)
{
    struct timeval t1, t2;
    if (print_time)
        gettimeofday(&t1, NULL);
    int i;
    animated_gif single_frame;
    single_frame.n_images = 1;
    single_frame.g = image->g;
    for (i = 0 ; i < image->n_images; i++){
        single_frame.width = (image->width) + i;
        single_frame.height = (image->height) + i;
        single_frame.p = (image->p) + i;
        apply_kernel_block(&single_frame, kernel_function, window_size, filter_name, 0);
    }
    if (print_time){
        gettimeofday(&t2, NULL);
        double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
        printf("%s done in %lf\n", filter_name, duration);
    }
}

void
apply_kernel( animated_gif * image, void (*kernel_function)(pixel *, pixel *, int, int, int, int),
            int window_size, const char *filter_name = (const char*)"filter", int print_time=1, int block=1){
    if (block) apply_kernel_block(image, kernel_function, window_size, filter_name, print_time);
    else apply_kernel_individual(image, kernel_function, window_size, filter_name, print_time);
}

__global__ void kernel_gray_line(pixel *d_image, pixel *d_gray_line, int N, int width, int height, int window_size)
{
    int index = blockIdx.x *blockDim.x + threadIdx.x;;
    if (index < N * width * height){
        int i, j, k;
        get_image_indices(index, width, height, &i, &j, &k);
        d_gray_line[index] = (j >= 0 && j < 10 && k >= width/2 && k < width) ? (pixel) {0, 0, 0} : d_image[index];
    }

}

__global__ void kernel_blur(pixel *d_image, pixel *d_blur, int N, int width, int height, int window_size){
    int index = blockIdx.x *blockDim.x + threadIdx.x;;
    if (index < N * width * height) {
        int i, j, k;
        get_image_indices(index, width, height, &i, &j, &k);

        if (j < window_size || j >= height - window_size || k < window_size || k >= width - window_size ||
            j >= 0.1 * height - window_size || j < 0.9 * height + window_size)
        {
            d_blur[index] = d_image[index];
            return;
        }

        int stencil_j, stencil_k ;
        int t_r = 0 ;
        int t_g = 0 ;
        int t_b = 0 ;

        for ( stencil_j = -window_size ; stencil_j <= window_size ; stencil_j++ )
        {
            for ( stencil_k = -window_size ; stencil_k <= window_size ; stencil_k++ )
            {
                t_r += d_image[FLATTEN(i, j+stencil_j,k+stencil_k,width, height)].r ;
                t_g += d_image[FLATTEN(i, j+stencil_j,k+stencil_k,width, height)].g ;
                t_b += d_image[FLATTEN(i, j+stencil_j,k+stencil_k,width, height)].b ;
            }
        }

        d_blur[FLATTEN(i, j,k,width, height)].r = t_r / ( (2*window_size+1)*(2*window_size+1) ) ;
        d_blur[FLATTEN(i, j,k,width, height)].g = t_g / ( (2*window_size+1)*(2*window_size+1) ) ;
        d_blur[FLATTEN(i, j,k,width, height)].b = t_b / ( (2*window_size+1)*(2*window_size+1) ) ;
    }
}


void apply_sobel_filter (animated_gif *image){
    apply_kernel(image, &kernel_sobel, NILL, (const char*)"sobel", 1);
}

void
apply_gray_filter( animated_gif * image )
{
    apply_kernel(image, &kernel_gray, NILL, (const char*)"gray", 1);
}

void apply_gray_line_filter(animated_gif *image){
    apply_kernel(image, &kernel_gray_line, NILL, (const char*)"gray_line", 1);
}

void get_maximum_diffs(animated_gif image, float **diffs){

}

void apply_blur_filter(animated_gif *image, int size, int threshold){
    const int NUM_MAX_ITER = 1;
    int i;
    int stable = 0;
    float **diffs = (float**) malloc(image->n_images * sizeof(float*));
    for (i = 0; i < image->n_images; i++)
        diffs[i] = (float*) malloc(3 * sizeof(float));
    for(i = 0;i < NUM_MAX_ITER && !stable;i++){
        apply_kernel(image, &kernel_blur, size, (const char*)"blur", 1);
        // get_maximum_diffs(image, diffs);
        // if (abs(diff[0]) < threshold && abs(diff[1]) < threshold && abs(diff[2]) < threshold)
        //     stable = 1;
    }
}

}
