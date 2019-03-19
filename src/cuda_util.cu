extern "C"{

#include <gif.h>
#include <cuda_util.cuh>

#include <stdio.h>
#include <cassert>
#include <sys/time.h>

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

__global__ void kernel_sobel(pixel *d_image, pixel *d_sobels, int N, int width, int height){
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

__global__ void kernel_gray(pixel *d_image, pixel *d_gray, int N, int width, int height){
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
apply_kernel( animated_gif * image, void (*kernel_function)(pixel *, pixel *, int, int, int))
{
    struct timeval t1, t2;
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
    // printf("%d %d %d\n", N, width, height);
    // printf("%d %d\n", blocks, 1024);
    assert( blocks < INT_MAX);
    kernel_function<<< blocks, 1024 >>>(d_image, d_sobels, N, width, height);
    // printf(cudaGetErrorString(cudaGetLastError()));
    // printf("\n");

    for (i = 0 ; i < N; i++)
        cudaMemcpy(image->p[i], d_sobels + i * width * height, width * height * sizeof(pixel), cudaMemcpyDeviceToHost);

    cudaFree(d_sobels);
    cudaFree(d_image);

    gettimeofday(&t2, NULL);
    double duration = (t2.tv_sec -t1.tv_sec)+((t2.tv_usec-t1.tv_usec)/1e6);
    printf("Sobel done in %lf\n", duration);
}

__global__ void kernel_gray_line(pixel *d_image, pixel *d_gray_line, int N, int width, int height)
{
    int i, j, k ;

    int index = blockIdx.x *blockDim.x + threadIdx.x;;
    if (index < N * width * height){
        int i, j, k;
        get_image_indices(index, width, height, &i, &j, &k);
        d_gray_line[index] = (j >= 0 && j < 10 && k >= width/2 && k < width) ? (pixel) {0, 0, 0} : d_image[index];
    }

}

void apply_sobel_filter (animated_gif *image){
    apply_kernel(image, &kernel_sobel);
}

void
apply_gray_filter( animated_gif * image )
{
    apply_kernel(image, &kernel_gray);
}

void apply_gray_line_filter(animated_gif *image){
    apply_kernel(image, &kernel_gray_line);
}

}