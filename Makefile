SRC_DIR=src
HEADER_DIR=include
OBJ_DIR=obj

include env.sh

CC=nvcc
CFLAGS=-O3 -I$(HEADER_DIR) -I${MPI_INCLUDE} -L${MPI_LIB}
LD_FLAGS=-lm -lmpi
OMP_FLAGS=-Xcompiler -fopenmp

SRC= dgif_lib.c \
	egif_lib.c \
	gif_err.c \
	gif_font.c \
	gif_hash.c \
	gifalloc.c \
	openbsd-reallocarray.c \
	cuda_util.cu \
	quantize.c \
	main.c

OBJ= $(OBJ_DIR)/dgif_lib.o \
	$(OBJ_DIR)/egif_lib.o \
	$(OBJ_DIR)/gif_err.o \
	$(OBJ_DIR)/gif_font.o \
	$(OBJ_DIR)/gif_hash.o \
	$(OBJ_DIR)/gifalloc.o \
	$(OBJ_DIR)/openbsd-reallocarray.o \
	$(OBJ_DIR)/cuda_util.o \
	$(OBJ_DIR)/quantize.o \
	$(OBJ_DIR)/main.o

all: $(OBJ_DIR) filter

$(OBJ_DIR):
	mkdir $(OBJ_DIR)

$(OBJ_DIR)/%.o : $(SRC_DIR)/%.c*
	$(CC) $(CFLAGS) -c -o $@ $^

filter:$(OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LD_FLAGS) $(OMP_FLAGS)

clean:
	rm -f filter -rf $(OBJ_DIR)
