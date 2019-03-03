import subprocess
import os
import sys

gif_folder = 'images/original'
binary_name = 'sobelf'

gif_images = [f for f in os.listdir(gif_folder) if f.endswith('.gif')]

n_threads = [1, 2, 3, 4]
n_procs = [1, 2, 3, 4]

omp_command = "OMP_NUM_THREADS={}"
mpi_command = "mpirun -n {}"
binary_command = "./{} {}/".format(binary_name, gif_folder) + "{} o.gif"

for nt in n_threads:
    for np in n_procs:
        n_imgs = 0
        sum_time = 0
        for gif in gif_images:
            try:
                command = ' '.join((omp_command.format(nt), mpi_command.format(np), binary_command.format(gif)))
                result = subprocess.run(command, shell=True, stdout=subprocess.PIPE)
                result = result.stdout.decode('utf-8')
                result = float(result)

                sum_time += result
                n_imgs += 1
            except:
                pass
        print("{} threads, {} procs, {} images: {} s".format(nt, np, n_imgs, sum_time))
