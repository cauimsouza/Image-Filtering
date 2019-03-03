import subprocess
import os
import sys

if len(sys.argv) < 2:
    raise Exception('usage: python {} NPROC'.format(sys.argv[0]))

n_proc = int(sys.argv[1])

gif_images = [f for f in os.listdir('images/original') if f.endswith('.gif')]

command = "mpirun -n {} ".format(n_proc) + " ./sobelf images/original/{} o.gif"
n_imgs = 0
sum_time = 0
for gif in gif_images:
    print("processing {}".format(gif))
    try:
        result = subprocess.run(command.format(gif), shell=True, stdout=subprocess.PIPE)
        result = result.stdout.decode('utf-8')
        result = float(result)

        sum_time += result
        n_imgs += 1
    except:
        pass

print("{} s for {} images with {} processes".format(sum_time, n_imgs, n_proc))
