export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpc_tambet/libjpeg/lib

python /home/hpc_tambet/cuda-convnet2/shownet.py --load-file $1 --show-filters=conv1
