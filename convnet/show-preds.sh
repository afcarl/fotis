export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpc_tambet/libjpeg/lib:/home/hpc_tambet/cuda-convnet2/util:/home/hpc_tambet/cuda-convnet2/nvmatrix:/home/hpc_tambet/cuda-convnet2/cudaconv3
export PYTHONPATH=/home/hpc_tambet/cuda-convnet2

python /home/hpc_tambet/cuda-convnet2/shownet.py --load-file $1 --show-preds=probs --channels=1
