export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpc_tambet/libjpeg/lib:/home/hpc_tambet/cuda-convnet2/util:/home/hpc_tambet/cuda-convnet2/nvmatrix:/home/hpc_tambet/cuda-convnet2/cudaconv3
export PYTHONPATH=/home/hpc_tambet/cuda-convnet2

#python /home/hpc_tambet/cuda-convnet2/convnet.py --load-file $1 --test-only 1 --logreg-name logprob --multiview-test 1 --test-out .
python /home/hpc_tambet/cuda-convnet2/convnet.py --test-only 1 --write-features probs --feature-path . --load-file $*
