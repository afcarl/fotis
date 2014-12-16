export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpc_tambet/libjpeg/lib:/home/hpc_tambet/cuda-convnet2/util:/home/hpc_tambet/cuda-convnet2/nvmatrix:/home/hpc_tambet/cuda-convnet2/cudaconv3
export PYTHONPATH=/home/hpc_tambet/cuda-convnet2

python /home/hpc_tambet/cuda-convnet2/convnet.py --save-path /storage/hpc_tambet/fotis --data-provider fotis --inner-size 48 --test-range 4 --train-range 0-4 --test-freq 1000 --force-save 1 --data-path /storage/hpc_tambet/fotis/isikud_batches_1001_64x64_gray_23 --gpu 0 --layer-def ~/fotis/layers/layers-fotis-simple-gray-23.cfg --layer-params ~/fotis/layers/layer-params-fotis.cfg --epochs 1000
