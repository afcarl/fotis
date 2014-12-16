export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpc_tambet/libjpeg/lib

python /home/hpc_tambet/cuda-convnet2/convnet.py --save-path /storage/hpc_tambet/fotis --data-provider fotis --inner-size 40 --test-range 4 --train-range 0-3 --data-path /storage/hpc_tambet/fotis/isikud_batches_1000_64x64_gray --gpu 0 --layer-def ~/fotis/layers/layers-fotis-simple-gray-49.cfg --layer-params ~/fotis/layers/layer-params-fotis.cfg --epochs 1000
