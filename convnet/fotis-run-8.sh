export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/hpc_tambet/libjpeg/lib

python /home/hpc_tambet/cuda-convnet2/convnet.py --save-path /storage/hpc_tambet/fotis --data-provider cifar --inner-size 24 --test-range 3 --train-range 0-2 --data-path /storage/hpc_tambet/fotis/isikud_batches_color_996_49 --gpu 0 --layer-def ~/fotis/layers/layers-fotis-simple-color-49.cfg --layer-params ~/fotis/layers/layer-params-fotis.cfg --epochs 1000
