CUDA_VISIBLE_DEVICES=0,1,2,3 python run_training.py --num-gpus=4 \
--data-dir=/home/jiaqing/DataSets/stylegan2/watercolour --config=config-f \
--dataset=watercolour --mirror-augment=true