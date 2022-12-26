task='GenShow'

if [ ${task} = 'GenShow' ];then
    python blend_models_show.py \
    --low-res-pkl=networks/stylegan2-ffhq-config-f.pkl \
    --high-res-pkl=results/00007-stylegan2-watercolour-4gpu-config-f/network-snapshot-000024.pkl \
    --generate-num=6 --result-dir ./results/watercolour-24 --seed 0
fi

if [ ${task} = 'GenBatch' ];then
    python blend_models_batch.py \
    --low-res-pkl=networks/stylegan2-ffhq-config-f.pkl \
    --high-res-pkl=results/00007-stylegan2-watercolour-4gpu-config-f/network-snapshot-000024.pkl \
    --resolution=8 --generate-num=40 --result-dir ./results/watercolour-24
fi