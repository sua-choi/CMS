export CUDA_VISIBLE_DEVICES=0

###################################################
# GCD
###################################################

python -m methods.meanshift_clustering \
        --dataset_name cifar100 \
        --warmup_model_dir 'cifar100_best'

python -m methods.meanshift_clustering \
        --dataset_name imagenet_100 \
        --warmup_model_dir 'imagenet_best'

python -m methods.meanshift_clustering \
        --dataset_name cub \
        --warmup_model_dir 'cub_best' 

python -m methods.meanshift_clustering \
        --dataset_name scars \
        --warmup_model_dir 'scars_best' 

python -m methods.meanshift_clustering \
        --dataset_name aircraft \
        --warmup_model_dir 'aircraft_best' 

python -m methods.meanshift_clustering \
        --dataset_name herbarium_19 \
        --warmup_model_dir 'herbarium_19_best' 

###################################################
# INDUCTIVE GCD
###################################################

python -m methods.meanshift_clustering \
        --dataset_name cifar100 \
        --warmup_model_dir 'cifar100_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name imagenet_100 \
        --warmup_model_dir 'imagenet_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name cub \
        --warmup_model_dir 'cub_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name scars \
        --warmup_model_dir 'scars_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name aircraft \
        --warmup_model_dir 'aircraft_best' \
        --inductive

python -m methods.meanshift_clustering \
        --dataset_name herbarium_19 \
        --warmup_model_dir 'herbarium_19_best' \
        --inductive