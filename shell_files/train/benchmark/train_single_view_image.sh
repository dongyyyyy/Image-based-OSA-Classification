# image_view="1"
k_fold=10
k_num=0
gpu_id=3

# random seed
# seed=2
num_workers=8
classification_mode='binary'
image_size=448
lr=0.008
dataset_info_path='/data/datasets/filename_list_croped/'


for image_view in "1" "2" "3" "4"
do
    for seed in 0 1 2
    do
        for backbone_architecture in 'efficient_b2' 'efficient_b0' 'resnet101'
        do
            
            model_save_path="/data2/kdy/AHI_classification_2025_01_09/results_single_image_classification_benchmark_models_pretrained_mode_${classification_mode}_croped_images/${backbone_architecture}/image_size_${image_size}_lr_${lr}/image_view_${image_view}/seed_${seed}/"
            python ./train/benchmark/train_single_view_image.py --gpu-id ${gpu_id} --num-workers ${num_workers} --seed ${seed} --image-view ${image_view} --classification-mode ${classification_mode} --k-num ${k_num} --k-fold ${k_fold} --backbone-architecture ${backbone_architecture} --dataset-info-path ${dataset_info_path} --out ${model_save_path} --image-size ${image_size} --lr ${lr}
        done
    done
done