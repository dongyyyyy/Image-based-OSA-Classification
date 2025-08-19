k_fold=10
k_num=0
gpu_id=3

# random seed
# seed=2
num_workers=8
classification_mode='binary'
image_size=448


backbone_architecture='efficient_b2'
positional_embedding=true
shared_attention=false
cls_token=false
lr=0.005

inversed_cross_attention=false
num_heads=8
emb_size=768

backbone_freezing=true
# for backbone_architecture in 'efficient_b0' 'efficient_b1' 'efficient_b2'
# image_view="1,3"
# depth=1
# seed=0
MHDA=false
cross_attention=true
for depth in 3
do
    for image_view in "1,2,3"
    do
        for seed in 0 1 2
        do
            load_path="insert load path of model weights file"
            
            python ./train/benchmark/Multi_view/three_inputs/train_attention_multi_view_image_three_inputs.py --gpu-id ${gpu_id} --num-workers ${num_workers} --seed ${seed} --image-view ${image_view} --depth ${depth} --classification-mode ${classification_mode} --lr ${lr} --k-num ${k_num} --k-fold ${k_fold} --backbone-output-dim 2048 --num-heads ${num_heads} --emb-size ${emb_size} --inversed-cross-attention ${inversed_cross_attention} --pretrained-model-weight-path ${load_path} --backbone-freezing ${backbone_freezing} --out ${model_save_path} --positional-embedding ${positional_embedding} --cls-token ${cls_token} --shared-attention ${shared_attention} --backbone-architecture ${backbone_architecture} --cross-attention ${cross_attention} --MHDA ${MHDA} --image-size ${image_size} --last-mlps ${last_mlps}
        done
    done
done
