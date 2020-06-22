
CUDA_VISIBLE_DEVICES="0,1,2,3" python tools/train.py \
    --cfg experiments/coco/hrnet/w32_256x192_adam_lr1e-3.yaml
