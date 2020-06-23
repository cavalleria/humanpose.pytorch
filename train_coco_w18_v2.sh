
CUDA_VISIBLE_DEVICES="0,1,2,3" python tools/train.py \
    --cfg experiments/coco/hrnet/w18_small_v2_256x192_adam_lr1e-3_softargmax.yaml
