python inference.py --cfg inference-config-w18.yaml \
    --videoFile ./seg_test2.mov \
    --writeBoxFrames \
    --outputDir output \
    TEST.MODEL_FILE ../../output/coco/pose_hrnet/w18_small_v1_256x192_adam_lr1e-3/model_best.pth
