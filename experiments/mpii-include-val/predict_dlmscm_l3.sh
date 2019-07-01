#!/usr/bin/env sh
CUDA_VISIBLE_DEVICES=0 th multiscale_predict.lua \
  -catPartEnds true \
	-dataset mpii-include-val \
	-expID mpii-include-val/dlcm_l3_predict \
	-batchSize 1 \
	-nGPU 1 \
	-nResidual 1 \
	-nThreads 5 \
	-minusMean true \
	-nFeats 256 \
  -struct 3levels_16joints \
  -nSemanticLevels 3 \
  -testRelease true \
  -loadModel checkpoints/model_250.t7
