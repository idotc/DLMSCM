#!/usr/bin/env sh
CUDA_VISIBLE_DEVICES=0 th multiscale_predict.lua \
  -catPartEnds true \
	-dataset mpii \
	-expID mpii/dlcm_l3_predict \
	-batchSize 1 \
	-nGPU 1 \
	-nResidual 1 \
	-nThreads 5 \
	-minusMean true \
	-nFeats 256 \
  -struct 3levels_16joints \
  -nSemanticLevels 3 \
  -testOnly true \
  -loadModel checkpoints/saved/dlcm_l3_mpii_exclude_val.t7
