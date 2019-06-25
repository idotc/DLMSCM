#!/usr/bin/env sh
CUDA_VISIBLE_DEVICES=0 th multiscale_predict.lua \
  -catPartEnds true \
	-dataset flic \
	-expID flic/dlcm_l3_predict \
	-batchSize 1 \
	-nGPU 1 \
	-nResidual 1 \
	-nThreads 5 \
	-minusMean true \
	-nFeats 256 \
  -struct 3levels_6joints \
  -nSemanticLevels 3 \
  -testOnly true \
  -loadModel checkpoints/saved/dlcm_l3_flic.t7
