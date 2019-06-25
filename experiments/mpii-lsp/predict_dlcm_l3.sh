#!/usr/bin/env sh
CUDA_VISIBLE_DEVICES=0 th multiscale_predict.lua \
  -catPartEnds true \
	-dataset mpii-lsp \
	-expID mpii-lsp/dlcm_l3_predict \
	-batchSize 1 \
	-nGPU 1 \
	-nResidual 1 \
	-nThreads 5 \
	-minusMean true \
	-nFeats 256 \
  -struct 3levels_14joints \
  -nSemanticLevels 3 \
  -testOnly true \
  -loadModel checkpoints/saved/model_250.t7
