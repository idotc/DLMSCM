# Deeply Learned Compositional Models for Human Pose Estimation

Source code for the paper:
Wei Tang, Pei Yu, and Ying Wu. "Deeply Learned Compositional Models for Human Pose Estimation." in Proceedings of European Conference on Computer Vision (ECCV'18), Munich, Germany, Sept. 2018.

This implementation is based on the code and data from [1-6]. We thank all the authors for kindly sharing these valuable resources.

Please contact Wei Tang (weitang2015@u.northwestern.edu) for questions concerning the code.

## Setting
1. Install Torch
- Option (a): Follow http://torch.ch/docs/getting-started.html
- Option (b): Use the Docker image `kaixhin/cuda-torch:8.0` (https://hub.docker.com/r/kaixhin/cuda-torch/)

2. Install dependences
  ```
  apt-get install libhdf5-serial-dev
  luarocks install hdf5
  ```

3. Prepare datasets
Download MPII [3], LSP [4] and FLIC [5] datasets and create symbolic links so that their respective JPEG images can be found in:
  ```
  data/mpii/images
  data/lsp_dataset/images
  data/lspet_dataset/images
  data/flic/images
  ```

## Training
Train a model with 3 semantic levels on 4 GPUs
  ```
  ./experiments/PLACEHOLDER/train_dlcm_l3.sh
  ```
where PLACEHOLDER can be:
- flic: Train with FLIC training data.
- mpii: Train with MPII training data excluding the 3K validation samples.
- mpii-include-val: Train with MPII training data including the 3K validation samples.
- mpii-lsp: Train with MPII training data and corrected LSP training data.

## Testing
1. Download trained models from our project website and put them in `checkpoints/saved`.

2. Get human pose predictions using a trained model with 3 semantic levels
  ```
  ./experiments/PLACEHOLDER/predict_dlcm_l3.sh
  ```
where PLACEHOLDER can be:
- flic: Predict on FLIC testing data.
- mpii: Predict on MPII 3K validation samples.
- mpii-include-val: Predict on MPII testing data.
- mpii-lsp: Predict on LSP testing data.

3. Evaluate the predictions by comparing them against the corresponding ground truth.
- Check http://human-pose.mpi-inf.mpg.de/#evaluation for evaluation on MPII data.
- Use http://human-pose.mpi-inf.mpg.de/results/lsp/evalLSP.zip for evaluation on LSP data. A few joint annotations in `joints-PC.mat` are on the wrong side. We recommend replacing it with `data/lsp/joints-PC-corrected.mat` for more accurate evaluation. 

## References
[1] Wei Yang, Shuang Li, Wanli Ouyang, Hongsheng Li, and Xiaogang Wang. "Learning feature pyramids for human pose estimation." In ICCV 2017.
[2] Alejandro Newell, Kaiyu Yang, and Jia Deng. "Stacked hourglass networks for human pose estimation." In ECCV 2016.
[3] https://github.com/facebook/fb.resnet.torch
[4] Mykhaylo Andriluka, Leonid Pishchulin, Peter Gehler, and Bernt Schiele. "2d human pose estimation: New benchmark and state of the art analysis." In CVPR 2014.
[5] Sam Johnson and Mark Everingham. "Clustered pose and nonlinear appearance models for human pose estimation." In BMVC 2010.
[6] Ben Sapp and Ben Taskar. "Modec: Multimodal decomposable models for human pose estimation." In CVPR 2013.