# DRRN_CVPR17
Code for our CVPR'17 paper "Image Super-Resolution via Deep Recursive Residual Network"

# Implement adjustable gradient clipping 
modify sgd_solver.cpp in your_caffe_root/src/caffe/solvers/, where we add the following codes in funciton ClipGradients():

Dtype rate = GetLearningRate();

const Dtype clip_gradients = this->param_.clip_gradients()/rate;

# Training
1. Preparing training/validation data using the files: generate_trainingset_x234/generate_testingset_x234 in "data" folder. "Train_291" folder contains 291 training images and "Set5" folder is a popular benchmark dataset.
2. We release two DRRN architectures: DRRN_B1U9_20C128 and DRRN_B1U25_52C128 in "caffe_files" folder. Choose either one to do training. E.g., run ./train_DRRN_B1U9_20C128.sh

# Test
1. Remember to compile the matlab wrapper: make matcaffe, since we use matlab to do testing.
2. We release two pretrained models: DRRN_B1U9_20C128 and DRRN_B1U25_52C128 in "model" folder. Choose either one to do testing on benchmark Set5. E.g., run file ./test/DRRN_B1U9_20C128/test_DRRN_B1U9, the results are stored in "results" folder, with both reconstructed images and PSNR/SSIM/IFCs.

<table align="center">
  <tr>
    <td> Dataset </td>
    <td> Scale </td>
    <td> Bicubic </td>
    <td> SRCNN </td>
    <td> SelfEx </td>
    <td> RFL </td>
    <td> VDSR </td>
    <td> DRCN </td>
    <td> DRRN_B1U9 </td>
    <td> DRRN_B1U25 </td>
  </tr>
  <tr>
    <td rowspan=3> Set5 </td>
    <td> x2 </td>
    <td> 33.66/0.9299 </td>
    <td> 36.66/0.9542 </td>
    <td> 36.49/0.9537 </td>
    <td> 36.54/0.9537 </td>
    <td> 37.53/0.9587 </td>
    <td> 37.63/0.9588 </td>
    <td> 37.66/0.9589 </td>
    <td> 37.74/0.9591 </td>
  </tr>
  <tr>
    <td> x3 </td>
    <td> 30.39/0.8682 </td>
    <td> 32.75/0.9090 </td>
    <td> 32.58/0.9093 </td>
    <td> 32.43/0.9057 </td>
    <td> 33.66/0.9213 </td>
    <td> 33.82/0.9226 </td>
    <td> 33.93/0.9234 </td>
    <td> 34.03/0.9244 </td>
  </tr>
  <tr>
    <td> x4 </td>
    <td> 28.42/0.8104 </td>
    <td> 30.48/0.8628 </td>
    <td> 30.31/0.8619 </td>
    <td> 30.14/0.8548 </td>
    <td> 31.35/0.8838 </td>
    <td> 31.53/0.8854 </td>
    <td> 31.58/0.8864 </td>
    <td> 31.68/0.8888 </td>
  </tr>  
</table>

![](figures/final.png) 
