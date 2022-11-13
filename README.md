## Mobile ViT implementation in both Tensorflow and Pytorch

The repository contains the code for the implementation of MobileViT in TensorFlow and in Pytorch. 
We train both of our networks in Caltech-256.

Paper: https://arxiv.org/pdf/2110.02178.pdf


![mvit](https://user-images.githubusercontent.com/65830412/201538565-090cfb7a-822f-48cb-9197-c687d61e9541.gif)

# Mobile-ViT architecture: 

![2022-11-13 20_44_26-2-Figure1-1 png ‎- Photos](https://user-images.githubusercontent.com/65830412/201538771-76ecfe36-fb08-4f05-aa01-9084f382a3b0.png)

 
It constists of:
 1) Conv blocks
 
 2) Inverted residual blocks from MobileNetV2 (https://arxiv.org/pdf/1801.04381.pdf) 
 ![inv--](https://user-images.githubusercontent.com/65830412/201538843-c11d165b-991e-403e-b6e7-50967fffa8b9.png)
 
 3) Mobile ViT block


# TRAINING
**/ 
