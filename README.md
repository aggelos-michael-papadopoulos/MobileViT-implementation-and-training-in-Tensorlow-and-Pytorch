## Mobile ViT implementation in both Tensorflow and Pytorch

The repository contains the code for the implementation of MobileViT in TensorFlow and in Pytorch. 
We train both of our networks in Caltech-256.

Paper: https://arxiv.org/pdf/2110.02178.pdf

# MobileViT presentation results. We also did training on non squered images (150x1920). Check the representation below to find out how it can be done
[MobileViT presentation.pptx](https://github.com/aggelos-michael-papadopoulos/xobile-ViT-impelmentation-and-training-on-Tensorlow-and-Pytorch/files/9998095/MobileViT.presentation.pptx)


![mvit](https://user-images.githubusercontent.com/65830412/201538565-090cfb7a-822f-48cb-9197-c687d61e9541.gif)

# Mobile-ViT architecture: 

![2022-11-13 20_44_26-2-Figure1-1 png ‎- Photos](https://user-images.githubusercontent.com/65830412/201538771-76ecfe36-fb08-4f05-aa01-9084f382a3b0.png)

 
The architecture constists of 3 Blocks:
 1) Simple Convolution blocks
 
 2) MV2 Blocks: Inverted residual blocks from MobileNetV2 (https://arxiv.org/pdf/1801.04381.pdf)

![inverted_residual_block](https://user-images.githubusercontent.com/65830412/201540063-40e3518b-358b-4f0b-a722-f50013088e57.jpg)

 
 3) Mobile ViT block: convolutions for local features + transformers for global features
 
![mobilevit-block](https://user-images.githubusercontent.com/65830412/201541247-01060e0a-82f2-4533-88f7-cb16b507a6c2.jpg)


# DATA COLLECTION
We download the caltech_256 from https://www.kaggle.com/datasets/jessicali9530/caltech256

# Train on Pytorch

1) We execute the "fix_caltech_256_pytorch.py" and get the train and validation data to a "caltech_data.csv". 

--> the one that i have uploaded (caltech_data.csv) contains the images on my computer's paths so you have to run the fix_caltech_256_pytorch.py to get your own csv file <--

2) We excecute the "paper_benchmark_pytorch.py" for training. We use Weights and Biases (https://wandb.ai/site) for visualizing our results. If you do not want to use it, just simple write "False" as the value of "wandb" key in the Config dictionary (in line 52 just write: "wand": "False")

3) Last we excecute "torch_inference.py" to see how fast mobilevit can inference an image on both GPU and CPU. Our results can be seen below:

![2022-11-13 21_31_29-Mobile-ViT   EfficientFormer presentation - Παρουσιάσεις Google](https://user-images.githubusercontent.com/65830412/201540580-2adcb5c0-8574-4be8-b358-79c88f5da730.png)

# Train on Tensorflow

1) We execute the "fix_caltech_256_tensorflow.py". Here we just seperate our data to train and val folder (because that is how tensorflow wants the dataset to be). To complete this step we must have already have the "caltech_data.csv" (see step 1 from # Train on Pytorch)

2) We excecute the "paper_benchmark_tensorflow.py" for training. We use Weights and Biases (https://wandb.ai/site) for visualizing our results. If you do not want to use it, just simple write "False" as the value of "wandb" key in the Config dictionary (in line 52 just write: "wand": "False")
