## Mobile ViT implementation in both Tensorflow and Pytorch

The repository contains the code for the implementation of MobileViT in TensorFlow and in Pytorch. 
We train both of our networks in Caltech-256.

Paper: https://arxiv.org/pdf/2110.02178.pdf


![mvit](https://user-images.githubusercontent.com/65830412/201538565-090cfb7a-822f-48cb-9197-c687d61e9541.gif)

# Mobile-ViT architecture: 

![2022-11-13 20_44_26-2-Figure1-1 png ‎- Photos](https://user-images.githubusercontent.com/65830412/201538771-76ecfe36-fb08-4f05-aa01-9084f382a3b0.png)

 
The architecture constists of 3 Blocks:
 1) Simple Convolution blocks
 
 2) MV2 Blocks: Inverted residual blocks from MobileNetV2 (https://arxiv.org/pdf/1801.04381.pdf)
![inverted_residual_block](https://user-images.githubusercontent.com/65830412/201540063-40e3518b-358b-4f0b-a722-f50013088e57.jpg)

 
 3) Mobile ViT block: convolutions for local features + transformers for global features


# DATA COLLECTION
We download the caltech_256 from https://www.kaggle.com/datasets/jessicali9530/caltech256

# Train on Pytorch

1) We execute the "fix_caltech_256_pytorch.py" and get the train and validation data to a "caltech_data.csv". 

--> the one that i have uploaded (caltech_data.csv) contains the images on my computer's paths so you have to run the fix_caltech_256_pytorch.py to get your own csv file <--

2) We excecute "paper_benchmark_pytorch.py" for training. We use Weights and Biases (https://wandb.ai/site) for visualizing our results. If you do not want to use it, just simple write "False" in the value of "wandb" key in the Config file (in line 52 write: "wand": "False")
