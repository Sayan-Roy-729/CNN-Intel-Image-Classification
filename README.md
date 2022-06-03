The custom model, you can find [here](https://drive.google.com/file/d/18F861Ppm2lKEQ4AdNC8cYhjw-jjl7ARp/view?usp=sharing).

<h1 style="align:center;">Intel Image Classification</h1>

<h2 style="color:red;">Problem Statement</h2>

I have got an image classification dataset on [Kaggle](https://www.kaggle.com) (the dataset link is givcen below). There are almost 25k images. Every image size is 150x150. And there are total **6 categories/classes named "buildings", "forest", "glacier", "mountain", "sea" and "street"**.

Inside the dataset directory, there are 3 sub-directories, one for *training*, one for *testing* and another one for *validation*. I used the *training* and *testing* datasets. For training, there are around 14k images and for testing there are around 3k images. Around 82% data is for testing and 18% data is for training.

When I looked out through the training images, I found that some "mountain" images are covered with ice fow which model will be confused between two classes, "mountain" and "glacier". This will also happen with more two classes, "street" and "buildings". Buildings can be seen in some "street" images.

**My problem is to classify the images correctly and also visualize how do the models see the images by *feature maps*.** For this, I have build 2 models. One is my custom model and the other one is the VGG-16.

---

<h2 style="color:red">Dataset</h2>

The dataset which I have used here, you can find in [Kaggle](https://www.kaggle.com/datasets/puneet6060/intel-image-classification).

---

<h2 style="color:red">My Solution Approach</h2>

<h3 style="color:cyan">Tools and modules</h3>

To solve this problem, I have used these exeternal modules:
- `PyTorch` (GPU version)
- `NumPy`
- `torchvision`
- `torchsummary`
- `Matplotlib`

<h3 style="color:cyan">Import the dataset & Image Transformations</h3>

- There are 6 subdirectories (for each class/category) inside the *training* directory. Same goes to the *testing* directory also. 
- I used `torchvision.dataset.ImageFolder` module (which is suitable for this kind of folder structure) that loads the images and their respective labels.
- After this, I have used `torch.utils.data.DataLoader` to create *train data loader* and *test data loader* which will help to pass the data to the model while training. It also helps to create batches simultaneously.
- I have created batches with 128 batch sizes.
- I also did image transformation like `normalization`, `resize` to 150x150 images, `RandomHorizontalFlip` to flip the random selected images horizontally and the finally to `tensor` object.
- For VGG-16 model, I resized images to 224x224 and normalize according to the documentation (mean=[0.485, 0.456, 0.406] and std = [0.229, 0.224, 0.225]).

<h3 style="color:cyan">Build Custom Model</h3>

- To build my custom nuural network, I have used `torch.nn.Module` to create a class for the model architecture. I also wrote some code to get the feature maps of the first convolution layer and 2nd convolution layer.
- Then have created a function that can create an instance of that class. Also choose the `torch.nn.CrossEntropyLoss` for *multi class classification*. This `loss function` use `SoftMax activation function`. So I didn't have to specify the activation function for the last layer. And choose the `Adam` obtimizer with `learning rate = 0.001`.
- Then I have created a function to train the model. Inside the function, for every `epoch`, I have stored the `training loss`, `training accuracy`, `test loss` and `test accuracy`. I also implemented code to save the best model as well.
- I have trained the model for 10 epochs.

<h3 style="color:cyan">Build VGG-16 Model</h3>

- I have used the pretrained VGG-16 model. I have downloaded by using the method `torchvision.models.vgg16`.
- Then I have freezed all the layers that prevent to update the model's weights and bias.
- But I have replaced the output layer to change the number of nodes (that's 6 for my case). And trained the model for 10 epochs.

<h3 style="color:cyan">Parametric Test: Width vs Depth</h3>

- I also wanted to do a small parametric test. I have choosed that there should be fixed numbers of neurons (80) but the number of layers and the number of nodes per hidden layers will be different.
- So, I choosed these combinations of hidden layers:
    - Case 1 - 1 hidden layer with 80 nodes
    - Case 2 - 5 hidden layers with 16 nodes for each layer.
    - Case 3 - 10 hidden layers with 8 nodes for each layer.
    - Case 4 - 20 hidden layers with 4 nodes for each layer.
- After calculating the number of trainable parameters, the output gives the surpurzing results though the number of nodes of hidden layers are same.
    - Case 1 - Total 14,408,962 parameters.
    - Case 2 - Total 2,889,602 parameters.
    - Case 3 - Total 1,449,106 parameters.
    - Case 4 - Total 728,810 parameters.

---

<h2 style="color:red">Results I Got</h2>

<h3 style="color:cyan">From My Custom Model</h3>

<h3 style="color:cyan">From The VGG-16 Model</h3>

<h3 style="color:cyan">From My Parametric Test: Width vs Depth</h3>
