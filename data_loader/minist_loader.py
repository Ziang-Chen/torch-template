from numpy import minimum
from dataloader import DataLoader,Stream
from torchvision.datasets import mnist

basic_path='../dataset/MNIST/'
train_image_path=basic_path+'train-images-idx3-ubyte'
train_label_path=basic_path+'train-labels-idx1-ubyte'

test_image_path=basic_path+'t10k-images-idx3-ubyte'
test_label_path=basic_path+'t10k-labels-idx1-ubyte'

train_x=DataLoader(mnist.read_image_file,train_image_path)
train_y=DataLoader(mnist.read_label_file,train_label_path)

test_x=DataLoader(mnist.read_image_file,test_image_path)
test_y=DataLoader(mnist.read_label_file,test_label_path)

train_steam=Stream(train_x,train_y)
test_stream=Stream(test_x,test_y)



