import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

def get_data_loader(training = True):
    """
    INPUT: An optional boolean argument (default value is True for training dataset)
    RETURNS: Dataloader for the training set (if training = True) or the test set (if training = False)
    """
    custom_transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    # load dataset with torchvision
    if training:
        dataset = datasets.FashionMNIST('./data', train=True, download=True, transform=custom_transform)
    else:
        dataset = datasets.FashionMNIST('./data', train=False, download=True, transform=custom_transform)
    # retrieve images and labels from dataset object, shuffle if training set
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=training)
    return loader

def build_model():
    """
    INPUT: None
    RETURNS: An untrained neural network model
    """
    # model creation statement
    model = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(in_features=784, out_features=128), # dense layer
        nn.ReLU(), 
        nn.Linear(in_features=128, out_features=64), # dense layer
        nn.ReLU(), 
        nn.Linear(in_features=64, out_features=10) # dense layer (output)
    )
    return model

def build_deeper_model():
    """
    INPUT: None
    RETURNS: An untrained neural network model
    """
    model = nn.Sequential(
        nn.Flatten(), 
        nn.Linear(in_features=784, out_features=256), # dense layer
        nn.ReLU(), 
        nn.Linear(in_features=256, out_features=128), # dense layer
        nn.ReLU(), 
        nn.Linear(in_features=128, out_features=64), # dense layer
        nn.ReLU(), 
        nn.Linear(in_features=64, out_features=32), # dense layer
        nn.ReLU(), 
        nn.Linear(in_features=32, out_features=10) # dense layer OUTPUT
    )
    return model


def train_model(model, train_loader, criterion, T):
    """
    INPUT: 
        model - the model produced by the previous function
        train_loader  - the train DataLoader produced by the first function
        criterion   - cross-entropy 
        T - number of epochs for training
    RETURNS: None
    """
    # optimazation algortihm: stochastic gradient descent (SGD)
    opt = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    model.train() # set "mode" of model to training
    for epoch in range(T):
        running_loss = 0.0 # track loss for this epoch
        running_correct = 0.0 # calculate accuracy for this epoch (correct / 60000)
        num_batches = 0; # used to average loss across all batches each epoch
        for data in train_loader:
            # get inputs
            inputs, labels = data
            # zero gradients
            opt.zero_grad();
            # feed inputs to model to generate outputs
            outputs = model(inputs)
            # calculate loss
            loss = criterion()(outputs, labels)
            loss.backward()
            opt.step()

            # get predicted class -> determines if model predicted correct label
            _, predicted = torch.max(outputs, dim=1)

            # update loss, total, correct
            running_loss += loss.item()
            running_correct += (predicted == labels).sum().item()

            # update batch counter
            num_batches += 1

        # print stats for each epoch
        accuracy = 100*(running_correct/60000)
        e_loss = running_loss/num_batches # not all batches equal size
        print('Train Epoch:', epoch, '\t', end='  ')
        print('Accuracy: ', int(running_correct),'/',60000,'(',f"{accuracy:.2f}", '%)', sep='', end='  ')
        print('Loss:', f"{e_loss:.3f}")


def evaluate_model(model, test_loader, criterion, show_loss = True):
    """
    INPUT: 
        model - the the trained model produced by the previous function
        test_loader    - the test DataLoader
        criterion   - cross-entropy 
    RETURNS: None
    """
    model.eval()
    correct = 0 # to calculate accuracy of test
    total_loss = 0 # accumulate losses of each batch
    num_samples = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data
            outputs = model(inputs)
            # number of correct labelings achieved by model
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            # calculate loss: weighted by batch size
            loss = criterion()(outputs, labels)
            total_loss += loss.item() * labels.size(0)
            num_samples += labels.size(0)

    # print stats
    if show_loss:
        avg_loss = total_loss / num_samples
        print('Average loss:', f"{avg_loss:.4f}")
    accuracy = 100*(correct/num_samples)
    print('Accuracy: ', f"{accuracy:.2f}", '%', sep='')

def predict_label(model, test_images, index):
    """
    INPUT: 
        model - the trained model
        test_images   -  a tensor. test image set of shape Nx1x28x28
        index   -  specific index  i of the image to be tested: 0 <= i <= N - 1
    RETURNS: None
    """
    class_names = ['T-shirt/top','Trouser','Pullover','Dress','Coat','Sandal','Shirt','Sneaker','Bag','Ankle Boot']
    logits = model(test_images)
    prob = F.softmax(logits, dim=1) # prob[index] = list of probablities
    top_3_probs = sorted(zip(prob[index], class_names), reverse=True)[:3]
    for label in top_3_probs:
        p = 100*label[0].item()
        print(label[1], ': ', f"{p:.2f}", '%', sep='')