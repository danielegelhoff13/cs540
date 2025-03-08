
from intro_pytorch import *

def main():

    # get data loaders for FashionMNSIT dataset
    train_loader = get_data_loader(training=True)
    test_loader = get_data_loader(training=False)

    # build, train, and test regular model
    model = build_model();
    train_model(model, train_loader, nn.CrossEntropyLoss, 5)
    evaluate_model(model, test_loader, criterion=nn.CrossEntropyLoss, show_loss=True)

    # build, train and test deeper model
    # deep_model = build_deeper_model();
    # train_model(deep_model, train_loader, nn.CrossEntropyLoss, 10)
    # evaluate_model(deep_model, test_loader, criterion=nn.CrossEntropyLoss, show_loss=True)

    # predict a label for a specific test image
    test_images = next(iter(test_loader))[0]
    predict_label(model=model, test_images=test_images, index=1)


main()