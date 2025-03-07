
from intro_pytorch import *

def main():

    loader = get_data_loader(training=True)
    print(type(loader))
    print(loader.dataset)

main()