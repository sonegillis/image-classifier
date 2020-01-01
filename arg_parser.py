import argparse

def get_train_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("data_dir", help="Directory where datasets are located")
    ap.add_argument("--save_dir", help="Directory where checkpoint.pth would be saved")
    ap.add_argument("--arch", help="Architechture used for training the network e.g vgg16")
    ap.add_argument("--learning_rate", help="Learning rate used in back propagation")
    ap.add_argument("--hidden_units", help="hidden units for network. Can be an integer or an array")
    ap.add_argument("--epochs", help="Number of epochs")
    ap.add_argument("--gpu", help="Indicate whether to use gpu for training")
    
    args = vars(ap.parse_args())
    return args

def get_predict_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="Path to image")
    ap.add_argument("checkpoint", help="Path to the saved checkpoint")
    ap.add_argument("--top_k")
    ap.add_argument("--category_names", help="Path to the json file containing category names")
    ap.add_argument("--gpu", help="Indicate whether to use gpu for training")
    
    args = vars(ap.parse_args())
    return args