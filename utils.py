import numpy as np
from PIL import Image
class Toolkits:
    def __init__(self):
        pass
    
    @classmethod
    def process_image():
        im = Image.open(image)
        transform = transforms.Compose([
           transforms.Resize(256),
           transforms.CenterCrop(224),
           transforms.ToTensor(),
           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        return transform(im)
    
    @classmethod
    def initialiseTrainArgs(cls, args):
        # Define default hyper parameters
        learning_rate = 0.0005
        hidden_units = [2048, 512, 256]
        epochs = 10

        gpu = True
        arch = "vgg16"

        save_dir = None
        data_dir = ""
        
        if args['data_dir'] is not None:
            data_dir = args['data_dir']
        if args['save_dir'] is not None:
            save_dir = args['save_dir']
        if args['learning_rate'] is not None:
            learning_rate = float(args['learning_rate'])
        if args['hidden_units'] is not None:
            hidden_units = args['hidden_units']
            if isinstance(hidden_units, str):
                hidden_units = [hidden_units]
            hidden_units = list(map(int, hidden_units))
        if args['epochs'] is not None:
            epochs = int(args['epochs'])
        if args['gpu'] is not None:
            gpu = args['gpu']
        if args['arch'] is not None:
            arch = args['arch']

        return learning_rate, hidden_units, epochs, gpu, arch, save_dir, data_dir
    
    @classmethod
    def initialisePredictArgs(cls, args):
        # Define default hyper parameters
        inputs = ""
        checkpoint = "checkpoint.pth"
        top_k = 5
        gpu = False
        category_names = ""
        
        if args['checkpoint'] is not None:
            checkpoint = args['checkpoint']
        if args['input'] is not None:
            input = args['input']
        if args['gpu'] is not None:
            gpu = args['gpu']
        if args['category_names'] is not None:
            category_names = args['category_names']
        if args['top_k'] is not None:
            top_k = args['top_k']

        return inputs, checkpoint, top_k, gpu, category_names