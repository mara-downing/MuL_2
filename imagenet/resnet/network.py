#replace return statement with code to execute input on network and return classification number
import torch
import numpy as np
import math
from torchvision import models as torch_models
from torch.nn import DataParallel
from torchvision import transforms
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from torch.utils.data import DataLoader

#Code for running network and processing imagenet input adapted from https://github.com/fuyuan-zhang/DeepRover

class Model:
    def __init__(self, batch_size, gpu_memory):
        self.batch_size = batch_size
        self.gpu_memory = gpu_memory

    def predict(self, x):
        raise NotImplementedError('custom model should implement this method')

    def objective(self, y, logits):
        """
        the objective function calculates logits(i)-logits(j),
        where i is the correct class, i !=j, and j is the class with largest logits value.
        """
        logits_correct = (logits * y).sum(1, keepdims=True)
        logits_difference = logits_correct - logits
        logits_difference[y] = np.inf 
        objective_val = logits_difference.min(1, keepdims=True)

        return objective_val.flatten()

class CustomModel(Model):
    def __init__(self, model_name, batch_size, gpu_memory):
        super().__init__(batch_size, gpu_memory)
        if model_name in ['imagenet_resnet', 'imagenet_inception']:
            self.mean = np.reshape([0.485, 0.456, 0.406], [1, 3, 1, 1])
            self.std = np.reshape([0.229, 0.224, 0.225], [1, 3, 1, 1])
        self.mean, self.std = self.mean.astype(np.float32), self.std.astype(np.float32)
        if model_name in ['imagenet_resnet', 'imagenet_inception']:
            model = model_definitions_dictionary[model_name](weights="DEFAULT")
            #print("here3")
            model = DataParallel(model.cuda())
            #print("here4")

        model.float()
        model.eval()
        self.model = model
        self.model_name = model_name

    def predict(self, x):
        x = (x - self.mean) / self.std
        x = x.astype(np.float32)

        n_batches = math.ceil(x.shape[0] / self.batch_size)
        logits_list = []
        with torch.no_grad():
            for i in range(n_batches):
                x_batch = x[i * self.batch_size:(i + 1) * self.batch_size]
                x_batch_torch = torch.as_tensor(x_batch, device=torch.device('cuda'))
                logits = self.model(x_batch_torch).cpu().numpy()
                logits_list.append(logits)
        logits = np.vstack(logits_list)
        return logits

model_definitions_dictionary = {
                     'imagenet_resnet': torch_models.resnet50,
                     'imagenet_inception': torch_models.inception_v3,
                    }

print("here1")
model = CustomModel("imagenet_resnet", 1, 0.99)
print("here2")


def run_network(inputfeatures):
    newinputfeatures = np.array(inputfeatures[:])
    newinputfeatures = newinputfeatures.reshape(3, 224, 224)
    logits_clean = model.predict([newinputfeatures])
    result = logits_clean.argmax(1)
    #print(type(result))
    #print(result.shape)
    #print(result[0])
    return result[0]

#tempinput = [0]*150528
#run_network(tempinput)

