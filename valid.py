
import torch
from torch.utils.data import dataloader
from dataloader import valid_dataloader
from utils import Adder
import torch.nn as nn

def _valid(model, args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    criterion = nn.CrossEntropyLoss()
    validloader = valid_dataloader(args.data_dir, batch_size=16)

    loss_adder = Adder()
    total = 0
    correct = 0
    for i, data in enumerate(validloader):
        input_img, label = data[0].to(device), data[1].to(device)
        input_img = input_img.to(device)
        # predict
        predict_label = model(input_img)
        # accuracy
        _, predicted = torch.max(predict_label, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

        # cal loss
        loss = criterion(predict_label, label)
        loss_adder(loss.item())
        
    
    return (100 * correct / total)