
import torch
from torch.utils.data import dataloader
from dataloader import test_dataloader
from utils import Adder
import torch.nn as nn

def _test(model, args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    testloader = test_dataloader(args.data_dir, batch_size=1)
    model_state_dict = torch.load(args.model)
    model.load_state_dict(model_state_dict)
    
    loss_adder = Adder()
    total = 0
    correct = 0
    for i, data in enumerate(testloader):
        input_img, label = data

        input_img, label = input_img.to(device), label.to(device)
        
        # predict
        predict_label = model(input_img)

        # accuracy
        _, predicted = torch.max(predict_label, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()
        print(predicted,label)
        # cal loss
        loss = criterion(predict_label, label)
        loss_adder(loss.item())
        
    print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))