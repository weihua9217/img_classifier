
import torch
from torch.utils.data import dataloader
from dataloader import test_dataloader
from utils import Adder
import torch.nn as nn
import numpy as np
def _test(model, args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.fc = nn.Linear(model.fc.in_features, args.class_num)
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    testloader = test_dataloader(args.data_dir, batch_size=1)
    
    state = torch.load(args.model)
    model.load_state_dict(state['model'])
    
    loss_adder = Adder()
    total = 0
    correct = 0
    
    count = np.zeros(args.class_num)
    corr = np.zeros(args.class_num)

    for i, data in enumerate(testloader):
        input_img, label = data

        input_img, label = input_img.to(device), label.to(device)
        
        # predict
        predict_label = model(input_img)

        # accuracy
        _, predicted = torch.max(predict_label, 1)
        total += label.size(0)
        count[label] = count[label]+ label.size(0)
        
        correct += (predicted == label).sum().item()
        corr[label] = corr[label] +(predicted == label).sum().item()

        print(predicted,label)
        # cal loss
        loss = criterion(predict_label, label)
        loss_adder(loss.item())
        
    print('Accuracy of the network on the test images: %d %%' % (
    100 * correct / total))
    print(count,corr)
    print('each category accuracy:')
    for i in range(len(corr)):
        print("Class:%d , Accuracy: %.2f " %(i, float(100*corr[i]/count[i]))+r'%')