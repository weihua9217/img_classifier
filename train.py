import torch.nn as nn
import torch
from utils import Adder
from torch import optim
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from dataloader import train_dataloader
from valid import _valid
import os
def _train(model, args):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.fc = nn.Linear(model.fc.in_features, args.class_num)
    model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()
    trainloader = train_dataloader(args.data_dir, args.batch_size)
    loss_adder = Adder()
    max_iter = len(trainloader)
    running_loss = 0.0
    best = -1
    resume_epoch = 0
    if args.resume: 
        state = torch.load(args.resume)
        resume_epoch = state['epoch']
        optimizer.load_state_dict(state['optimizer'])
        model.load_state_dict(state['model'])
        print('Resume from %d'%resume_epoch)
        resume_epoch += 1

    for epoch in range(resume_epoch, args.epoch_num):
        for i, data in enumerate(trainloader):
            
            inputs, labels = data
        
            inputs, labels = inputs.to(device), labels.to(device)
            model.train()
            optimizer.zero_grad()
            output = model(inputs)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()
            loss_adder(loss.item())
            running_loss += loss.item()
            
            if i % args.print_freq == 0:
                writer.add_scalar("Loss",loss_adder.average(),i + (epoch-1)* max_iter)
                print('[EPOCH:%d, Iter: %d] loss: %.5f' %
                    (epoch + 1, i + 1, running_loss/args.print_freq))
                running_loss = 0.0
            
        if epoch % args.val_freq == 0:
            val = _valid(model, args)
            print('(VAL) %03d epoch \n Accuracy %.2f /100' % (epoch, val))
            if val> best:
                torch.save(model.state_dict(),"./results/0511/weights/best.pkl")
        
        if epoch % args.save_freq == 0:
            save_name = os.path.join(args.model_save_dir, 'model_%d.pkl'% epoch)
            torch.save({'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch}, save_name)
    # save the final weight
    save_name = os.path.join(args.model_save_dir, 'final.pkl')
    torch.save({'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch}, save_name)

    print('Finished Training')

