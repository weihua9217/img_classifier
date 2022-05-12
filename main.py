import os
import torch
import argparse
from torchvision import models

from train import _train as train
from test import _test as test


def main(args):
    if not os.path.exists('results/'):
        os.makedirs('results/')
    if not os.path.exists(args.model_save_dir):
        os.makedirs(args.model_save_dir)

    model = models.resnet50(pretrained=True)

    if torch.cuda.is_available():
        model.cuda()
    if args.mode == 'train':
        train(model, args)
    elif args.mode == 'test':
        test(model, args)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='tmp', type=str)
    parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--learning_rate', type=int, default=1e-4)
    parser.add_argument('--print_freq', type=int, default=20) 
    parser.add_argument('--val_freq', type=int, default=50) 
    parser.add_argument('--save_freq', type=int, default=10) 
    parser.add_argument('--epoch_num', type=int, default=1000)
    parser.add_argument('--class_num', type=int, default=10)
    parser.add_argument('--resume', type=str, default='') 
    parser.add_argument('--data_dir', type=str, default="./dataset/") 
    
    # Test Detail
    parser.add_argument('--model', type=str, default='results/tmp/weights/model_10.pkl')

    args = parser.parse_args()
    args.model_save_dir = os.path.join('results/', args.name, 'weights/')

    print(args)
    main(args)