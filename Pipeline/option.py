import torch
import argparse

parser = argparse.ArgumentParser(description='AML Project')

parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                    help='input batch size for testing (default: 64)')

parser.add_argument('--optimizer',default='Adam', metavar='OPTM',
                    help='define optimizer (default: Adam)')

parser.add_argument('--epochs', type=int, default=100, metavar='N',
                    help='number of epochs to train (default: 100)')

parser.add_argument('--lr', type=float, default=0.0005, metavar='LR',
                    help='learning rate (default: 0.0005)')

parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')

parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')

parser.add_argument('--log-interval', type=int, default=5, metavar='N',
                    help='how many batches to wait before logging training status')

parser.add_argument('--train-percent', type=float, default=0.7, metavar='T',
                    help='percetage of the dataset to be used for training')

parser.add_argument('--save_model_epoch', type=int, default=5, metavar='N',
                    help='how many epochs to wait before saving model')

parser.add_argument('--load', type=int, default=0, metavar='L',
                    help='load pretrained')

parser.add_argument('--load-path', default='./models/Inception_bs4e80', metavar='LP',
                    help='pretrained path')

parser.add_argument('--model', default='Custom', metavar='M',
                    help='network model')

parser.add_argument('--aug', type=int, default=1, metavar='A',
                    help='use augmented (with noise)?')



args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
