from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn
import torch

from Pipeline.option import args
from Data.data import train_loader, test_loader
from Architecture.model import model
from Architecture.optim import optimizer 

criterion = nn.CrossEntropyLoss()

def train(epoch):
    model.train()
    correct = 0
    correct_batch = 0
    for batch_idx, batch in enumerate(train_loader):
        data = batch['image']
        target = batch['label'].view(-1)
        if args.cuda:
            data = data.cuda()
            target =  target.cuda()
        data, target = Variable(data).float(), Variable(target).long()
        optimizer.zero_grad()
        output = model(data)
        
        _, pred = torch.max(output.data, 1)
        correct_batch += pred.eq(target.data).sum()
        correct += pred.eq(target.data).sum()
        # loss = F.nll_loss(output, target)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t Accuracy: {}/{}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0],correct_batch, len(pred)*args.log_interval))
            correct_batch = 0

    print('\nTrain set: Accuracy: {}/{} ({:.0f}%)'.format(
        correct, len(train_loader.dataset),
        100. * correct / len(train_loader.dataset)))
    if epoch % args.save_model_epoch==0:
        torch.save(model.state_dict(), 'model_'+str(epoch)+'.pth')


def test():
    model.eval()
    test_loss = 0
    correct = 0
    for batch in test_loader:
        data = batch['image']
        target = batch['label'].view(-1)
        if args.cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True).float() , Variable(target).long()
        output = model(data)
        test_loss += criterion(output, target).data[0] # sum up batch loss
        # pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        # correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        _, pred = torch.max(output.data, 1)
        correct += pred.eq(target.data).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

