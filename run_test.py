import torch
from torch.autograd import Variable
from torchvision import transforms
import torch.nn as nn
import numpy as np
from skimage import io
import torchvision.transforms.functional as F
import torch.nn.functional as f
from PIL import Image
import ardrone
import time
import pygame

from Pipeline.option import args
from Architecture.model import model
# from Data.data import trans

drone = ardrone.ARDrone()
drone.speed = 0.1

pygame.init()
W, H = 299,299
screen = pygame.display.set_mode((W,H))


imsize=299
threshold = 0.6


trans = transforms.Compose([transforms.Resize(imsize),transforms.CenterCrop(imsize), transforms.ToTensor()])

# model.load_state_dict(torch.load('Inception_bs16e85.pth'))
model.eval()

def image_loader():
    """load image, returns cuda tensor"""
    image = drone.image
    image = trans(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

drone.trim()
# drone.reset()
time.sleep(2)

drone.takeoff()
time.sleep(4)

temp = 1

while temp == 1:
    drone.speed = 0.08
    pygame.display.flip()
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                drone.land()
                time.sleep(1)
                # drone.reset
                temp = 0
    image = image_loader()
    output = model(image)
    output = f.softmax(output,1)
    val, ind = output.transpose(0,1).max(0)

    if (val.data[0]<threshold):
        drone.hover()
        print('hover')
    elif ind.data[0] == 0 :
        print('Class: Forward with Prob: {} \n'.format(val.data[0]))
        drone.move_forward()
    elif ind.data[0] == 1 :
        print('Class: Left with Prob: {} \n'.format(val.data[0]))
        drone.move_left()
    elif ind.data[0] == 2 :
        print('Class: Right with Prob: {} \n'.format(val.data[0]))
        drone.move_right()
    elif ind.data[0] == 3 :
        print('Class: RotLeft with Prob: {} \n'.format(val.data[0]))
        drone.speed = 0.8
        drone.turn_left()
    elif ind.data[0] == 4 :
        print('Class: RotRight with Prob: {} \n'.format(val.data[0]))
        drone.speed = 0.8
        drone.turn_right()











