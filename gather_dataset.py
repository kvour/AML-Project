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


drone = ardrone.ARDrone()
drone.speed = 0.1

pygame.init()
W, H = 299,299
screen = pygame.display.set_mode((W,H))


imsize=299
threshold = 0.8


trans = transforms.Compose([transforms.Resize(imsize),transforms.CenterCrop(imsize), transforms.ToTensor()])

# model.load_state_dict(torch.load('Inception_bs16e85.pth'))
model.eval()

def image_loader(frame):
    """load image, returns cuda tensor"""
    image = drone.image
    image.save('./images/frame'+str(frame)+'.png',"PNG")
    image = trans(image).float()
    image = Variable(image, requires_grad=True)
    image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
    return image.cuda()  #assumes that you're using GPU

drone.reset()
time.sleep(2)

drone.takeoff()
time.sleep(4)

temp = 1
frame = 0
cmd_type = 'none'
file = open('log.txt','w')

while temp == 1:
    pygame.display.flip()
    file.write(cmd_type+' '+str(frame)+'\r\n')
    events = pygame.event.get()
    for event in events:
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                drone.land()
                time.sleep(3)
                temp = 0
            if event.key == pygame.K_1:
                manual = 1
                speed = 0.2
                while manual == 1:
                    frame = frame + 1
                    drone.image.save('./images/frame'+str(frame)+'.png',"PNG")
                    pygame.display.flip()
                    events = pygame.event.get()
                    # print("in 1 \n")
                    for event in events:
                        if event.type == pygame.KEYUP:
                            drone.hover()
                            cmd_type = 'hover'
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                drone.land()
                                cmd_type = 'land'
                            # forward / backward
                            elif event.key == pygame.K_w:
                                drone.move_forward()
                                cmd_type = 'move_forward'
                                # file.write(cmd_type+'\r\n')
                            elif event.key == pygame.K_s:
                                drone.move_backward()
                                cmd_type = 'move_backward'
                                # file.write(cmd_type+'\r\n')
                            # left / right
                            elif event.key == pygame.K_a:
                                drone.move_left()
                                cmd_type = 'move_left'
                                # file.write(cmd_type+'\r\n')
                            elif event.key == pygame.K_d:
                                drone.move_right()
                                cmd_type = 'move_right'
                                # file.write(cmd_type+'\r\n')
                            # up / down
                            elif event.key == pygame.K_q:
                                drone.turn_left()
                                cmd_type = 'turn_left'
                                # file.write(cmd_type+'\r\n')
                            elif event.key == pygame.K_e:
                                drone.turn_right()
                                cmd_type = 'turn_right'
                                # file.write(cmd_type+'\r\n')
                            elif event.key == pygame.K_2:
                                manual = 0
                                speed = 0.1
                    file.write(cmd_type+' '+str(frame)+'\r\n')
    frame = frame + 1
    image = image_loader(frame)
    output = model(image)
    output = f.softmax(output,1)
    val, ind = output.transpose(0,1).max(0)

    if (val.data[0]<threshold):
        drone.hover()
        print('hover')
        cmd_type = 'hover_network'
    elif ind.data[0] == 0 :
        print('Class: Forward with Prob: {} \n'.format(val.data[0]))
        drone.move_forward()
        cmd_type = 'move_forward_network'
    elif ind.data[0] == 1 :
        print('Class: Left with Prob: {} \n'.format(val.data[0]))
        drone.move_left()
        cmd_type = 'move_left_network'
    elif ind.data[0] == 2 :
        print('Class: Right with Prob: {} \n'.format(val.data[0]))
        drone.move_right()
        cmd_type = 'move_right_network'
    elif ind.data[0] == 3 :
        print('Class: RotLeft with Prob: {} \n'.format(val.data[0]))
        drone.turn_left()
        cmd_type = 'turn_left_network'
    elif ind.data[0] == 4 :
        print('Class: RotRight with Prob: {} \n'.format(val.data[0]))
        drone.turn_right()
        cmd_type = 'turn_right_network'











