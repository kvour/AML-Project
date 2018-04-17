import torch.optim as optim

from Pipeline.option import args
from Architecture.model import model

optimizer = None
if args.optimizer =='SGD':
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.1)
elif args.optimizer =='Adam':
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
elif args.optimizer =='Adadelta':
    optimizer = optim.Adadelta(model.parameters())
else:
    raise ValueError('Wrong name of optimizer')
    
print('\n---Training Details---')
print('batch size:',args.batch_size)
print('seed number', args.seed)

print('\n---Optimization Information---')
print('optimizer:', args.optimizer)
if args.optimizer != 'Adadelta':
	print('lr:', args.lr)
