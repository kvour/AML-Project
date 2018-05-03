import time, datetime
import numpy as np

from Pipeline.option import args
from Pipeline.run import train, test

start_time = datetime.datetime.now().replace(microsecond=0)
print('\n---Started training at---', (start_time))

train_acc = np.zeros([args.epochs,2])
test_acc = np.zeros([args.epochs,2])

for epoch in range(1, args.epochs + 1):
    tr_corr , tr_per = train(epoch)
    ts_corr , ts_per = test()
    train_acc[epoch-1,0] = tr_corr
    train_acc[epoch-1,1] = tr_per
    test_acc[epoch-1,0] = ts_corr
    test_acc[epoch-1,1] = ts_per
    current_time = datetime.datetime.now().replace(microsecond=0)
    print('Time Interval:', current_time - start_time, '\n')

    if args.aug == 0:
        np.save('train_acc_'+args.model+'_bs'+str(args.batch_size)+'.npy',train_acc)
        np.save('test_acc_'+args.model+'_bs'+str(args.batch_size)+'.npy',test_acc)
    else:
        np.save('train_acc_'+args.model+'_bs'+str(args.batch_size)+'_aug'+'.npy',train_acc)
        np.save('test_acc_'+args.model+'_bs'+str(args.batch_size)+'_aug'+'.npy',test_acc)
