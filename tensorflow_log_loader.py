import argparse
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt

def plot_tensorflow_log(args):

    # Loading too much data is slow...
    tf_size_guidance = {'scalars': args.num_load}

    event_acc = EventAccumulator(args.log_name, tf_size_guidance)
    event_acc.Reload()
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)

    # Show all tags in the log file
    #print(event_acc.Tags())
    if event_acc.Tags()["scalars"][0] == "loss":
        loss = event_acc.Scalars('loss')
        
        steps = len(loss)
        x = np.zeros([steps, 1])
        y = np.zeros([steps, 1])
        for i in range(steps):
            x[i] = loss[i][1] #step
            y[i] = loss[i][2] # value
        ax1.plot(x, y[:], label='loss_auto')
    else:
        loss_content = event_acc.Scalars('loss_content')
        loss_style = event_acc.Scalars('loss_style')

        steps = len(loss_content)
        x = np.zeros([steps, 1])#np.arange(steps)
        y = np.zeros([steps, 2])

        for i in range(steps):
            x[i] = loss_content[i][1] #step
            y[i, 0] = loss_content[i][2] # value
            y[i, 1] = loss_style[i][2]

        ax1.plot(x, y[:,0], label='loss_content')
        ax1.plot(x, y[:,1], label='loss_style')

    ax1.set_xlabel("Iter")
    ax1.set_ylabel("loss")
    ax1.set_title("Training Progress")
    ax1.legend(loc='upper right', frameon=True)
    ax1.set_yscale("log")
    if args.save:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        plt.savefig('{:s}/losses_{:s}.eps'.format(args.save_dir, args.log_name[-17:]))
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs' , help='Log dir')
    parser.add_argument('--log_name', type=str, default=None ,help='Log name')
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--save_dir', type=str, default='./plots',
			help='Dir for save plot, default ./plots')
    parser.add_argument('--num_load', type=int, default=160000)
    args = parser.parse_args()
    
    if args.log_name == None:
        logs = [os.path.join(args.log_dir, f) for f in os.listdir(args.log_dir)]
        timestamps = [os.path.getmtime(log) for log in logs]
        recent_idx = np.argmax(timestamps)
        log_name = logs[recent_idx]
        args.log_name = log_name

plot_tensorflow_log(args)
