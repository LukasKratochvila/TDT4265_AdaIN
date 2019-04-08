import argparse
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt

def plot_tensorflow_log(path, save_dir, auto_loss):

    # Loading too much data is slow...
    #tf_size_guidance = {
    #    'compressedHistograms': 10,
    #    'images': 0,
    #    'scalars': 100,
    #    'histograms': 1
    #}

    event_acc = EventAccumulator(path)#, tf_size_guidance)
    event_acc.Reload()

    # Show all tags in the log file
    #print(event_acc.Tags())
    if auto_loss:
        loss = event_acc.Scalars('loss')
        
        steps = len(loss)
        x = np.zeros([steps, 1])
        y = np.zeros([steps, 1])
        for i in range(steps):
            x[i] = loss[i][1] #step
            y[i] = loss[i][2] # value
        plt.plot(x, y[:], label='loss_auto')
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

        plt.plot(x, y[:,0], label='loss_content')
        plt.plot(x, y[:,1], label='loss_style')

    plt.xlabel("Iter")
    plt.ylabel("loss")
    plt.title("Training Progress")
    plt.legend(loc='upper right', frameon=True)
    #plt.savefig('{:s}/losses_{:s}.eps'.format(save_dir, path[-17:]))
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_name', type=str, required=True,
			help='Log name')
    parser.add_argument('--save_dir', type=str, default="./plots",
			help='Dir for save plot, default ./plots')
    parser.add_argument('--auto_loss', type=bool, default=False)
    log_file = parser.parse_args().log_name
    save_dir = parser.parse_args().save_dir
    auto_loss = parser.parse_args().auto_loss
    if not os.path.exists(save_dir):
    	os.mkdir(save_dir)
    #print(log_file)
plot_tensorflow_log(log_file, save_dir,auto_loss)
