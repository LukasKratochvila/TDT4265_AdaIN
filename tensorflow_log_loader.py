import argparse
import os
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

import matplotlib.pyplot as plt

def plot_tensorflow_log(args):
    """
    Plot data from tensorboard event file.
    """

    # Loading too much data is slow...
    tf_size_guidance = {'scalars': args.num_load}

    event_acc = EventAccumulator(args.log_name, tf_size_guidance)
    event_acc.Reload()
    
    assert event_acc.Tags()["scalars"] != [], "Did you give the log file?"
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)

    loss_content = event_acc.Scalars('loss_content')
    loss_style = event_acc.Scalars('loss_style')

    steps = args.num_print if args.num_print != 0 else len(loss_content)
    x = np.zeros(steps, )
    y = np.zeros([steps,2])

    for i in range(steps):
        x[i] = loss_content[i][1] #step
        y[i, 0] = loss_style[i][2]
        y[i, 1] = loss_content[i][2] # value

    if args.trend:
        y_trend = np.zeros([steps,2])
        for i in range(steps):
            if i < 1000:
                y_trend[i, 0] = np.mean(y[:i+1, 0])
                y_trend[i, 1] = np.mean(y[:i+1, 1])
            elif i > steps-1000:
                y_trend[i, 0] = np.mean(y[i-1000:, 0])
                y_trend[i, 1] = np.mean(y[i-1000:, 1])
            else:
                y_trend[i, 0] = np.mean(y[i-1000:i+1000, 0])
                y_trend[i, 1] = np.mean(y[i-1000:i+1000, 1])
                
    ax1.plot(x, y[:,0], label='loss_style', color="tab:orange")
    ax1.plot(x, y[:,1], label='loss_content', color="tab:blue")
    ax1.plot(x, y_trend[:, 0], label='loss_style_trend', color="red")
    ax1.plot(x, y_trend[:, 1], label='loss_content_trend', color="blue")
    print("Final content loss: %.3f style loss: %.3f"%(np.mean(y[-1000:,0]),np.mean(y[-1000:,1])))

    ax1.set_xlabel("Iter")
    ax1.set_ylabel("loss")
    ax1.set_ylim([0.1,1000])
    ax1.set_title("Training Progress")
    ax1.legend(loc='upper right', frameon=True)
    if not args.linear:
        ax1.set_yscale("log")
    if args.save:
        if not os.path.exists(args.save_dir):
            os.mkdir(args.save_dir)
        plt.savefig('{:s}/losses_{:s}.eps'.format(args.save_dir, args.log_name[-17:]))
    else:
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--log_dir', type=str, default='./logs', help='Log dir')
    parser.add_argument('--log_name', type=str, default=None, help='Log name')
    parser.add_argument('--save', action='store_true', help='Save in to file - default only plot')
    parser.add_argument('--linear', action='store_true', help='Linear scale [default: logaritmic]')
    parser.add_argument('--trend', action='store_false', help='disable plotting trend')
    parser.add_argument('--save_dir', type=str, default='./plots', help='Dir for save plot, default ./plots')
    parser.add_argument('--num_print', type=int, default=0, help='Number of print data samples [default: all]')
    parser.add_argument('--num_load', type=int, default=80000, help='Number of loaded data samples [default: 80000]')
    args = parser.parse_args()

    # If not specify log_name find the newest file in log_dir
    if args.log_name == None:
        logs = [os.path.join(args.log_dir, f) for f in os.listdir(args.log_dir)]
        timestamps = [os.path.getmtime(log) for log in logs]
        recent_idx = np.argmax(timestamps)
        log_name = logs[recent_idx]
        args.log_name = log_name

plot_tensorflow_log(args)
