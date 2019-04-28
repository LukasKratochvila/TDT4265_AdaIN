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
    
    assert event_acc.Tags()["scalars"] != [], "Did you give the log file?"
    
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(1, 1, 1)

    # Show all tags in the log file
    #print(event_acc.Tags())
    if event_acc.Tags()["scalars"][0] == "loss":
        loss = event_acc.Scalars('loss')
        
        steps = len(loss)
        x = np.zeros(steps,)
        y = np.zeros(steps,)
        for i in range(steps):
            x[i] = loss[i][1] #step
            y[i] = loss[i][2] # value
        ax1.plot(x, y[:], label='loss_auto', color="tab:orange")
        if args.trend:
            if not args.linear:
                y_r=np.log(y)
                y_trend = np.exp(np.poly1d(np.polyfit(x,y_r[:,0],5))(x))
            else:
                y_trend = np.poly1d(np.polyfit(x,y[:,0],5))(x)

            ax1.plot(x, y_trend, label='loss_trend', color="red")
        print("Final auto loss: %.3f "%np.mean(y[-100:]))
    else:
        loss_content = event_acc.Scalars('loss_content')
        loss_style = event_acc.Scalars('loss_style')

        steps = len(loss_content)
        x = np.zeros(steps, )
        y = np.zeros([steps,2])

        for i in range(steps):
            x[i] = loss_content[i][1] #step
            y[i, 0] = loss_style[i][2]
            y[i, 1] = loss_content[i][2] # value

        ax1.plot(x, y[:,0], label='loss_style', color="tab:orange")
        ax1.plot(x, y[:,1], label='loss_content', color="tab:blue")
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

            ax1.plot(x, y_trend[:, 0], label='loss_style_trend', color="red")
            ax1.plot(x, y_trend[:, 1], label='loss_content_trend', color="blue")
        print("Final content loss: %.3f style loss: %.3f"%(np.mean(y[-100:,1]),np.mean(y[-100:,1])))

    ax1.set_xlabel("Iter")
    ax1.set_ylabel("loss")
    ax1.set_title("Training Progress")
    ax1.legend(loc='upper right', frameon=True)
    if not args.linear:
        ax1.set_yscale("log")
    ax1.set_ylim([0.5,5*10**2])
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
    parser.add_argument('--save', action='store_true')
    parser.add_argument('--linear', action='store_true', help='Linear scale [default: logaritmic]')
    parser.add_argument('--trend', action='store_false', help='disable plotting trend')
    parser.add_argument('--save_dir', type=str, default='./plots',
			help='Dir for save plot, default ./plots')
    parser.add_argument('--num_load', type=int, default=80000, help='Number of loaded data samples [default: 80000]')
    args = parser.parse_args()
    
    if args.log_name == None:
        logs = [os.path.join(args.log_dir, f) for f in os.listdir(args.log_dir)]
        timestamps = [os.path.getmtime(log) for log in logs]
        recent_idx = np.argmax(timestamps)
        log_name = logs[recent_idx]
        args.log_name = log_name

plot_tensorflow_log(args)
