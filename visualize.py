'''
Visualization script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 3rd April 2017

Update:
1) Added option for with background
2) Trying to add option to see only the ped IDs which are at the 8th time step == pedIDs whose trajectories are being predicted
'''


import numpy as np
import matplotlib.pyplot as plt
import pickle
from torch.autograd import Variable
import argparse
import seaborn
import os

def plot_trajectories(true_trajs, pred_trajs, nodesPresent, obs_length, name, plot_directory, args):
    '''
    Parameters
    ==========

    true_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the true trajectories of the nodes

    pred_trajs : Numpy matrix of shape seq_length x numNodes x 2
    Contains the predicted trajectories of the nodes

    nodesPresent : A list of lists, of size seq_length
    Each list contains the nodeIDs present at that time-step

    obs_length : Length of observed trajectory

    name : Name of the plot

    withBackground : Include background or not
    '''

    traj_length, numNodes, _ = true_trajs.shape
    #Initialize figure
    #Load the background
    im = plt.imread('./maps/0.png')
    if args.with_background:
       implot = plt.imshow(im,extent=[-4,4,-4,4])

    width_true = im.shape[0]
    height_true = im.shape[1]

    # if withBackground:
    #    width = width_true
    #    height = height_true
    # else:
    #width = 1
    #height = 1

    traj_data = {}
    for tstep in range(traj_length):
        pred_pos = pred_trajs[tstep, :]
        true_pos = true_trajs[tstep, :]

        pred_pos = pred_pos[:,[1,0]]    # since y axis is shown before x
        true_pos = true_pos[:,[1,0]]

        for ped in range(numNodes):
            if (ped in nodesPresent[7]) or args.show_all:
                if ped not in traj_data and tstep < obs_length:
                    traj_data[ped] = [[], []]

                if ped in nodesPresent[tstep]:                
                    traj_data[ped][0].append(true_pos[ped, :])
                    traj_data[ped][1].append(pred_pos[ped, :])

    for j in traj_data:
        c = np.random.rand(3)
        true_traj_ped = traj_data[j][0]  # List of [x,y] elements
        pred_traj_ped = traj_data[j][1]

        true_x = [p[0] for p in true_traj_ped] #[(p[0]+1)/2*height for p in true_traj_ped]
        true_y = [p[1] for p in true_traj_ped] #[(p[1]+1)/2*width for p in true_traj_ped]
        pred_x = [p[0] for p in pred_traj_ped] #[(p[0]+1)/2*height for p in pred_traj_ped]
        pred_y = [p[1] for p in pred_traj_ped] #[(p[1]+1)/2*width for p in pred_traj_ped]

        plt.plot(true_x, true_y, color=c, linestyle='solid', linewidth=2, marker='o')
        plt.plot(pred_x, pred_y, color=c, linestyle='dashed', linewidth=2, marker='+')

    #if not withBackground:
    #    plt.ylim((1, 0))
    #    plt.xlim((0, 1))

    #plt.show()

    #plt.xlim(-3.1, 3.1)
    #plt.ylim(-3.1, 3.1)

    plt.show()
    plt.close('all')
    plt.savefig(plot_directory+'/'+name+'.png')

    plt.gcf().clear()
    
    plt.clf()


def main():
    parser = argparse.ArgumentParser()

    # Train Dataset
    # Use like:
    # python transpose_inrange.py --train_dataset index_1 index_2 ...
    parser.add_argument('-l','--train_dataset', nargs='+', help='<Required> training dataset(s) the model is trained on: --train_dataset index_1 index_2 ...', default=[0,1,2,4], type=int)    

    parser.add_argument('--test_dataset', type=int, default=3,
                        help='test dataset index')
    parser.add_argument('--with_background', default = False, action = 'store_true')
    parser.add_argument('--show_all', default = False, action = 'store_true', help = 'Show the peds for whon prediction isn\'t posssible becuase they aren\'t present at last observed step')

    # Parse the parameters
    args = parser.parse_args()

    # Save directory
    save_directory = 'save/'
    save_directory += 'trainedOn_'+str(args.train_dataset) + '/testedOn_' + str(args.test_dataset)
    plot_directory = 'plot/trainedOn_'+str(args.train_dataset) + '/testedOn_' + str(args.test_dataset)

    if args.with_background:
        plot_directory+='with_background'

    if not os.path.exists(plot_directory):
        os.makedirs(plot_directory)

    f = open(save_directory+'/results.pkl', 'rb')
    results = pickle.load(f)

    # print "Enter 0 (or) 1 for without/with background"
    # withBackground = int(input())

    print('Plotting and saving in '+plot_directory)
    for i in range(len(results)):
        print('Sequence', i)
        name = 'sequence' + str(i)
        ## results[i], each i for (1 batch = 1 seq) in test time because batch_size is 1
        ## true_trajs, pred_trajs, nodesPresent, obs_length, name, plot_directory, with_background
        plot_trajectories(results[i][0], results[i][1], results[i][2], results[i][3], name, plot_directory, args)  


if __name__ == '__main__':
    main()
