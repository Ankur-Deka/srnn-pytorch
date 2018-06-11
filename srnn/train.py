'''
Train script for the structural RNN model
introduced in https://arxiv.org/abs/1511.05298
Author : Anirudh Vemula
Date : 29th March 2017
'''

import argparse
import os
import pickle
import time

import torch
from torch.autograd import Variable

from utils import DataLoader
from st_graph import ST_GRAPH
from model import SRNN
from criterion import Gaussian2DLikelihood

import numpy as np

def main():
    parser = argparse.ArgumentParser()

    # RNN size
    parser.add_argument('--human_node_rnn_size', type=int, default=128,
                        help='Size of Human Node RNN hidden state')
    parser.add_argument('--human_human_edge_rnn_size', type=int, default=256,
                        help='Size of Human Human Edge RNN hidden state')

    # Input and output size
    parser.add_argument('--human_node_input_size', type=int, default=2,
                        help='Dimension of the node features')
    parser.add_argument('--human_human_edge_input_size', type=int, default=2,
                        help='Dimension of the edge features')
    parser.add_argument('--human_node_output_size', type=int, default=5,
                        help='Dimension of the node output')

    # Embedding size
    parser.add_argument('--human_node_embedding_size', type=int, default=128,
                        help='Embedding size of node features')
    parser.add_argument('--human_human_edge_embedding_size', type=int, default=128,
                        help='Embedding size of edge features')

    # Attention vector dimension
    parser.add_argument('--attention_size', type=int, default=64,
                        help='Attention size')

    # Sequence length
    ## Total sequence length (input+ouput)
    parser.add_argument('--seq_length', type=int, default=20,
                        help='Sequence length')
    parser.add_argument('--pred_length', type=int, default=12,
                        help='Predicted sequence length')

    # Batch size
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')

    # Number of epochs
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='number of epochs')

    # Gradient value at which it should be clipped
    parser.add_argument('--grad_clip', type=float, default=10.,
                        help='clip gradients at this value')
    # Lambda regularization parameter (L2)
    parser.add_argument('--lambda_param', type=float, default=0.00005,
                        help='L2 regularization parameter')

    # Learning rate parameter
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    # Decay rate for the learning rate parameter
    parser.add_argument('--decay_rate', type=float, default=0.99,
                        help='decay rate for the optimizer')

    # Dropout rate
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout probability')

    # Use GPU or CPU
    parser.add_argument('--use_cuda', action="store_true", default=False,
                        help='Use GPU or CPU')

    # Save every model or only improved models
    parser.add_argument('--save_every', action='store_true',default=False,
                        help='Save after every epoch?')
    # Train dataset
    # Use like:
    # python transpose_inrange.py --train_dataset index_1 index_2 ...
    parser.add_argument('-l','--train_dataset', nargs='+', help='<Required> training dataset(s): --train_dataset index_1 index_2 ...', default=[0,1,2,4], type=int)
    args=parser.parse_args()

    train(args)


def train(args):

    # Construct the DataLoader object
    ## args: (batch_size=50, seq_length=5, datasets=[0, 1, 2, 3, 4, 5, 6], forcePreProcess=False, infer=False)
    dataloader = DataLoader(args.batch_size, args.seq_length + 1, args.train_dataset, forcePreProcess=True) ##** not sure why seq_length+1

    # Construct the ST-graph object
    ## args: (batch_size=50, seq_length=5)
    stgraph = ST_GRAPH(1, args.seq_length + 1)  ##**not sure why batch_size=1 and seq_length+1

    # Log directory
    log_directory = 'log/trainedOn_'+ str(args.train_dataset)
    if not os.path.exists(log_directory):
            os.makedirs(log_directory)

    # Logging file
    log_file_curve = open(os.path.join(log_directory, 'log_curve.txt'), 'w')
    log_file = open(os.path.join(log_directory, 'val.txt'), 'w')

    # Save directory for saving the model
    save_directory = 'save/trainedOn_'+str(args.train_dataset)
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)

    # Open the configuration file
    ## store arguments from parser
    with open(os.path.join(save_directory, 'config.pkl'), 'wb') as f:
        pickle.dump(args, f)

    # Path to store the checkpoint file, i.e. model after the particular epoch
    def checkpoint_path(x):
        return os.path.join(save_directory, 'srnn_model_'+str(x)+'.tar')

    # Initialize net
    net = SRNN(args)
    if args.use_cuda:        
        net = net.cuda()

    # optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    # optimizer = torch.optim.RMSprop(net.parameters(), lr=args.learning_rate, momentum=0.0001, centered=True)
    optimizer = torch.optim.Adagrad(net.parameters())

    learning_rate = args.learning_rate
    print('Training begin')
    best_val_loss = 100
    best_epoch = 0

    # Training
    for epoch in range(args.num_epochs):
        dataloader.reset_batch_pointer(valid=False)
        loss_epoch = 0

        # For each batch
        for batch in range(dataloader.num_batches):
            start = time.time()

            # Get batch data
            ## Format:
            ## x_batch:     input sequence of length self.seq_length
            ## y_batch:     output seq of same length shifted y 1 step in time
            ## frame_batch: frame IDs in the batch
            ## d:           current position of dataset pointer (points to the next batch to be loaded)
            x, _, _, d = dataloader.next_batch(randomUpdate=True)

            # Loss for this batch
            loss_batch = 0

            # For each sequence in the batch
            for sequence in range(dataloader.batch_size):
                # Construct the graph for the current sequence
                stgraph.readGraph([x[sequence]])

                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float())
                if args.use_cuda:
                    nodes = nodes.cuda()
                edges = Variable(torch.from_numpy(edges).float())
                if args.use_cuda:
                    edges = edges.cuda()

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size))
                if args.use_cuda:
                    hidden_states_node_RNNs = hidden_states_node_RNNs.cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size))
                if args.use_cuda:
                    hidden_states_edge_RNNs = hidden_states_edge_RNNs.cuda()

                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size))
                if args.use_cuda:
                    cell_states_node_RNNs = cell_states_node_RNNs.cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size))
                if args.use_cuda:
                    cell_states_edge_RNNs = cell_states_edge_RNNs.cuda()

                # Zero out the gradients
                net.zero_grad()
                optimizer.zero_grad()

                # Forward prop
                outputs, _, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1], hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)
                loss_batch += loss.data[0]

                # Compute gradients
                loss.backward()

                # Clip gradients
                torch.nn.utils.clip_grad_norm(net.parameters(), args.grad_clip)

                # Update parameters
                optimizer.step()

                # Reset the stgraph
                stgraph.reset()

            end = time.time()
            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

            print(
                '{}/{} (epoch {}), train_loss = {:.3f}, time/batch = {:.3f}'.format(epoch * dataloader.num_batches + batch,
                                                                                    args.num_epochs * dataloader.num_batches,
                                                                                    epoch,
                                                                                    loss_batch, end - start))

        # Compute loss for the entire epoch
        loss_epoch /= dataloader.num_batches
        # Log it
        log_file_curve.write(str(epoch)+','+str(loss_epoch)+',')

        # Validation
        dataloader.reset_batch_pointer(valid=True)
        loss_epoch = 0

        for batch in range(dataloader.valid_num_batches):
            # Get batch data
            x, _, d = dataloader.next_valid_batch(randomUpdate=False)

            # Loss for this batch
            loss_batch = 0

            for sequence in range(dataloader.batch_size):
                stgraph.readGraph([x[sequence]])

                nodes, edges, nodesPresent, edgesPresent = stgraph.getSequence()

                # Convert to cuda variables
                nodes = Variable(torch.from_numpy(nodes).float())
                if args.use_cuda:
                    nodes = nodes.cuda()
                edges = Variable(torch.from_numpy(edges).float())
                if args.use_cuda:
                    edges = edges.cuda()

                # Define hidden states
                numNodes = nodes.size()[1]
                hidden_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size))
                if args.use_cuda:
                    hidden_states_node_RNNs = hidden_states_node_RNNs.cuda()
                hidden_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size))
                if args.use_cuda:
                    hidden_states_edge_RNNs = hidden_states_edge_RNNs.cuda()
                cell_states_node_RNNs = Variable(torch.zeros(numNodes, args.human_node_rnn_size))
                if args.use_cuda:
                    cell_states_node_RNNs = cell_states_node_RNNs.cuda()
                cell_states_edge_RNNs = Variable(torch.zeros(numNodes*numNodes, args.human_human_edge_rnn_size))
                if args.use_cuda:
                    cell_states_edge_RNNs = cell_states_edge_RNNs.cuda()

                outputs, _, _, _, _, _ = net(nodes[:args.seq_length], edges[:args.seq_length], nodesPresent[:-1], edgesPresent[:-1],
                                             hidden_states_node_RNNs, hidden_states_edge_RNNs,
                                             cell_states_node_RNNs, cell_states_edge_RNNs)

                # Compute loss
                loss = Gaussian2DLikelihood(outputs, nodes[1:], nodesPresent[1:], args.pred_length)

                loss_batch += loss.data[0]

                # Reset the stgraph
                stgraph.reset()

            loss_batch = loss_batch / dataloader.batch_size
            loss_epoch += loss_batch

        loss_epoch = loss_epoch / dataloader.valid_num_batches


        #Saving the model\
        if loss_epoch < best_val_loss or args.save_every:
            print('Saving model')
            torch.save({
            'epoch': epoch,
            'state_dict': net.state_dict(),
            'optimizer_state_dict': optimizer.state_dict()
            }, checkpoint_path(epoch)) 

        #save_best_model overwriting the earlier file
        #if loss_epoch < best_val_loss:


        # Update best validation loss until now
        if loss_epoch < best_val_loss:
            best_val_loss = loss_epoch
            best_epoch = epoch
                      
        # Record best epoch and best validation loss
        print('(epoch {}), valid_loss = {:.3f}'.format(epoch, loss_epoch))
        print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
        # Log it
        log_file_curve.write(str(loss_epoch)+'\n')

        

    # Record the best epoch and best validation loss overall
    print('Best epoch {}, Best validation loss {}'.format(best_epoch, best_val_loss))
    # Log it
    log_file.write(str(best_epoch)+','+str(best_val_loss))

    # Close logging files
    log_file.close()
    log_file_curve.close()


if __name__ == '__main__':
	main()