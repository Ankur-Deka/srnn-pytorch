'''
The structural RNN model
introduced in https://arxiv.org/abs/1511.05298

Author : Anirudh Vemula
Date : 16th March 2017


##
Note that computations are done trogetehr for all the people in time steps. The information is all stored in arrays/lists so that everything can be dealt with together

##
'''
import torch.nn as nn
from torch.autograd import Variable
import torch
import numpy as np


class HumanNodeRNN(nn.Module):
    '''
    Class representing human Node RNNs in the st-graph
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(HumanNodeRNN, self).__init__()

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        # Store required sizes
        self.rnn_size = args.human_node_rnn_size
        self.output_size = args.human_node_output_size
        self.embedding_size = args.human_node_embedding_size
        self.input_size = args.human_node_input_size
        self.edge_rnn_size = args.human_human_edge_rnn_size

        # Linear layer to embed input ## 'e'
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # Linear layer to embed edgeRNN hidden states ## 'a'
        self.edge_embed = nn.Linear(self.edge_rnn_size, self.embedding_size)

        # Linear layer to embed attention module output ##' a'
        self.edge_attention_embed = nn.Linear(self.edge_rnn_size*2, self.embedding_size)

        # The LSTM cell
        self.cell = nn.LSTMCell(2*self.embedding_size, self.rnn_size)

        # Output linear layer
        self.output_linear = nn.Linear(self.rnn_size, self.output_size)

    def forward(self, pos, h_temporal, h_spatial_other, h, c):
        #print('forward2')
        '''
        Forward pass for the model
        params:
        pos : input position
        h_temporal : hidden state of the temporal edgeRNN corresponding to this node
        h_spatial_other : output of the attention module
        h : hidden state of the current nodeRNN
        c : cell state of the current nodeRNN
        '''
        # Encode the input position
        ## 'e'
        encoded_input = self.encoder_linear(pos)
        encoded_input = self.relu(encoded_input) 
        encoded_input = self.dropout(encoded_input)

        # Concat both the embeddings
        ## 'a'
        h_edges = torch.cat((h_temporal, h_spatial_other), 1)
        h_edges_embedded = self.relu(self.edge_attention_embed(h_edges))
        h_edges_embedded = self.dropout(h_edges_embedded)

        ## 'concat(e,a)'
        ##$ This is important for me
        concat_encoded = torch.cat((encoded_input, h_edges_embedded), 1)
        #print('concat_encoded',concat_encoded.shape) 1x256
        
        # One-step of LSTM
        ##$ This is important for me
        h_new, c_new = self.cell(concat_encoded, (h, c))

        # Get output
        out = self.output_linear(h_new)


        ## Things I require:
        ## h(t-1):  h
        ## c(t-1):  c
        ## (e,a):   concat_encoded
        ## W:       self.cell.weight_ih/weight_hh/bias_ih/bias_hh       
        ## H in parts:

        
        #w_hi, w_hf, w_hc, w_ho = self.cell.weight_hh_l0.chunk(4, 0)
        # if self.infer:
        #     print('cell parameters')
        #     i=0
        #     for name,parameter in self.cell.named_parameters():
        #         print('parameter',i,name,parameter.shape)
        #         i+=1  
        ## cell parameters
        ## parameter 0 weight_ih torch.Size([512, 256])
        ## parameter 1 weight_hh torch.Size([512, 128])
        ## parameter 2 bias_ih torch.Size([512])
        ## parameter 3 bias_hh torch.Size([512])

        #print('weight',self.cell.weight_if.shape)
        return out, h_new, c_new


class HumanHumanEdgeRNN(nn.Module):
    '''
    Class representing the Human-Human Edge RNN in the s-t graph
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(HumanHumanEdgeRNN, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.rnn_size = args.human_human_edge_rnn_size
        self.embedding_size = args.human_human_edge_embedding_size
        self.input_size = args.human_human_edge_input_size

        # Linear layer to embed input
        self.encoder_linear = nn.Linear(self.input_size, self.embedding_size)

        # ReLU and Dropout layers
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(args.dropout)

        # The LSTM cell
        self.cell = nn.LSTMCell(self.embedding_size, self.rnn_size)

    def forward(self, inp, h, c):
        '''
        Forward pass for the model
        params:
        inp : input edge features
        h : hidden state of the current edgeRNN
        c : cell state of the current edgeRNN
        '''
        # Encode the input position
        encoded_input = self.encoder_linear(inp)
        encoded_input = self.relu(encoded_input)
        encoded_input = self.dropout(encoded_input)

        # One-step of LSTM
        h_new, c_new = self.cell(encoded_input, (h, c))

        return h_new, c_new


class EdgeAttention(nn.Module):
    '''
    Class representing the attention module
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(EdgeAttention, self).__init__()

        self.args = args
        self.infer = infer

        # Store required sizes
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.human_node_rnn_size = args.human_node_rnn_size
        self.attention_size = args.attention_size

        # Linear layer to embed temporal edgeRNN hidden state
        self.temporal_edge_layer = nn.Linear(self.human_human_edge_rnn_size, self.attention_size)

        # Linear layer to embed spatial edgeRNN hidden states
        self.spatial_edge_layer = nn.Linear(self.human_human_edge_rnn_size, self.attention_size)

    def forward(self, h_temporal, h_spatials):
        '''
        Forward pass for the model
        params:
        h_temporal : Hidden state of the temporal edgeRNN
        h_spatials : Hidden states of all spatial edgeRNNs connected to the node.
        '''
        # Number of spatial edges
        num_edges = h_spatials.size()[0]

        # Embed the temporal edgeRNN hidden state
        temporal_embed = self.temporal_edge_layer(h_temporal)
        temporal_embed = temporal_embed.squeeze(0)

        # Embed the spatial edgeRNN hidden states
        spatial_embed = self.spatial_edge_layer(h_spatials)

        # Dot based attention
        attn = torch.mv(spatial_embed, temporal_embed)

        # Variable length
        temperature = num_edges / np.sqrt(self.attention_size)
        attn = torch.mul(attn, temperature)

        # Softmax
        ## Softmax values of attention weights
        attn = torch.nn.functional.softmax(attn)

        # Compute weighted value
        # Uppercase H
        weighted_value = torch.mv(torch.t(h_spatials), attn)

        return weighted_value, attn


class SRNN(nn.Module):
    '''
    Class representing the SRNN model
    '''
    def __init__(self, args, infer=False):
        '''
        Initializer function
        params:
        args : Training arguments
        infer : Training or test time (True at test time)
        '''
        super(SRNN, self).__init__()    #initialize the parent class

        self.args = args
        self.infer = infer
        self.use_cuda = args.use_cuda

        if self.infer:
            # Test time
            self.seq_length = 1
            self.obs_length = 1
        else:
            # Training time
            self.seq_length = args.seq_length
            self.obs_length = args.seq_length - args.pred_length

        # Store required sizes
        self.human_node_rnn_size = args.human_node_rnn_size
        self.human_human_edge_rnn_size = args.human_human_edge_rnn_size
        self.output_size = args.human_node_output_size

        # Initialize the Node and Edge RNNs
        self.humanNodeRNN = HumanNodeRNN(args, infer)
        self.humanhumanEdgeRNN_spatial = HumanHumanEdgeRNN(args, infer)
        self.humanhumanEdgeRNN_temporal = HumanHumanEdgeRNN(args, infer)

        # Initialize attention module
        self.attn = EdgeAttention(args, infer)

    def forward(self, nodes, edges, nodesPresent, edgesPresent, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs):
        #print('forward1')
        '''
        Forward pass for the model
        params:
        nodes : input node features
        edges : input edge features
        nodesPresent : A list of lists, of size seq_length
        Each list contains the nodeIDs that are present in the frame
        edgesPresent : A list of lists, of size seq_length
        Each list contains tuples of nodeIDs that have edges in the frame
        hidden_states_node_RNNs : A tensor of size numNodes x node_rnn_size
        Contains hidden states of the node RNNs
        hidden_states_edge_RNNs : A tensor of size numNodes x numNodes x edge_rnn_size
        Contains hidden states of the edge RNNs

        returns:
        outputs : A tensor of shape seq_length x numNodes x 5
        Contains the predictions for next time-step
        hidden_states_node_RNNs
        hidden_states_edge_RNNs
        '''
        # Get number of nodes
        ## In the whole sequence, some may be missing from some frames
        numNodes = nodes.size()[1]

        # Initialize output array
        outputs = Variable(torch.zeros(self.seq_length*numNodes, self.output_size))
        if self.use_cuda:
            outputs = outputs.cuda()

        # Data structure to store attention weights
        attn_weights = [{} for _ in range(self.seq_length)]

        # New attention weights
        new_attn_weights = [{} for _ in range(self.seq_length)]        

        # For each frame
        for framenum in range(self.seq_length):
            # Find the edges present in the current frame
            edgeIDs = edgesPresent[framenum]

            # Separate temporal and spatial edges
            temporal_edges = [x for x in edgeIDs if x[0] == x[1]]
            spatial_edges = [x for x in edgeIDs if x[0] != x[1]]

            # Find the nodes present in the current frame
            nodeIDs = nodesPresent[framenum]

            # Get features of the nodes and edges present
            nodes_current = nodes[framenum]
            edges_current = edges[framenum]

            # Initialize temporary tensors
            hidden_states_nodes_from_edges_temporal = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size))
            hidden_states_nodes_from_edges_spatial = Variable(torch.zeros(numNodes, self.human_human_edge_rnn_size))
            if self.use_cuda:
                hidden_states_nodes_from_edges_spatial = hidden_states_nodes_from_edges_spatial.cuda()
                hidden_states_nodes_from_edges_temporal = hidden_states_nodes_from_edges_temporal.cuda()

            # If there are any edges
            if len(edgeIDs) != 0:

                # Temporal Edges
                if len(temporal_edges) != 0:
                    # Get the temporal edges
                    list_of_temporal_edges = Variable(torch.LongTensor([x[0]*numNodes + x[0] for x in edgeIDs if x[0] == x[1]]))
                    if self.use_cuda:
                        list_of_temporal_edges = list_of_temporal_edges.cuda()
                    # Get nodes associated with the temporal edges
                    list_of_temporal_nodes = torch.LongTensor([x[0] for x in edgeIDs if x[0] == x[1]])
                    if self.use_cuda:
                        list_of_temporal_nodes = list_of_temporal_nodes.cuda()

                    # Get the corresponding edge features
                    edges_temporal_start_end = torch.index_select(edges_current, 0, list_of_temporal_edges)
                    # Get the corresponding hidden states
                    hidden_temporal_start_end = torch.index_select(hidden_states_edge_RNNs, 0, list_of_temporal_edges)
                    # Get the corresponding cell states
                    cell_temporal_start_end = torch.index_select(cell_states_edge_RNNs, 0, list_of_temporal_edges)

                    # Do forward pass through temporaledgeRNN
                    h_temporal, c_temporal = self.humanhumanEdgeRNN_temporal(edges_temporal_start_end, hidden_temporal_start_end, cell_temporal_start_end)

                    # Update the hidden state and cell state
                    hidden_states_edge_RNNs[list_of_temporal_edges.data] = h_temporal
                    cell_states_edge_RNNs[list_of_temporal_edges.data] = c_temporal

                    # Store the temporal hidden states obtained in the temporary tensor
                    hidden_states_nodes_from_edges_temporal[list_of_temporal_nodes] = h_temporal

                # Spatial Edges
                if len(spatial_edges) != 0:
                    # Get the spatial edges
                    list_of_spatial_edges = Variable(torch.LongTensor([x[0]*numNodes + x[1] for x in edgeIDs if x[0] != x[1]]))
                    if self.use_cuda:
                        list_of_spatial_edges = list_of_spatial_edges.cuda()
                    # Get nodes associated with the spatial edges
                    list_of_spatial_nodes = np.array([x[0] for x in edgeIDs if x[0] != x[1]])

                    # Get the corresponding edge features
                    edges_spatial_start_end = torch.index_select(edges_current, 0, list_of_spatial_edges)
                    # Get the corresponding hidden states
                    hidden_spatial_start_end = torch.index_select(hidden_states_edge_RNNs, 0, list_of_spatial_edges)
                    # Get the corresponding cell states
                    cell_spatial_start_end = torch.index_select(cell_states_edge_RNNs, 0, list_of_spatial_edges)

                    # Do forward pass through spatialedgeRNN
                    h_spatial, c_spatial = self.humanhumanEdgeRNN_spatial(edges_spatial_start_end, hidden_spatial_start_end, cell_spatial_start_end)

                    # Update the hidden state and cell state
                    hidden_states_edge_RNNs[list_of_spatial_edges.data] = h_spatial
                    cell_states_edge_RNNs[list_of_spatial_edges.data] = c_spatial

                    # pass it to attention module
                    # For each node
                    for node in range(numNodes):
                        # Get the indices of spatial edges associated with this node
                        l = np.where(list_of_spatial_nodes == node)[0]
                        if len(l) == 0:
                            # If the node has no spatial edges, nothing to do
                            continue
                        l = torch.LongTensor(l)
                        if self.use_cuda:
                            l = l.cuda()
                        # What are the other nodes with these edges?
                        node_others = [x[1] for x in edgeIDs if x[0] == node and x[0] != x[1]]                        
                        # If it has spatial edges
                        # Get its corresponding temporal edgeRNN hidden state
                        h_node = hidden_states_nodes_from_edges_temporal[node]

                        # Do forward pass through attention module
                        # Returns: uppercase H, softmax values of attention (score)
                        hidden_attn_weighted, attn_w = self.attn(h_node.view(1, -1), h_spatial[l])

                        # Store the attention weights
                        attn_weights[framenum][node] = (attn_w.data.cpu().numpy(), node_others)

                        # Store the output of attention module in temporary tensor
                        hidden_states_nodes_from_edges_spatial[node] = hidden_attn_weighted

                    
                    # NEW ATTENTION interpretation
                    if self.infer:
                        #print('New set of nodes')
                        #print('numNodes',numNodes)  #Total nodes in the sequence
                        #print('len(nodeIDs)',len(nodeIDs))  #Nodes in the frame
                        #print('nodeIDs',nodeIDs)
                        i=0
                        # for each node
                        for node in range(numNodes):

                            # REFERENFCE
                            # h_oth: spatial hidden states from other nodes
                            # a_w: attention weights (after softmax)
                            # w_ih,w_hh,b_ih,b_hh: weights and biases
                            # cur_h_states: current hidden states of all relevant nodes. One row for each node.
                            # cur_c_states: current cell states of all the relevant nodes. One row for each node.

                            # Get the indices of spatial edges associated with this node
                            l = np.where(list_of_spatial_nodes == node)[0]
                            if len(l) == 0:
                                # If the node has no spatial edges, nothing to do
                                continue

                            ## Perform data extraction in 3 parts
                            ## PART 1
                            # Get the indices of spatial edges associated with this node
                            l = np.where(list_of_spatial_nodes == node)[0]
                            l = torch.LongTensor(l)
                            if self.use_cuda:
                                l = l.cuda()
                            # What are the other nodes with these edges?
                            node_others = [x[1] for x in edgeIDs if x[0] == node and x[0] != x[1]]                        
                            # If it has spatial edges
                            # Get its corresponding temporal edgeRNN hidden state
                            h_node = hidden_states_nodes_from_edges_temporal[node]
                            h_oth=h_spatial[l].data
                            a_w_withID=attn_weights[framenum][node]
                            a_w=a_w_withID[0]   #extract only weights
                            a_w=a_w.reshape(a_w.shape[0],1) #reshape as column vector
                            a_w=torch.Tensor(a_w)
                            #print('h spatial others:',h_oth.shape)
                            #print('attn_weights',a_w)


                            ## PART 2
                            # Get list of nodes
                            list_of_nodes = Variable(torch.LongTensor(nodeIDs))
                            if self.use_cuda:
                                list_of_nodes = list_of_nodes.cuda()
                            
                            # Get their node features
                            nodes_current_selected = torch.index_select(nodes_current, 0, list_of_nodes)

                            # Get the hidden and cell states of the corresponding nodes. Data for different nodes in  different rows
                            cur_h_states = torch.index_select(hidden_states_node_RNNs, 0, list_of_nodes)
                            cur_c_states = torch.index_select(cell_states_node_RNNs, 0, list_of_nodes)
                            cur_h_node = cur_h_states[i].data.view(cur_h_states.shape[1],1)    #select the required node's data
                            i+=1

                            #print('Current',cur_h_states.shape,cur_c_states.shape)


                            ## PART 3
                            #print('cell parameters')
                            w_ih=self.humanNodeRNN.cell.weight_ih.data
                            w_hh=self.humanNodeRNN.cell.weight_hh.data
                            b_ih=self.humanNodeRNN.cell.bias_ih.data
                            b_hh=self.humanNodeRNN.cell.bias_hh.data
                            #add dimension
                            b_ih=b_ih.view(b_ih.shape[0],1)
                            b_hh=b_hh.view(b_hh.shape[0],1)
                            #print(w_ih.shape,w_hh.shape,b_ih.shape,b_hh.shape)


                            ## Perform the actual computation
                            ##! Assuming that the hidden and cell states are stored in order of IDs
                            ##! Assuming that weights inside LSTMCell are in the order i,f,g,o

                            # Get weighted hidden states as different columns                            
                            h_oth_w=(h_oth*a_w).t()

                            # Get q_k's first part with different k's in different columns
                            q=torch.matmul(w_ih,h_oth_w)
                            
                            # Update q_k with the later terms
                            q+=(b_ih+torch.matmul(w_hh,cur_h_node)+b_hh)/(len(nodeIDs)-1)

                            # In exponential term
                            exp_q=torch.sigmoid(q)

                            # Should we normalize gatewise (128 gates), I think no
                            # # Q-tot = normalizing term
                            # q_tot=torch.ones(exp_q.shape[0],1)
                            # for ii in range(exp_q.shape[1]):
                            #     q_tot*=exp_q[:,ii]

                            # # normalized q
                            # norm_q=exp_q#/q_tot
                            
                            # size of hidden states
                            size=self.human_node_rnn_size
                            
                            # computing over forget and new information gates
                            new_info=(exp_q[0:size,:]-exp_q[size:size*2,:])

                            #considering the output gate
                            eff_info=new_info*exp_q[3*size:4*size,:]
                            
                            # totalling over all gates
                            new_attn=torch.sum(eff_info,dim=0)
                            new_attn/=sum(new_attn)
                            new_attn*=3                            
                            
                            # Store the attention weights
                            new_attn_weights[framenum][node] = (new_attn.numpy(), node_others)
                            


                            
                                                    




            # If there are nodes in this frame
            if len(nodeIDs) != 0:

                # Get list of nodes
                list_of_nodes = Variable(torch.LongTensor(nodeIDs))
                if self.use_cuda:
                    list_of_nodes = list_of_nodes.cuda()

                ## Select as array because we need the quantities of all the nodes
                ##* I didn't understand why selection is needed here-it's basically selecting everything. Maybe it's just because of data type conversion or my lack of understanding of pytorch tensors

                # Get their node features
                nodes_current_selected = torch.index_select(nodes_current, 0, list_of_nodes)

                # Get the hidden and cell states of the corresponding nodes
                hidden_nodes_current = torch.index_select(hidden_states_node_RNNs, 0, list_of_nodes)
                cell_nodes_current = torch.index_select(cell_states_node_RNNs, 0, list_of_nodes)

                # Get the temporal edgeRNN hidden states corresponding to these nodes
                h_temporal_other = hidden_states_nodes_from_edges_temporal[list_of_nodes.data]
                h_spatial_other = hidden_states_nodes_from_edges_spatial[list_of_nodes.data]

                # Do a forward pass through nodeRNN
                ##** it was written self.humanNodeRNN instead of self.humanNodeRNN.forward but it still worked
                outputs[framenum * numNodes + list_of_nodes.data], h_nodes, c_nodes = self.humanNodeRNN.forward(nodes_current_selected, h_temporal_other, h_spatial_other, hidden_nodes_current, cell_nodes_current)

                # Update the hidden and cell states
                hidden_states_node_RNNs[list_of_nodes.data] = h_nodes
                cell_states_node_RNNs[list_of_nodes.data] = c_nodes

        # Reshape the outputs carefully
        outputs_return = Variable(torch.zeros(self.seq_length, numNodes, self.output_size))
        if self.use_cuda:
            outputs_return = outputs_return.cuda()
        for framenum in range(self.seq_length):
            for node in range(numNodes):
                outputs_return[framenum, node, :] = outputs[framenum*numNodes + node, :]

        return outputs_return, hidden_states_node_RNNs, hidden_states_edge_RNNs, cell_states_node_RNNs, cell_states_edge_RNNs, attn_weights, new_attn_weights
