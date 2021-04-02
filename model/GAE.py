from libs_funcs import *

class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        super(MLP, self).__init__()

        self.linear_or_not = True #default is linear model
        self.num_layers = num_layers

        if num_layers == 1:
            #Linear model
            self.linear = nn.Linear(input_dim, output_dim)
        else:
            self.linear_or_not = False #linear model
            self.linears = torch.nn.ModuleList()
        
            for layer in range(num_layers - 1):
                if layer == 0:
                    self.linears.append(nn.Linear(input_dim, hidden_dim))
                else:
                    self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))


    def forward(self, x, batch_norm = False):
        if self.linear_or_not:
            #If linear model
            return self.linear(x)
        else:
            #If MLP
            h = x
            for layer in range(self.num_layers - 1):
                h = F.relu(self.linears[layer](h))
                
            return self.linears[self.num_layers - 1](h)
        
class GAE_GIN(nn.Module):
    def __init__(self, in_dim, hidden_dims, aggregator_type):
        num_mlp_layers = 2
        super(GAE_GIN, self).__init__()
        self.layers = nn.ModuleList()
        # cur_func = MLP(num_mlp_layers, in_dim, hidden_dims[0], hidden_dims[0])
        # layers = [GINConv(cur_func, aggregator_type)]
        if len(hidden_dims)>=2:
            cur_func = MLP(num_mlp_layers, in_dim, hidden_dims[0], hidden_dims[0])
            self.layers.append(GINConv(cur_func, aggregator_type))
            for i in range(1,len(hidden_dims)):
                if i != len(hidden_dims)-1:
                    cur_func = MLP(num_mlp_layers, hidden_dims[i - 1], hidden_dims[i], hidden_dims[i])
                    self.layers.append(GINConv(cur_func, aggregator_type))
                else:
                    cur_func = MLP(num_mlp_layers, hidden_dims[i - 1], hidden_dims[i], hidden_dims[i])
                    self.layers.append(GINConv(cur_func, aggregator_type))
        else:
            cur_func = MLP(num_mlp_layers, in_dim, hidden_dims[0], hidden_dims[0])
            self.layers.append(GINConv(cur_func, aggregator_type))
        self.decoder = InnerProductDecoder(activation=lambda x:x)
    
    def forward(self, g, features):
        h = features
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        adj_rec = self.decoder(h)
        return adj_rec

    def encode(self, g):
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)
        return h

class GAE(nn.Module):
    def __init__(self, in_dim, hidden_dims):
        super(GAE, self).__init__()
        layers = [GraphConv(in_dim, hidden_dims[0], activation = F.relu)]
        if len(hidden_dims)>=2:
            layers = [GraphConv(in_dim, hidden_dims[0], activation = F.relu)]
            for i in range(1,len(hidden_dims)):
                if i != len(hidden_dims)-1:
                    layers.append(GraphConv(hidden_dims[i-1], hidden_dims[i], activation = F.relu))
                else:
                    layers.append(GraphConv(hidden_dims[i-1], hidden_dims[i], activation = lambda x:x))
        else:
            layers = [GraphConv(in_dim, hidden_dims[0], activation = lambda x:x)]
        self.layers = nn.ModuleList(layers)
        self.decoder = InnerProductDecoder(activation=lambda x:x)
    
    def forward(self, g, features):
        h = features
        for conv in self.layers:
            h = conv(g, h)
        g.ndata['h'] = h
        adj_rec = self.decoder(h)
        return adj_rec

    def encode(self, g):
        h = g.ndata['h']
        for conv in self.layers:
            h = conv(g, h)
        return h

