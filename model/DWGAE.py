from libs_funcs import *

class EWConv(nn.Module):
    def __init__(self, in_feats, out_feats, edge_func, aggregator_type='mean'):
        super().__init__()
        self._in_src_feats, self._in_dst_feats = du.expand_as_pair(in_feats)
        self.out_feats = out_feats
        self.edge_func = edge_func
        self.aggregator_type = aggregator_type
        self.pool_func = nn.Linear(self._in_src_feats, self.out_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self.out_feats, self.out_feats, batch_first=True)
        self.self_func = nn.Linear(self._in_src_feats, self.out_feats)
        self.reset_parameters()
        
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.pool_func.weight, gain=gain)
        nn.init.xavier_uniform_(self.self_func.weight, gain=gain)
        if self.aggregator_type == 'lstm':
            self.lstm.reset_parameters()
        
    
    def udf_edge(self, edges):
        return {'edge_features': edges.data['w'], 'neighbors' : edges._src_data['h']}
    
    def udf_u_mul_e(self, nodes):
        m = self.edge_func
        weights = nodes.mailbox['edge_features']
        # weights = torch.div(weights.squeeze(dim = 2), weights.sum(1)).unsqueeze(dim = 2)
        # soft_ed = m(weights)
        soft_ed = m(torch.FloatTensor(np.squeeze(np.apply_along_axis(scaling, 1, weights.numpy()), axis = 2)))
        # num_edges = nodes.mailbox['edge_features'].shape[1]
        res = soft_ed * nodes.mailbox['neighbors']
        if self.aggregator_type == 'sum':
            res = res.sum(axis = 1)
        elif self.aggregator_type == 'mean':
            res = res.mean(axis = 1)
        elif self.aggregator_type == 'max':
            res = res.max(axis = 1)[0]
        elif self.aggregator_type == 'lstm':
            batch_size = res.shape[0]
            hid = (res.new_zeros((1, batch_size, self.out_feats)), res.new_zeros((1, batch_size, self.out_feats)))
            _, (res, _) = self.lstm(res, hid) # only get hidden state
            res = res.permute(1, 0, 2)
        return {'h_reduced' : res}
    
    def forward(self, graph, feat, efeat):
        with graph.local_scope():
            feat_src, feat_dst = du.expand_as_pair(feat, graph)
            graph.srcdata['h'] = self.pool_func(feat_src) 
            graph.edata['w'] = efeat
            graph.update_all(self.udf_edge, self.udf_u_mul_e) 
            result = self.self_func(feat_dst) + graph.dstdata['h_reduced'].squeeze()
            
            return result

class GAE_DWGNN(nn.Module):
    def __init__(self, in_feats, hid_feats, edge_func, aggregator_type):
        super(GAE_DWGNN, self).__init__()
        layers = [EWConv(in_feats, hid_feats[0], edge_func = edge_func, aggregator_type=aggregator_type)]
        if len(hid_feats)>=2:
            layers = [EWConv(in_feats, hid_feats[0], edge_func = edge_func, aggregator_type=aggregator_type)]
            for i in range(1,len(hid_feats)):
                if i != len(hid_feats)-1:
                    layers.append(EWConv(hid_feats[i-1], hid_feats[i], edge_func = edge_func, aggregator_type=aggregator_type))
                else:
                    layers.append(EWConv(hid_feats[i-1], hid_feats[i], edge_func = edge_func, aggregator_type=aggregator_type))
        else:
            layers = [EWConv(in_dim, hid_feats[0], edge_func = edge_func, aggregator_type=aggregator_type)]
        self.layers = nn.ModuleList(layers)
        self.decoder = InnerProductDecoder(activation=lambda x:x)
    
    def forward(self, g, features, edge_features):
        h = features
        e = edge_features
        for conv in self.layers:
            h = conv(g, h, e)
        g.ndata['h'] = h
        adj_rec = self.decoder(h)
        return adj_rec

    def encode(self, g, features, edge_features):
        h = features
        e = edge_features
        for conv in self.layers:
            h = conv(g, h, e)
        return h
