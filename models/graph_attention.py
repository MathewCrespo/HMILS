import torch
from torch._C import set_flush_denormal
import torch.nn as nn
import torch.nn.functional as F
from .ResNet import ResNet18, ResNet10
from torch_geometric.nn import SAGPooling
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp

class FE_Res(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(FE_Res, self).__init__()
        self.input_dim = input_dim
        self.L = L
        self.D = D
        self.K = K

        self.feature_extractor_part1 = ResNet10()

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(512 * 4 * 4, self.L),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.squeeze(0)
        #print(x.shape)
        H = self.feature_extractor_part1(x)
        #print(H.shape)       
        H = H.view(-1, 512 * 4* 4)       
        H = self.feature_extractor_part2(H)  # NxL
        return H


class FE(nn.Module):
    def __init__(self, input_dim=3, L=500, D=128, K=1):
        super(FE, self).__init__()
        self.input_dim = input_dim
        self.L = L
        self.D = D
        self.K = K

        self.feature_extractor_part1 = nn.Sequential(
            nn.Conv2d(self.input_dim, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(50 * 25 * 25, self.L),
            nn.ReLU(),
        )

    def forward(self, x):
        x = x.squeeze(0)
        #print(x.shape)
        H = self.feature_extractor_part1(x)
        #print(H.shape)  
        H = H.view(-1, 50 * 25* 25)       
        H = self.feature_extractor_part2(H)  # NxL
        return H

class GCN_H (nn.Module):  
    def __init__(self,num_features=500, nhid=256, num_classes=2, pooling_ratio = 0.75):
        super(GCN_H,self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio
        
        self.conv1 = GCNConv(500, self.nhid)
        #print(1)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)

    def get_threshold (self,x):

        gamma = 0
        
        node_num = x.shape[0]
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            temp_max,_ = torch.max(f_dist,dim=0)
            gamma = max(gamma,temp_max.item())
        
        return gamma*0.5

    def get_edge_index(self,x):        
        t = self.get_threshold(x)
        node_num = x.shape[0]
        edge_index = [[],[]] # source nodes and target nodes
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            index = (f_dist < t)
            #print(index)
            for j in range(i+1,node_num):
                if index[j]:

                    edge_index[0].append(i) #source
                    edge_index[1].append(j)

        return torch.LongTensor(edge_index).cuda()
    
    def forward(self, feature):
        edge_index = self.get_edge_index(feature)        
        x = F.relu(self.conv1(feature,edge_index))
        x, edge_index, _, batch, perm1, score1 = self.pool1(x, edge_index)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch,perm2, score2 = self.pool2(x, edge_index)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, perm3, score3 = self.pool3(x, edge_index)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3   # image feature after SAGPooling
        '''
        l1 = perm1.tolist()
        s1 = score1.tolist()
        l2 = perm2.tolist()
        s2 = score2.tolist()
        l3 = perm3.tolist()
        s3 = score3.tolist()
        resume1 = [l2[i] for i in l3 ]
        resume2 = [l1[i] for i in resume1]
        '''
        return x


class GCN_Pos (nn.Module):  
    def __init__(self,num_features=500, nhid=256, num_classes=2, pooling_ratio = 0.75):
        super(GCN_Pos,self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio

        self.pos_embed = nn.Sequential(
            nn.Linear(6, 12),
            nn.ReLU()
        )
        
        self.conv1 = GCNConv(512, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.layernorm_f = nn.LayerNorm(num_features)
        self.layernorm_p = nn.LayerNorm(num_features)

    def get_threshold (self,x):

        gamma = 0
        
        node_num = x.shape[0]
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            temp_max,_ = torch.max(f_dist,dim=0)
            gamma = max(gamma,temp_max.item())
        
        return gamma*0.5

    def get_edge_index(self,x):        
        t = self.get_threshold(x)
        node_num = x.shape[0]
        edge_index = [[],[]] # source nodes and target nodes
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            index = (f_dist < t)
            #print(index)
            for j in range(i+1,node_num):
                if index[j]:

                    edge_index[0].append(i) #source
                    edge_index[1].append(j)

        return torch.LongTensor(edge_index).cuda()
    
    def forward(self, feature, img_info):
        # position embedding using simple linear layer
        #feature = self.layernorm_f(feature)
        pos_info = self.pos_embed(img_info)
        #pos_info = self.layernorm_p(pos_info)

        feature = torch.cat([feature,pos_info],dim=1)
        edge_index = self.get_edge_index(feature)        
        x = F.relu(self.conv1(feature,edge_index))
        x, edge_index, _, batch, perm1, score1 = self.pool1(x, edge_index)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch,perm2, score2 = self.pool2(x, edge_index)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, perm3, score3 = self.pool3(x, edge_index)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3   # image feature after SAGPooling
        return x

class GCN_Pos_normadd (nn.Module):  
    def __init__(self,num_features=500, nhid=256, num_classes=2, pooling_ratio = 0.75):
        super(GCN_Pos_normadd,self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio

        self.pos_embed = nn.Sequential(
            nn.Linear(6, 500)
        )
        
        self.conv1 = GCNConv(500, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.layernorm_f = nn.LayerNorm(num_features)
        self.layernorm_p = nn.LayerNorm(num_features)

    def get_threshold (self,x):

        gamma = 0
        
        node_num = x.shape[0]
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            temp_max,_ = torch.max(f_dist,dim=0)
            gamma = max(gamma,temp_max.item())
        
        return gamma*0.5

    def get_edge_index(self,x):        
        t = self.get_threshold(x)
        node_num = x.shape[0]
        edge_index = [[],[]] # source nodes and target nodes
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            index = (f_dist < t)
            #print(index)
            for j in range(i+1,node_num):
                if index[j]:

                    edge_index[0].append(i) #source
                    edge_index[1].append(j)

        return torch.LongTensor(edge_index).cuda()
    
    def forward(self, feature, img_info):
        # position embedding using simple linear layer
        feature = self.layernorm_f(feature)
        pos_info = self.pos_embed(img_info)
        pos_info = self.layernorm_p(pos_info)

        feature = feature + pos_info
        edge_index = self.get_edge_index(feature)        
        x = F.relu(self.conv1(feature,edge_index))
        x, edge_index, _, batch, perm1, score1 = self.pool1(x, edge_index)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch,perm2, score2 = self.pool2(x, edge_index)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, perm3, score3 = self.pool3(x, edge_index)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3   # image feature after SAGPooling
        return x

class GCN_Pos_normcat (nn.Module):  
    def __init__(self,t,num_features=500, nhid=256, num_classes=2, pooling_ratio = 0.75):
        super(GCN_Pos_normcat,self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.t = t
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio

        self.pos_embed = nn.Sequential(
            nn.Linear(6, 12)
        )
        
        self.conv1 = GCNConv(512, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.layernorm_f = nn.LayerNorm(num_features)
        self.layernorm_p = nn.LayerNorm(12)

    def get_threshold (self,x):

        gamma = 0
        
        node_num = x.shape[0]
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            temp_max,_ = torch.max(f_dist,dim=0)
            gamma = max(gamma,temp_max.item())
        
        return gamma*self.t

    def get_edge_index(self,x):        
        t = self.get_threshold(x)
        node_num = x.shape[0]
        edge_index = [[],[]] # source nodes and target nodes
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            index = (f_dist < t)
            #print(index)
            for j in range(i+1,node_num):
                if index[j]:

                    edge_index[0].append(i) #source
                    edge_index[1].append(j)

        return torch.LongTensor(edge_index).cuda()
    
    def forward(self, feature, img_info):
        # position embedding using simple linear layer
        feature = self.layernorm_f(feature)
        pos_info = self.pos_embed(img_info)
        pos_info = self.layernorm_p(pos_info)

        feature = torch.cat([feature,pos_info],dim=1)
        edge_index = self.get_edge_index(feature)        
        x = F.relu(self.conv1(feature,edge_index))
        x, edge_index, _, batch, perm1, score1 = self.pool1(x, edge_index)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch,perm2, score2 = self.pool2(x, edge_index)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, perm3, score3 = self.pool3(x, edge_index)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3   # image feature after SAGPooling
        return x

class GCN_Pos_attention (nn.Module):  
    def __init__(self,num_features=500, nhid=256, num_classes=2, pooling_ratio = 0.75):
        super(GCN_Pos_attention,self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio

        self.pos_embed = nn.Sequential(
            nn.Linear(6, 12)
        )
        
        self.pos_conv = GCNConv(12, 12)
        self.conv1 = GCNConv(512, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.layernorm_f = nn.LayerNorm(num_features)
        self.layernorm_p = nn.LayerNorm(12)

    def get_weights(self,pos_embed):  # for positional graph
        ## output type torch.Tensor NxN
        node_num = pos_embed.shape[0]
        edge_weights = torch.zeros((node_num,node_num))
        edge_index = [[],[]]
        for i in range(node_num):
            for j in range(i+1,node_num):
                edge_weights[i][j] = edge_weights[j][i] = torch.cosine_similarity(pos_embed[i,:],pos_embed[j,:],
                dim = 0)
                edge_index[0].append(i)
                edge_index[1].append(j)
        edge_weights = edge_weights.view(1,-1)

        return edge_weights.cuda(), torch.LongTensor(edge_index).cuda()
    
    def get_attention(self,pos_embed):
        q = pos_embed
        k = pos_embed
        weights = torch.mm(q,k.T)/ torch.sqrt(torch.tensor(12.))
        '''
        for i in range(node_num):
            for j in range(i+1,node_num):
                a = pos_embed[i,:]
                b = pos_embed[j,:]
                sim = torch.dot(a,b)/torch.sqrt(torch.dot(a,a)*torch.dot(b,b))
                edge_weights[i][j] = edge_weights[j][i] = sim
        '''
        return weights


    def get_threshold (self,x):  # for semantic graph
        gamma = 0
        node_num = x.shape[0]
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            temp_max,_ = torch.max(f_dist,dim=0)
            gamma = max(gamma,temp_max.item())
        return gamma*0.5

    def get_edge_index(self,x):  # for semantic graph
        t = self.get_threshold(x)
        node_num = x.shape[0]
        edge_index = [[],[]] # source nodes and target nodes
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            index = (f_dist < t)
            #print(index)
            for j in range(i+1,node_num):
                if index[j]:

                    edge_index[0].append(i) #source
                    edge_index[1].append(j)

        return torch.LongTensor(edge_index).cuda()
    
    def forward(self, feature, img_info):
        # position projection
        pos_info = self.pos_embed(img_info)
        '''
        # position graph conv
        edge_weights,pos_edge = self.get_weights(pos_info)
        pos_info = F.relu(self.pos_conv(pos_info,pos_edge,edge_weights))
        '''
        weights = self.get_attention(pos_info)
        weights = F.softmax(weights,dim=1)
        pos_info = torch.mm(weights.cuda(),pos_info)
        # merge position embding and semantic embding
        feature = self.layernorm_f(feature)
        pos_info = self.layernorm_p(pos_info)


        feature = torch.cat([feature,pos_info],dim=1)
        edge_index = self.get_edge_index(feature)        
        x = F.relu(self.conv1(feature,edge_index))
        x, edge_index, _, batch, perm1, score1 = self.pool1(x, edge_index)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch,perm2, score2 = self.pool2(x, edge_index)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, perm3, score3 = self.pool3(x, edge_index)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3   # image feature after SAGPooling
        return x

class GCN_Pos_conv (nn.Module):  
    def __init__(self,num_features=500, nhid=256, num_classes=2, pooling_ratio = 0.75):
        super(GCN_Pos_conv,self).__init__()
        self.num_features = num_features
        self.nhid = nhid
        self.num_classes = num_classes
        self.pooling_ratio = pooling_ratio

        self.pos_embed = nn.Sequential(
            nn.Linear(6, 12)
        )
        self.pos_conv = GCNConv(12, 12)
        self.conv1 = GCNConv(512, self.nhid)
        self.pool1 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.pool2 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.pool3 = SAGPooling(self.nhid, ratio=self.pooling_ratio)
        self.layernorm_f = nn.LayerNorm(num_features)
        self.layernorm_p = nn.LayerNorm(12)

    def get_threshold (self,x):
        gamma = 0
        node_num = x.shape[0]
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            temp_max,_ = torch.max(f_dist,dim=0)
            gamma = max(gamma,temp_max.item())
        return gamma*0.5

    def get_edge_index(self,x):
        t = self.get_threshold(x)
        node_num = x.shape[0]
        edge_index = [[],[]] # source nodes and target nodes
        for i in range(node_num):
            f_dist = torch.sum((x-x[i,:])**2,dim=1)
            index = (f_dist < t)
            #print(index)
            for j in range(i+1,node_num):
                if index[j]:

                    edge_index[0].append(i) #source
                    edge_index[1].append(j)

        return torch.LongTensor(edge_index).cuda()
    
    def forward(self, feature, img_info):
        # position projection
        pos_info = self.pos_embed(img_info)
        # for positional graph
        pos_edge = self.get_edge_index(pos_info)
        pos_info = F.relu(self.pos_conv(pos_info,pos_edge))
        # merge position embding and semantic embding after layer normalization
        feature = self.layernorm_f(feature)
        pos_info = self.layernorm_p(pos_info)
        feature = torch.cat([feature,pos_info],dim=1)
        edge_index = self.get_edge_index(feature)        
        x = F.relu(self.conv1(feature,edge_index))
        x, edge_index, _, batch, perm1, score1 = self.pool1(x, edge_index)
        x1 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch,perm2, score2 = self.pool2(x, edge_index)
        x2 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)
        
        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, perm3, score3 = self.pool3(x, edge_index)
        x3 = torch.cat([gmp(x, batch), gap(x, batch)], dim=1)

        x = x1 + x2 + x3   # image feature after SAGPooling
        return x

class Attention(nn.Module):
    def __init__(self, task, D=128, K=1):
        super(Attention, self).__init__()
        self.D = D
        self.K = K
        if task=='BM':
            self.th = 0.5
            self.f = 1.0
        else:
            self.th = 0.45
            self.f = 3.0

        self.attention_layer = nn.Sequential(
            nn.Linear(512, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
        nn.Linear(512, 1),
        nn.Sigmoid()
        )

    def forward(self, x):
        A = self.attention_layer(x)
        A = torch.transpose(A, 0, 1)
        A = F.softmax(A, dim=1)

        bag_f = torch.mm(A, x)

        prob = self.classifier(bag_f)
        pred = torch.ge(prob, self.th).float()

        return prob, pred, A
    
    def cal_loss(self, X, Y):
        Y = Y.float()
        Y_prob, Y_pred, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)
        neg_log_likelihood = -self.f * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))

        return Y_prob, Y_pred, neg_log_likelihood, A

class H_Attention_Graph(nn.Module):
    def __init__(self,t,task):
        super(H_Attention_Graph,self).__init__()
        if task =='BM':
            self.fe = FE()
        else:
            self.fe = FE_Res()
        self.gcn = GCN_Pos_normcat(t=t)
        self.attn = Attention(task=task)
        
    def forward(self, x, img_info, y, idx_list):
        #print('Input Shape is ', x.shape)
        # part1: feature extractor
        H = self.fe(x)
        img_info = img_info.squeeze(0)

        
        #print('Patch Feature Shape is ', H.shape)

        # part2: construct a graph and get image level feature
        img_features = []
        #perms_set = []
        #scores_set = []
        for i in range(len(idx_list)-1):
            h_patch = H[idx_list[i]:idx_list[i+1], :]
            i_info = img_info[idx_list[i]:idx_list[i+1], :]
            i_feature = self.gcn(h_patch.cuda(), i_info)
            #perms_set.append(perms)
            #scores_set.append(scores)
            #print(i_feature.shape)
            img_features.append(i_feature)

        instance_f = torch.cat([x for x in img_features],dim = 0)
        #print('Image Feature Shape is ', instance_f.shape)

        # part3 bag aggregation and prediction
        Y_prob, Y_pred, loss, weights = self.attn.cal_loss(instance_f,y)

        return Y_prob, Y_pred, loss, weights






            
            
            

        