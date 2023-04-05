import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# ------ TO DO ------
class FeatNet(nn.Module):
    def __init__(self, global_feat = False, transform=False):
        super(FeatNet, self).__init__()
        if transform:
            self.trans1 = trans()
            self.trans2 = trans(input=64)
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.bn2 = nn.BatchNorm1d(128)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.transform = transform

    def forward(self, x):
        x = x.transpose(2, 1)
        if self.transform:
            trans = self.trans1(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans).transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))
        local_feat = x

        if self.transform:
            trans = self.trans2(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x, trans).transpose(2, 1)
        x = F.relu(self.bn2(self.conv2(x)))
        skip = x
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)

        if self.global_feat:
            return x
        elif self.transform:
            x = x.unsqueeze(2).repeat(1, 1, local_feat.shape[2])
            return x, skip
        else:
            x = x.unsqueeze(2).repeat(1, 1, local_feat.shape[2])
            return torch.cat([x, local_feat], 1)

class trans(nn.Module):
    def __init__(self, input=3):
        super(trans, self).__init__()
        self.conv1 = torch.nn.Conv1d(input, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, input*input)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.input = input

    def forward(self, x):
        batchsize = x.size()[0]

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0].view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.from_numpy(np.eye(self.input).flatten().astype(np.float32)).view(1,self.input*self.input).repeat(batchsize,1)
        if x.is_cuda: iden = iden.cuda()
        x = x + iden

        return x.view(-1, self.input, self.input)

    
class cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(cls_model, self).__init__()
        self.feat = FeatNet(global_feat=True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        pass

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = self.feat(points)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)



# ------ TO DO ------
class seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(seg_model, self).__init__()
        self.num_seg_classes = num_seg_classes
        self.feat = FeatNet(global_feat=False)
        self.conv1 = torch.nn.Conv1d(1088, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = torch.nn.Conv1d(128, self.num_seg_classes, 1)


    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        B, N, _= points.size()
        x = self.feat(points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.conv4(x)
        x = F.log_softmax(x.transpose(2,1).contiguous().view(-1,self.num_seg_classes), dim=-1)
        return x.view(B, N, self.num_seg_classes)


class trans_cls_model(nn.Module):
    def __init__(self, num_classes=3):
        super(trans_cls_model, self).__init__()
        self.feat = FeatNet(global_feat=True, transform = True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, num_classes)
        self.dropout = nn.Dropout(p=0.3)
        self.relu = nn.ReLU()
        pass

    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, num_classes)
        '''
        x = self.feat(points)
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)

class trans_seg_model(nn.Module):
    def __init__(self, num_seg_classes=6):
        super(trans_seg_model, self).__init__()
        self.num_seg_classes = num_seg_classes
        self.feat = FeatNet(transform=True)
        self.conv1 = torch.nn.Conv1d(1024, 512, 1)
        self.bn1 = nn.BatchNorm1d(512)
        self.conv2 = torch.nn.Conv1d(512, 256, 1)
        self.bn2 = nn.BatchNorm1d(256)
        self.conv3 = torch.nn.Conv1d(256, 128, 1)
        self.bn3 = nn.BatchNorm1d(128)
        self.conv4 = torch.nn.Conv1d(256, self.num_seg_classes, 1)


    def forward(self, points):
        '''
        points: tensor of size (B, N, 3)
                , where B is batch size and N is the number of points per object (N=10000 by default)
        output: tensor of size (B, N, num_seg_classes)
        '''
        B, N, _= points.size()
        x, skip = self.feat(points)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.cat([skip, x], 1)
        x = self.conv4(x)
        x = F.log_softmax(x.transpose(2,1).contiguous().view(-1,self.num_seg_classes), dim=-1)
        return x.view(B, N, self.num_seg_classes)



