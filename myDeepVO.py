import torch
import numpy as np
import torch.nn as nn
from torch.nn.init import kaiming_normal_, orthogonal_

def conv(in_planes,out_planes,kernel_size,stride,padding,dropout):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(),
        nn.Dropout(dropout)
    )

class myDeepVo(nn.Module):
    def __init__(self,img_w,img_h,frame=7,hidden_size=1000,n_layers=2):
        super(myDeepVo,self).__init__()

        self.conv1   = conv(in_planes=  6,out_planes=  64,kernel_size=(7,7),stride=2,padding=3,dropout=0.2)
        self.conv2   = conv(in_planes= 64,out_planes= 128,kernel_size=(5,5),stride=2,padding=2,dropout=0.2)
        self.conv3   = conv(in_planes=128,out_planes= 256,kernel_size=(5,5),stride=2,padding=2,dropout=0.2)
        self.conv3_1 = conv(in_planes=256,out_planes= 256,kernel_size=(3,3),stride=1,padding=1,dropout=0.2)
        self.conv4   = conv(in_planes=256,out_planes= 512,kernel_size=(3,3),stride=2,padding=1,dropout=0.2)
        self.conv4_1 = conv(in_planes=512,out_planes= 512,kernel_size=(3,3),stride=1,padding=1,dropout=0.2)
        self.conv5   = conv(in_planes=512,out_planes= 512,kernel_size=(3,3),stride=2,padding=1,dropout=0.2)
        self.conv5_1 = conv(in_planes=512,out_planes= 512,kernel_size=(3,3),stride=1,padding=1,dropout=0.2)
        self.conv6   = conv(in_planes=512,out_planes=1024,kernel_size=(3,3),stride=2,padding=1,dropout=0.5)

        # Comput the shape based on diff image size
        # frame預設是7 這邊去頭尾相疊所以是6
        tmp = torch.zeros(1, frame-1, img_w, img_h)
        tmp = self.encode_image(tmp)

        self.lstm    = nn.LSTM(input_size=np.prod(tmp.shape),hidden_size=hidden_size,
                                num_layers=n_layers,batch_first=True,dropout=(0 if n_layers == 1 else 0.5),bidirectional=False)

        self.classifier = nn.Sequential(nn.Linear(in_features=hidden_size,out_features=6))

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  #orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def forward(self,x):
        # x: (batch, seq_len, channel, width, height)
        # 要把照片疊起來，作法是前 6 張 配 後6 張
        x = torch.cat(( x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.shape[0]
        seq_len = x.shape[1]

        # CNN
        x = x.view(batch_size*seq_len, x.shape[2], x.shape[3], x.shape[4])
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)

        # RNN
        out,(h0,c0) = self.lstm(x)
        out = self.classifier(out)

        return out  # batch * frame * 6個參數 theta , x y z
        
def test():

    torch.manual_seed(1214)

    batch_size = 4
    img_w = 200
    img_h = 300
    frame = 7

    # batch * frame * RGB * img_w , img_h
    x = torch.randn(batch_size,frame,3,img_w,img_h)
    y = torch.randn(batch_size,frame,6) # pose 是 6 維度

    MyDeep = myDeepVo(img_w,img_h,frame=frame,hidden_size=1000,n_layers=2)
    print(MyDeep)
    
    predict = MyDeep(x)
    print(predict , predict.shape)

if __name__ == '__main__':
    test()

    

class DeepVO(nn.Module):
    def __init__(self, imsize1, imsize2, batchNorm=True):
        super(DeepVO,self).__init__()
        # CNN
        self.batchNorm = batchNorm
        self.clip = par.clip
        self.conv1   = conv(self.batchNorm,   6,   64, kernel_size=7, stride=2, dropout=par.conv_dropout[0])
        self.conv2   = conv(self.batchNorm,  64,  128, kernel_size=5, stride=2, dropout=par.conv_dropout[1])
        self.conv3   = conv(self.batchNorm, 128,  256, kernel_size=5, stride=2, dropout=par.conv_dropout[2])
        self.conv3_1 = conv(self.batchNorm, 256,  256, kernel_size=3, stride=1, dropout=par.conv_dropout[3])
        self.conv4   = conv(self.batchNorm, 256,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[4])
        self.conv4_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[5])
        self.conv5   = conv(self.batchNorm, 512,  512, kernel_size=3, stride=2, dropout=par.conv_dropout[6])
        self.conv5_1 = conv(self.batchNorm, 512,  512, kernel_size=3, stride=1, dropout=par.conv_dropout[7])
        self.conv6   = conv(self.batchNorm, 512, 1024, kernel_size=3, stride=2, dropout=par.conv_dropout[8])
        # Comput the shape based on diff image size
        __tmp = Variable(torch.zeros(1, 6, imsize1, imsize2))
        __tmp = self.encode_image(__tmp)

        # RNN
        self.rnn = nn.LSTM(
                    input_size=int(np.prod(__tmp.size())), 
                    hidden_size=par.rnn_hidden_size, 
                    num_layers=2, 
                    dropout=par.rnn_dropout_between, 
                    batch_first=True)
        self.rnn_drop_out = nn.Dropout(par.rnn_dropout_out)
        self.linear = nn.Linear(in_features=par.rnn_hidden_size, out_features=6)

        # Initilization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Linear):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.LSTM):
                # layer 1
                kaiming_normal_(m.weight_ih_l0)  #orthogonal_(m.weight_ih_l0)
                kaiming_normal_(m.weight_hh_l0)
                m.bias_ih_l0.data.zero_()
                m.bias_hh_l0.data.zero_()
                # Set forget gate bias to 1 (remember)
                n = m.bias_hh_l0.size(0)
                start, end = n//4, n//2
                m.bias_hh_l0.data[start:end].fill_(1.)

                # layer 2
                kaiming_normal_(m.weight_ih_l1)  #orthogonal_(m.weight_ih_l1)
                kaiming_normal_(m.weight_hh_l1)
                m.bias_ih_l1.data.zero_()
                m.bias_hh_l1.data.zero_()
                n = m.bias_hh_l1.size(0)
                start, end = n//4, n//2
                m.bias_hh_l1.data[start:end].fill_(1.)

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


    def forward(self, x): 
        # x: (batch, seq_len, channel, width, height)
        # stack_image
        x = torch.cat(( x[:, :-1], x[:, 1:]), dim=2)
        batch_size = x.size(0)
        seq_len = x.size(1)
        # CNN
        x = x.view(batch_size*seq_len, x.size(2), x.size(3), x.size(4))
        x = self.encode_image(x)
        x = x.view(batch_size, seq_len, -1)


        # RNN
        out, hc = self.rnn(x)
        out = self.rnn_drop_out(out)
        out = self.linear(out)
        return out
        

    def encode_image(self, x):
        out_conv2 = self.conv2(self.conv1(x))
        out_conv3 = self.conv3_1(self.conv3(out_conv2))
        out_conv4 = self.conv4_1(self.conv4(out_conv3))
        out_conv5 = self.conv5_1(self.conv5(out_conv4))
        out_conv6 = self.conv6(out_conv5)
        return out_conv6

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]

    def get_loss(self, x, y):
        predicted = self.forward(x)
        y = y[:, 1:, :]  # (batch, seq, dim_pose) 避開第一章，因為第一個pose為基本已知
        # Weighted MSE Loss
        angle_loss = torch.nn.functional.mse_loss(predicted[:,:,:3], y[:,:,:3])
        translation_loss = torch.nn.functional.mse_loss(predicted[:,:,3:], y[:,:,3:])
        loss = (100 * angle_loss + translation_loss)
        return loss

    def step(self, x, y, optimizer):
        optimizer.zero_grad()
        loss = self.get_loss(x, y)
        loss.backward()
        if self.clip != None:
            torch.nn.utils.clip_grad_norm(self.rnn.parameters(), self.clip)
        optimizer.step()
        return loss