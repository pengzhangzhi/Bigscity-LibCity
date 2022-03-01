'''=================================================

@Project -> File：ST-Transformer->STTransformer

@IDE：PyCharm

@coding: utf-8

@time:2021/7/23 17:01

@author:Pengzhangzhi

@Desc：
=================================================='''
from __future__ import absolute_import
import sys
sys.path.append('/home/ubuntu/Bigscity-LibCity/libcity/model/traffic_flow_prediction')
sys.path.append('/home/ubuntu/Bigscity-LibCity')
sys.path.append('/home/ubuntu/Bigscity-LibCity/libcity')

import os
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
import sys
import torch
import torch.nn as nn
from einops import rearrange, reduce
from torch.nn import init
from base_layers import BasicBlock
from vit import ViT
from logging import getLogger
from libcity.model import loss

print(sys.path)

class Rc(nn.Module):
    def __init__(self, input_shape):
        super(Rc, self).__init__()
        self.nb_flow = input_shape[0]
        self.ilayer = iLayer(input_shape)

    def forward(self, x):
        """
            x: (*, c, h, w)
          out: (*, 2, h ,w)
        """
        # x = rearrange(x,"b (nb_flow c) h w -> b nb_flow c h w",nb_flow=self.nb_flow)
        # x = reduce(x,"b nb_flow c h w -> b nb_flow h w","sum")
        x = reduce(x, "b (c1 c) h w -> b c1 h w", "sum", c1=self.nb_flow)
        out = self.ilayer(x)
        return out


class iLayer(nn.Module):
    '''    elementwise multiplication
    '''

    def __init__(self, input_shape):
        '''
        input_shape: (,*,c,,h,w)
        self.weights shape: (,*,c,h,w)
        '''
        super(iLayer, self).__init__()
        self.weights = nn.Parameter(torch.randn(*input_shape))  # define the trainable parameter
        init.xavier_uniform_(self.weights.data)

    def forward(self, x):
        '''
        x: (batch, c, h,w)
        self.weights shape: (c,h,w)
        output: (batch, c, h,w)
        '''
        return x * self.weights


class STTSNet(AbstractTrafficStateModel):
    def __init__(self,config, data_feature):
        """
        map_height=32, map_width=32, patch_size=4,
                 close_channels=6, trend_channels=6, close_dim=1024, trend_dim=1024,
                 close_depth=4, trend_depth=4, close_head=2,
                 trend_head=2, close_mlp_dim=2048, trend_mlp_dim=2048, nb_flow=2,
                 seq_pool=True,
                 pre_conv=True,
                 shortcut=True,
                 conv_channels=64,
                 drop_prob=0.1,
                 conv3d=False,
                 **kwargs

        :param seq_pool: whether to use sequence pooling.
        :param pre_conv: whether to use pre-conv
        :param conv_channels: number of channels inside pre_conv.
        :param map_height:
        :param map_width:
        :param patch_size:
        :param close_channels: number of channels in Xc,
        :param trend_channels: number of channels in Xc,
        :param close_dim: embedding dimension of closeness component.
        :param trend_dim: embedding dimension of trend component.
        :param close_depth: number of transformer in closeness component
        :param trend_depth: number of transformer in trend component
        :param close_head: number of head in closeness component
        :param trend_head: number of head in trend component
        :param close_mlp_dim: embedding dimension of a head in closeness component
        :param trend_mlp_dim: embedding dimension of a head in trend component
        :param nb_flow: number of flow.
        :param kwargs: filter out useless args.
        """
        super().__init__(config, data_feature)
        self._scaler = data_feature.get('scaler')
        self._logger = getLogger()
        self.map_height = data_feature.get("map_height",20)
        self.map_width = data_feature.get("map_width",10)
        self.nb_flow = data_feature.get("nb_flow",2)
        self.len_closeness = data_feature.get("len_closeness",6)
        self.len_trend = data_feature.get("len_trend",6)


        self.close_dim = config.get("close_dim",64)
        self.trend_dim = config.get("trend_dim",64)
        self.close_head = config.get("close_head",2)
        self.trend_head = config.get("trend_head",2)
        self.conv_channels = config.get("conv_channels",64)
        self.patch_size = config.get("patch_size",(10,10))
        self.close_channels = self.len_closeness * 2
        self.trend_channels = self.len_trend * 2
        self.close_depth = config.get("close_depth",2)
        self.trend_depth = config.get("trend_depth",2)
        self.close_mlp_dim = config.get("close_mlp_dim",128)
        self.trend_mlp_dim = config.get("trend_mlp_dim",128)
        output_dim = self.nb_flow * self.map_height * self.map_width
        close_dim_head = int(self.close_dim / self.close_head)
        trend_dim_head = int(self.trend_dim / self.trend_head)
        self.pre_conv = config.get("pre_conv",True)
        self.drop_prob = config.get("drop_prob",0.1)
        self.shortcut = config.get("short_cut",True)

        if self.pre_conv:
                self.pre_close_conv = nn.Sequential(
                    BasicBlock(inplanes=self.close_channels, planes=self.conv_channels),
                    # BasicBlock(inplanes=close_channels,planes=conv_channels),
                )
                self.pre_trend_conv = nn.Sequential(
                    BasicBlock(inplanes=self.trend_channels, planes=self.conv_channels),
                    # BasicBlock(inplanes=trend_channels,planes=conv_channels)
                )


        # close_channels, trend_channels = nb_flow * close_channels, nb_flow * trend_channels

        self.closeness_transformer = ViT(
            image_size=[self.map_height, self.map_width],
            patch_size=self.patch_size,
            num_classes=output_dim,
            dim=self.close_dim,
            depth=self.close_depth,
            heads=self.close_head,
            mlp_dim=self.close_mlp_dim,
            dropout=self.drop_prob,
            emb_dropout=self.drop_prob,
            channels=self.close_channels,
            dim_head=close_dim_head,
            seq_pool=False
        )
        self.trend_transformer = ViT(
            image_size=[self.map_height, self.map_width],
            patch_size=self.patch_size,
            num_classes=output_dim,
            dim=self.trend_dim,
            depth=self.trend_depth,
            heads=self.trend_head,
            mlp_dim=self.trend_mlp_dim,
            dropout=self.drop_prob,
            emb_dropout=self.drop_prob,
            channels=self.trend_channels,
            dim_head=trend_dim_head,
            seq_pool=False

        )
        input_shape = (self.nb_flow, self.map_height, self.map_width)

        if self.shortcut:
            self.Rc_Xc = Rc(input_shape)
            self.Rc_Xt = Rc(input_shape)
            # self.Rc_conv_Xc = Rc(input_shape)
            # self.Rc_conv_Xt = Rc(input_shape)

        self.close_ilayer = iLayer(input_shape=input_shape)
        self.trend_ilayer = iLayer(input_shape=input_shape)

        self.output_window = config.get('output_window', 1) # multi-step prediction.
        self.device = config.get('device', torch.device('cpu'))


    def forward(self, batch):
        """

        :param xc: batch size, num_close,map_height,map_width
        :param xt: batch size, num_week,map_height,map_width
        :param x_ext: No external data.
        :return:
                outputs: [batch size, output_window, nb_flow, height, width]
        """
        x = batch["X"].clone()
        x = rearrange(x,"b l h w n -> b n l h w")
        xc = x[:,:,:self.len_closeness,:,:]
        xt = x[:,:,-self.len_trend:,:,:]
        X_ext = batch["y_ext"].clone()
        
        if len(xc.shape) == 5:
            # reshape 5 dimensions to 4 dimensions.
            xc, xt = list(map(lambda x: rearrange(x, "b n l h w -> b (n l) h w"), [xc, xt]))
        batch_size = xc.shape[0]
        outputs = []
        for i in range(self.output_window):
            
            identity_xc, identity_xt = xc, xt
            if self.pre_conv:
                xc = self.pre_close_conv(xc)
                xt = self.pre_trend_conv(xt)

            close_out = self.closeness_transformer(xc)
            trend_out = self.trend_transformer(xt)

            # relu + linear

            close_out = close_out.reshape(batch_size, self.nb_flow, self.map_height, self.map_width)
            trend_out = trend_out.reshape(batch_size, self.nb_flow, self.map_height, self.map_width)

            close_out = self.close_ilayer(close_out)
            trend_out = self.trend_ilayer(trend_out)
            out = close_out + trend_out

            if self.shortcut:
                shortcut_out = self.Rc_Xc(identity_xc) + self.Rc_Xt(identity_xt)
                # +self.Rc_conv_Xc(xc_conv)+self.Rc_conv_Xt(xt_conv)
                out += shortcut_out

            if not self.training:
                out = out.relu()
            outputs.append(rearrange(out.clone(),"b n h w -> b h w n"))
            xc = torch.cat([xc[:,2:,:,:],out],dim=1)
        otuputs = torch.stack(outputs,dim=1)
        return otuputs
    def predict(self,batch):
        return self.forward(batch)

    def calculate_loss(self,batch):
        """
        this func is not tested.
        """
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true)
        y_predicted = self._scaler.inverse_transform(y_predicted)

        
        MSEloss = loss.masked_mse_torch(y_predicted, y_true, 0)
        print("Loss:",MSEloss)
        return MSEloss

if __name__ == '__main__':
    shape = (1, 12, 20, 10,2)
    # 1,12,32,32 -> 1,64,16*12
    x = torch.randn(shape)
    xc = torch.randn(shape)
    transformer = STTSNet(config={"output_window":5},data_feature={})
    batch = {
        "X": x ,
        "y_ext":xc,
    }
    pred = transformer(batch)
    print(pred.shape)
