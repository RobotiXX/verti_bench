#!/usr/bin/env python3

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
import torch.nn.init as init

# Model definition
class ElevMapEncDec(nn.Module):
    def __init__(self):
        super(ElevMapEncDec, self).__init__()
        self.map_encode = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1, bias=True, dilation=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True, dilation=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.Conv2d(in_channels=16, out_channels=18, kernel_size=5, stride=2, padding=1, bias=True, dilation=1),
            )
        nn.init.kaiming_normal_(self.map_encode[0].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_encode[4].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_encode[7].weight, nonlinearity='leaky_relu', mode='fan_in')

        self.map_decode = nn.Sequential(
            # nn.AdaptiveAvgPool2d((89, 89)),
            nn.ConvTranspose2d(in_channels=18, out_channels=16, kernel_size=5, stride=2, padding=1, bias=True, dilation=1),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(16),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=5, stride=2, padding=1, bias=True, dilation=1),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=2, stride=2),
            nn.LeakyReLU(0.1),
            nn.BatchNorm2d(8),
            nn.ConvTranspose2d(in_channels=8, out_channels=1, kernel_size=3, stride=1, padding=0, bias=True, dilation=1),
        )

        nn.init.kaiming_normal_(self.map_decode[0].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_decode[3].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_decode[7].weight, nonlinearity='leaky_relu', mode='fan_in')

    def forward(self, elev_map):
        elev_map1 = self.map_encode(elev_map)
        elev_map2 = self.map_decode(elev_map1)
        return elev_map2.squeeze()

class PatchDecoder(nn.Module):
    def __init__(self, map_encoder):
        super().__init__()
        self.map_encoder = map_encoder # 18 * 44 * 44

        self.map_conv = nn.Sequential(
            nn.Conv2d(in_channels=18, out_channels=64, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1),
            nn.Conv2d(in_channels=128, out_channels=160, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(160),
            nn.LeakyReLU(0.1),
        )
        nn.init.kaiming_normal_(self.map_conv[0].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_conv[3].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_conv[6].weight, nonlinearity='leaky_relu', mode='fan_in')

        self.offset_ecoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.LayerNorm(64),
            nn.Sigmoid(),
            nn.Linear(64, 160*6*6),
            nn.LayerNorm(160*6*6),
            nn.Sigmoid(),
        )
        nn.init.xavier_normal_(self.offset_ecoder[0].weight)
        nn.init.xavier_normal_(self.offset_ecoder[3].weight)

        self.glu = nn.GLU()

        self.fusion = nn.Sequential(
            nn.Linear(160*6*6, 64*6*6),
            nn.LayerNorm(64*6*6),
            nn.LeakyReLU(0.1),
        )
        nn.init.xavier_normal_(self.fusion[0].weight)

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.1),
            nn.ConvTranspose2d(in_channels=16, out_channels=1, kernel_size=3, stride=2, padding=0, bias=False),
            nn.LeakyReLU(0.01),
        )
        nn.init.kaiming_normal_(self.decoder[0].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.decoder[3].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.decoder[6].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.decoder[9].weight, nonlinearity='leaky_relu', mode='fan_in')



    def forward(self, elev_map, map_offset):
        elev_map = self.map_encoder(elev_map)
        elev_map = self.map_conv(elev_map)
        map_offset = self.offset_ecoder(map_offset)
        # elev_map_aggr = torch.cat((elev_map.flatten(1), map_offset), dim=1)
        
        elev_map_aggr = elev_map * map_offset.view(-1, 160, 6, 6)
        elev_map = self.fusion(elev_map_aggr.flatten(start_dim=1))
        elev_map = elev_map.view(-1, 64, 6, 6)
        
        elev_map = self.decoder(elev_map)#.flatten(start_dim=1)
        # print(f"{elev_map.shape = }")
        #elev_map = self.shape_shifter(elev_map).view(-1, 40, 100)
        return elev_map
    
    def get_tal_embedding(self, elev_map):
        elev_map = self.map_encoder(elev_map)
        elev_map = self.map_conv(elev_map)
        return elev_map

    def decode(self, elev_map, map_offset):
        map_offset = self.offset_ecoder(map_offset)
        elev_map_aggr = elev_map * map_offset.view(-1, 160, 6, 6)
        elev_map = self.fusion(elev_map_aggr.flatten(start_dim=1))
        return elev_map
        
class TAL(nn.Module):
    def __init__(self, elev_map_encoder, elev_map_decoder):
        super(TAL, self).__init__()
        
        for param in elev_map_encoder.parameters():
            param.requires_grad = False
        
        self.map_encoder = elev_map_encoder

        for param in elev_map_decoder.parameters():
            param.requires_grad = False
        
        self.map_decoder = elev_map_decoder

        self.map_dropout = nn.Dropout(0.3)

        self.map_process_fc = nn.Sequential(
            # patch size is 45 x 109 = 4905
            nn.Linear(2304, 128),
            nn.LayerNorm(128),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(128, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        init.xavier_uniform_(self.map_process_fc[0].weight)
        init.xavier_uniform_(self.map_process_fc[4].weight)

        self.state_process = nn.Sequential(
            nn.Linear(6, 8),
            nn.LayerNorm(8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        init.xavier_uniform_(self.state_process[0].weight)
        init.xavier_uniform_(self.state_process[3].weight)
        
        self.cmd_process = nn.Sequential(
            nn.Linear(2, 8),
            nn.LayerNorm(8),
            nn.Tanh(),
            nn.Linear(8, 16),
            nn.Tanh(),
        )

        init.xavier_uniform_(self.cmd_process[0].weight)
        init.xavier_uniform_(self.cmd_process[3].weight)

        self.fc = nn.Sequential(
            nn.Linear(48, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Linear(16, 4),
        )
        init.xavier_uniform_(self.fc[0].weight)
        init.xavier_uniform_(self.fc[3].weight)

        self.fc2 = nn.Sequential(
            nn.Linear(48, 16),
            nn.LayerNorm(16),
            nn.Tanh(),
            nn.Linear(16, 2),
        )

    def forward(self, pose_dot, cmd_vel, map_offset, elev_map):
        elev_map = self.map_encoder(elev_map)#.flatten(1)
        elev_map = self.map_decoder.decode(elev_map, map_offset)#.flatten(1)
        
        elev_map = self.map_process_fc(elev_map)
        pose_dot = self.state_process(pose_dot)
        cmd_vel = self.cmd_process(cmd_vel)
        output = torch.zeros((pose_dot.shape[0], 6), device=pose_dot.device, dtype=pose_dot.dtype)
        output[:, [0,1,2,5]] = self.fc( torch.cat((pose_dot, cmd_vel, elev_map), dim=1) )
        x = torch.cat((self.map_dropout(pose_dot), self.map_dropout(cmd_vel), elev_map), dim=1)
        output[:, [3,4]] = self.fc2( x )

        return output


    def predict(self, pose_dot, cmd_vel, map_offset, elev_map):
        elev_map = self.map_decoder.decode(elev_map, map_offset)#.flatten(1)
        elev_map = self.map_process_fc(elev_map)
        pose_dot = self.state_process(pose_dot)
        cmd_vel = self.cmd_process(cmd_vel)
        output = torch.zeros((pose_dot.shape[0], 6), device=pose_dot.device, dtype=pose_dot.dtype)
        output[:, [0,1,2,5]] = self.fc( torch.cat((pose_dot, cmd_vel, elev_map), dim=1) )
        x = torch.cat((self.map_dropout(pose_dot), self.map_dropout(cmd_vel), elev_map), dim=1)
        output[:, [3,4]] = self.fc2( x )

        return output
        
    def process_map(self, elev_map):
        elev_map = self.map_decoder.get_tal_embedding(elev_map)
        return elev_map
        
class wmvct_alt(nn.Module):
    def __init__(self):
        super(wmvct_alt, self).__init__()
        
        self.map_encode = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1, bias=True, dilation=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=1, padding=1, bias=True, dilation=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=8, out_channels=12, kernel_size=5, stride=2, padding=1, bias=True, dilation=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1, bias=True, dilation=1),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(in_channels=24, out_channels=32, kernel_size=5, stride=2, padding=1, bias=True, dilation=1),
            nn.LeakyReLU(0.1),
            nn.AdaptiveAvgPool2d((12, 12)),
            )
        nn.init.kaiming_normal_(self.map_encode[0].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_encode[3].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_encode[6].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_encode[9].weight, nonlinearity='leaky_relu', mode='fan_in')
        nn.init.kaiming_normal_(self.map_encode[12].weight, nonlinearity='leaky_relu', mode='fan_in')

        self.offset_ecoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32*12*12),
            nn.Sigmoid(),
        )
        nn.init.xavier_normal_(self.offset_ecoder[0].weight)
        nn.init.xavier_normal_(self.offset_ecoder[2].weight)

        self.map_process_fc = nn.Sequential(
            # patch size is 45 x 109 = 4905
            nn.Linear(32*12*12*2, 64),
            nn.LayerNorm(64),
            nn.Tanh(),
            nn.Dropout(0.25),
            nn.Linear(64, 24),
            nn.LayerNorm(24),
            nn.Tanh(),
            nn.Dropout(0.1),
        )
        init.xavier_uniform_(self.map_process_fc[0].weight)
        init.xavier_uniform_(self.map_process_fc[4].weight)

        self.state_process = nn.Sequential(
            nn.Linear(6, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 24),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )
        nn.init.kaiming_normal_(self.state_process[0].weight, nonlinearity='leaky_relu', mode='fan_in' )
        nn.init.kaiming_normal_(self.state_process[3].weight, nonlinearity='leaky_relu', mode='fan_in' )
        
        self.cmd_process = nn.Sequential(
            nn.Linear(2, 8),
            nn.LayerNorm(8),
            nn.LeakyReLU(0.1),
            nn.Linear(8, 24),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),
        )
        nn.init.kaiming_normal_(self.cmd_process[0].weight, nonlinearity='leaky_relu', mode='fan_in' )
        nn.init.kaiming_normal_(self.cmd_process[3].weight, nonlinearity='leaky_relu', mode='fan_in' )

        self.fc = nn.Sequential(
            nn.Linear(24*3, 12),
            nn.LeakyReLU(0.1),
            nn.Linear(12, 6),
        )
        nn.init.kaiming_normal_(self.fc[0].weight, nonlinearity='leaky_relu', mode='fan_in'  )
        nn.init.kaiming_normal_(self.fc[2].weight, nonlinearity='leaky_relu', mode='fan_in'  )

    def forward(self, pose_dot, cmd_vel, map_offset, elev_map):
        elev_map = self.map_encode(elev_map).flatten(1)
        map_offset = self.offset_ecoder(map_offset)
        elev_map = torch.cat((elev_map, map_offset), dim=1)
        
        elev_map = self.map_process_fc(elev_map)
        pose_dot = self.state_process(pose_dot)
        cmd_vel = self.cmd_process(cmd_vel)
        output = self.fc( torch.cat((pose_dot, cmd_vel, elev_map), dim=1) )

        return output
    
    def process_map(self, elev_map):
        elev_map = self.map_encode(elev_map).flatten(0)
        return elev_map
    
    def predict(self, pose_dot, cmd_vel, map_offset, elev_map):
        map_offset = self.offset_ecoder(map_offset)
        elev_map = torch.cat((elev_map, map_offset), dim=1)
        
        elev_map = self.map_process_fc(elev_map)
        pose_dot = self.state_process(pose_dot)
        cmd_vel = self.cmd_process(cmd_vel)
        output = self.fc( torch.cat((pose_dot, cmd_vel, elev_map), dim=1) )

        return output
    
# class wmvct_alt(nn.Module):
#     def __init__(self):
#         super(wmvct_alt, self).__init__()
        
#         self.map_encode = nn.Sequential(
#             nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, stride=2, padding=1, bias=True, dilation=1),
#             nn.LeakyReLU(0.1),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=4, out_channels=8, kernel_size=5, stride=2, padding=1, bias=True, dilation=1),
#             nn.LeakyReLU(0.1),
#             nn.MaxPool2d(kernel_size=2, stride=2),
#             nn.Conv2d(in_channels=8, out_channels=12, kernel_size=5, stride=2, padding=1, bias=True, dilation=1),
#             nn.LeakyReLU(0.1),
#             nn.AdaptiveAvgPool2d((6, 6)),
#             )
#         nn.init.kaiming_normal_(self.map_encode[0].weight, nonlinearity='leaky_relu', mode='fan_in')
#         nn.init.kaiming_normal_(self.map_encode[3].weight, nonlinearity='leaky_relu', mode='fan_in')
#         nn.init.kaiming_normal_(self.map_encode[6].weight, nonlinearity='leaky_relu', mode='fan_in')

#         self.offset_ecoder = nn.Sequential(
#             nn.Linear(4, 64),
#             nn.Sigmoid(),
#             nn.Linear(64, 12*6*6),
#             nn.Sigmoid(),
#         )
#         nn.init.xavier_normal_(self.offset_ecoder[0].weight)
#         nn.init.xavier_normal_(self.offset_ecoder[2].weight)

#         self.map_process_fc = nn.Sequential(
#             # patch size is 45 x 109 = 4905
#             nn.Linear(12*6*6*2, 64),
#             nn.LayerNorm(64),
#             nn.Tanh(),
#             nn.Dropout(0.25),
#             nn.Linear(64, 12),
#             nn.LayerNorm(12),
#             nn.Tanh(),
#             nn.Dropout(0.1),
#         )
#         init.xavier_uniform_(self.map_process_fc[0].weight)
#         init.xavier_uniform_(self.map_process_fc[4].weight)

#         self.state_process = nn.Sequential(
#             nn.Linear(6, 8),
#             nn.LayerNorm(8),
#             nn.LeakyReLU(0.1),
#             nn.Linear(8, 12),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.2),
#         )
#         nn.init.kaiming_normal_(self.state_process[0].weight, nonlinearity='leaky_relu', mode='fan_in' )
#         nn.init.kaiming_normal_(self.state_process[3].weight, nonlinearity='leaky_relu', mode='fan_in' )
        
#         self.cmd_process = nn.Sequential(
#             nn.Linear(2, 8),
#             nn.LayerNorm(8),
#             nn.LeakyReLU(0.1),
#             nn.Linear(8, 12),
#             nn.LeakyReLU(0.1),
#             nn.Dropout(0.2),
#         )
#         nn.init.kaiming_normal_(self.cmd_process[0].weight, nonlinearity='leaky_relu', mode='fan_in' )
#         nn.init.kaiming_normal_(self.cmd_process[3].weight, nonlinearity='leaky_relu', mode='fan_in' )

#         self.fc = nn.Sequential(
#             nn.Linear(36, 12),
#             nn.LeakyReLU(0.1),
#             nn.Linear(12, 6),
#         )
#         nn.init.kaiming_normal_(self.fc[0].weight, nonlinearity='leaky_relu', mode='fan_in'  )
#         nn.init.kaiming_normal_(self.fc[2].weight, nonlinearity='leaky_relu', mode='fan_in'  )

#     def forward(self, pose_dot, cmd_vel, map_offset, elev_map):
#         elev_map = self.map_encode(elev_map).flatten(1)
#         map_offset = self.offset_ecoder(map_offset)
#         elev_map = torch.cat((elev_map, map_offset), dim=1)
        
#         elev_map = self.map_process_fc(elev_map)
#         pose_dot = self.state_process(pose_dot)
#         cmd_vel = self.cmd_process(cmd_vel)
#         output = self.fc( torch.cat((pose_dot, cmd_vel, elev_map), dim=1) )

#         return output
    
#     def process_map(self, elev_map):
#         elev_map = self.map_encode(elev_map).flatten(0)
#         return elev_map
    
#     def predict(self, pose_dot, cmd_vel, map_offset, elev_map):
#         map_offset = self.offset_ecoder(map_offset)
#         elev_map = torch.cat((elev_map, map_offset), dim=1)
        
#         elev_map = self.map_process_fc(elev_map)
#         pose_dot = self.state_process(pose_dot)
#         cmd_vel = self.cmd_process(cmd_vel)
#         output = self.fc( torch.cat((pose_dot, cmd_vel, elev_map), dim=1) )

#         return output
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=2, padding=1):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU()
        nn.init.kaiming_normal_(self.conv.weight)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))
        
class TraversabilityMapPredictor(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, initial_features=32, depth=4):
        super(TraversabilityMapPredictor, self).__init__()
        
        # Encoder (Feature extractor)
        self.encoder = nn.ModuleList()
        in_channels = input_channels
        for i in range(depth):
            out_channels = initial_features * (2**i)
            self.encoder.append(ConvBlock(in_channels, out_channels, stride=2))
            in_channels = out_channels

        # Final layer: output 1 predicted traversability map
        self.fc = nn.Conv2d(in_channels, output_channels, kernel_size=1)

    def forward(self, x):
        # Forward pass through encoder
        for encoder_layer in self.encoder:
            x = encoder_layer(x)

        # Output 1 predicted traversability map
        x = self.fc(x)

        # Ensure output size matches the target size (320x260)
        x = F.interpolate(x, size=(320, 260), mode='bilinear', align_corners=False)
        
        return x
