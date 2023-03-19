import torch.nn as nn
import torch.nn.functional as F
import torch



########################################################################################################################################################################
########################################################################################################################################################################
################################################# U_net_spec2
########################################################################################################################################################################
########################################################################################################################################################################
class U_net_spec(nn.Module):
    def __init__(self,p,N):
        super(U_net_spec, self).__init__()
        self.N=N
        self.p = p
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels = 40, out_channels=self.N*8,kernel_size=3,padding="same"),
            nn.BatchNorm2d(self.N*8),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.N*8, out_channels=self.N*8,kernel_size=3,padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(self.N*8),
            nn.ReLU()
         )


        self.Encoder = nn.ModuleList()
        Encoder_input  = (self.N*8,self.N*16,self.N*32)
        Encoder_output = (self.N*16,self.N*32,self.N*64)
        for idx in range(len(Encoder_output)):
            layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels = Encoder_input[idx], out_channels=Encoder_output[idx],kernel_size=3,padding="same"),
                nn.Dropout(p=self.p),
                nn.ReLU(),
                nn.BatchNorm2d(Encoder_output[idx]),
                nn.Conv2d(in_channels = Encoder_output[idx], out_channels=Encoder_output[idx],kernel_size=3,padding="same"),
                nn.Dropout(p=self.p),
                nn.ReLU(),
                nn.BatchNorm2d(Encoder_output[idx])
             )
            self.Encoder.append(layer)

        self.lent_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels = Encoder_output[-1], out_channels=Encoder_output[-1],kernel_size=3,padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(Encoder_output[-1]),
            nn.ReLU(),
            nn.Conv2d(in_channels =Encoder_output[-1], out_channels=Encoder_output[-1],kernel_size=3,padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(Encoder_output[-1]),
            nn.ReLU(),
            # nn.Conv2d(in_channels = Encoder_output[-1], out_channels=Encoder_output[-1],kernel_size=3,padding="same"),
            # nn.BatchNorm2d(Encoder_output[-1]),
            # nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )


        self.Decoder = nn.ModuleList()
        Decoder_input  = (self.N*64,self.N*32,self.N*16)
        Decoder_output = (self.N*32,self.N*16,self.N*8)
        for idx in range(len(Decoder_output)):
            layer = nn.Sequential(
                nn.Conv2d(in_channels = 2*Decoder_input[idx], out_channels=Decoder_output[idx],kernel_size=3,padding="same"),
                nn.Dropout(p=self.p),
                nn.ReLU(),
                nn.BatchNorm2d(Decoder_output[idx]),
                nn.Conv2d(in_channels = Decoder_output[idx], out_channels=Decoder_output[idx],kernel_size=3,padding="same"),
                nn.Dropout(p=self.p),
                nn.ReLU(),
                nn.BatchNorm2d(Decoder_output[idx]),
                nn.Upsample(scale_factor=2),
             )
            self.Decoder.append(layer)

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels = Decoder_output[-1]+Encoder_input[0], out_channels=3,kernel_size=3,padding="same"),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels = 3, out_channels=3,kernel_size=3,padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels = 3, out_channels=3,kernel_size=7,padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels = 3, out_channels=1,kernel_size=7,padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
         )

    def forward(self, x):
        input = self.input_layer(x)
        EN0 = self.Encoder[0](input)
        EN1 = self.Encoder[1](EN0)
        EN2 = self.Encoder[2](EN1)
        lent = self.lent_layer(EN2)
        DE0 = self.Decoder[0](torch.cat((EN2, lent), 1))
        DE1 = self.Decoder[1](torch.cat((EN1, DE0),  1))
        DE2 = self.Decoder[2](torch.cat((EN0, DE1),  1))
        output = self.output_layer(torch.cat((input, DE2),  1))
        return output




########################################################################################################################################################################
########################################################################################################################################################################
################################################# U_net_RGB
########################################################################################################################################################################
########################################################################################################################################################################
class U_net_RGB(nn.Module):
    def __init__(self,p,N):
        super(U_net_RGB, self).__init__()

        self.N = N
        self.p = p
        self.input_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=self.N * 8, kernel_size=3, padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(self.N * 8),
            nn.ReLU(),
            nn.Conv2d(in_channels=self.N * 8, out_channels=self.N * 8, kernel_size=3, padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(self.N * 8),
            nn.ReLU()
        )

        self.Encoder = nn.ModuleList()
        Encoder_input = (self.N * 8, self.N * 16, self.N * 32)
        Encoder_output = (self.N * 16, self.N * 32, self.N * 64)
        for idx in range(len(Encoder_output)):
            layer = nn.Sequential(
                nn.MaxPool2d(kernel_size=2),
                nn.Conv2d(in_channels=Encoder_input[idx], out_channels=Encoder_output[idx], kernel_size=3,
                          padding="same"),
                nn.Dropout(p=self.p),
                nn.ReLU(),
                nn.BatchNorm2d(Encoder_output[idx]),
                nn.Conv2d(in_channels=Encoder_output[idx], out_channels=Encoder_output[idx], kernel_size=3,
                          padding="same"),
                nn.Dropout(p=self.p),
                nn.ReLU(),
                nn.BatchNorm2d(Encoder_output[idx])
            )
            self.Encoder.append(layer)

        self.lent_layer = nn.Sequential(
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=Encoder_output[-1], out_channels=Encoder_output[-1], kernel_size=3, padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(Encoder_output[-1]),
            nn.ReLU(),
            nn.Conv2d(in_channels=Encoder_output[-1], out_channels=Encoder_output[-1], kernel_size=3, padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(Encoder_output[-1]),
            nn.ReLU(),
            # nn.Conv2d(in_channels = Encoder_output[-1], out_channels=Encoder_output[-1],kernel_size=3,padding="same"),
            # nn.BatchNorm2d(Encoder_output[-1]),
            # nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )

        self.Decoder = nn.ModuleList()
        Decoder_input = (self.N * 64, self.N * 32, self.N * 16)
        Decoder_output = (self.N * 32, self.N * 16, self.N * 8)
        for idx in range(len(Decoder_output)):
            layer = nn.Sequential(
                nn.Conv2d(in_channels=2 * Decoder_input[idx], out_channels=Decoder_output[idx], kernel_size=3,
                          padding="same"),
                nn.Dropout(p=self.p),
                nn.ReLU(),
                nn.BatchNorm2d(Decoder_output[idx]),
                nn.Conv2d(in_channels=Decoder_output[idx], out_channels=Decoder_output[idx], kernel_size=3,
                          padding="same"),
                nn.Dropout(p=self.p),
                nn.ReLU(),
                nn.BatchNorm2d(Decoder_output[idx]),
                nn.Upsample(scale_factor=2),
            )
            self.Decoder.append(layer)

        self.output_layer = nn.Sequential(
            nn.Conv2d(in_channels=Decoder_output[-1] + Encoder_input[0], out_channels=3, kernel_size=3, padding="same"),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=3, padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=3, kernel_size=7, padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Conv2d(in_channels=3, out_channels=1, kernel_size=7, padding="same"),
            nn.Dropout(p=self.p),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        input = self.input_layer(x)
        EN0 = self.Encoder[0](input)
        EN1 = self.Encoder[1](EN0)
        EN2 = self.Encoder[2](EN1)
        lent = self.lent_layer(EN2)
        DE0 = self.Decoder[0](torch.cat((EN2, lent), 1))
        DE1 = self.Decoder[1](torch.cat((EN1, DE0), 1))
        DE2 = self.Decoder[2](torch.cat((EN0, DE1), 1))
        output = self.output_layer(torch.cat((input, DE2), 1))
        return output


########################################################################################################################################################################
########################################################################################################################################################################
#################################################
################################################# Dice BCE Loss
########################################################################################################################################################################
########################################################################################################################################################################
class DiceBCELoss(nn.Module):
    def __init__(self,Lambda=1, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.Lambda = Lambda

    def forward(self, inputs, targets, smooth=1):
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = F.sigmoid(inputs)

        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        # weights = weights.view(-1)

        intersection = (inputs * targets).sum()
        dice_loss = 1 - (intersection + smooth)/(inputs.sum() + targets.sum() + smooth)
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        Dice_BCE = BCE + (1/self.Lambda)*dice_loss

        return Dice_BCE

class binary_Loss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(binary_Loss, self).__init__()

    def forward(self, inputs, targets,weights, smooth=1):
        return (-weights*(targets*torch.log(1+inputs)+(1-targets)*torch.log(2-inputs))).mean()