import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from torch.autograd import Variable


class conv_block(nn.Module):
    def __init__(self, t_size, n_step):
        """

        :param t_size: the size of input channel and output channel
        :type t_size: int
        :param n_step: input shape from an expected input of size
        :type n_step: int
        """
        super(conv_block, self).__init__()
        self.cov1d_1 = nn.Conv2d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov1d_2 = nn.Conv2d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov1d_3 = nn.Conv2d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.norm_3 = nn.LayerNorm(n_step)
        self.relu_1 = nn.ReLU()
        self.relu_2 = nn.ReLU()
        self.relu_3 = nn.ReLU()


    def forward(self, x):
        x_input = x
        out = self.cov1d_1(x)
        out = self.relu_1(out)
        out = self.cov1d_2(out)
        out = self.relu_2(out)
        out = self.cov1d_3(out)
        out = self.norm_3(out)
        out = self.relu_3(out)
        # Add the input to the output
        out = out.add(x_input)

        return out


class identity_block(nn.Module):
    def __init__(self, t_size, n_step):
        """

        :param t_size: the size of input channel and output channel
        :type t_size: int
        :param n_step: input shape from an expected input of size
        :type n_step: int
        """
        super(identity_block, self).__init__()
        self.cov1d_1 = nn.Conv2d(t_size, t_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.norm_1 = nn.LayerNorm(n_step)
        self.relu = nn.ReLU()

    def forward(self, x):

        out = self.cov1d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)

        return out


class Encoder(nn.Module):
    def __init__(self,
                 original_step_size,
                 pool_list,
                 conv_size,
                 transform_type='rst',
                 scale_limit = 1.0,
                 translation_limit = 1.0
                 ):
        """

        :param original_step_size: the image size trained in the model
        :type original_step_size: list of int
        :param pool_list: the parameter of each maxpool layer, the length of list decides the number of pool layers
        :type pool_list: list of int
        :param conv_size: the number of filters for each convolutional layer
        :type conv_size: int
        :param transform_type: choose the type of affine transformation,
                                'rst': rotation, scale, translation;
                                'rs': rotation, scale
                                'rt': rotation, translation
                                'combine': a compound matrix that includes every possible affine transformation
        :type transform_type: string
        :param scale_limit: set the limitation of the scale parameters ,usually between (0, 1)
        :type scale_limit: float
        :param translation_limit: set the limitation of the translation parameters ,usually between (0, 1)
        :type translation_limit: float

        """
        super(Encoder, self).__init__()

        blocks = []
        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        number_of_blocks = len(pool_list)
        blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(nn.MaxPool2d(pool_list[0], stride=pool_list[0]))
        for i in range(1, number_of_blocks):
            original_step_size = [original_step_size[0] // pool_list[i - 1], original_step_size[1] // pool_list[i - 1]]
            blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(nn.MaxPool2d(pool_list[i], stride=pool_list[i]))

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)
        original_step_size = [original_step_size[0] // pool_list[-1], original_step_size[1] // pool_list[-1]]

        input_size = original_step_size[0] * original_step_size[1]
        self.scale_limit = scale_limit
        self.translation_limit = translation_limit
        self.cov2d = nn.Conv2d(1, conv_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov2d_1 = nn.Conv2d(conv_size, 1, 3, stride=1, padding=1, padding_mode='zeros')

        if transform_type=='combine':
            embedding_size = 6
        elif transform_type=='rst':
            embedding_size = 5
        elif transform_type=='rs' or transform_type=='rt':
            embedding_size = 3
        else:
            raise Exception('the type of affine transformation is invalid, the valid type are: "combine","rst","rs","rt".')

        self.before = nn.Linear(input_size, 20)
        self.dense = nn.Linear(20, embedding_size)

    def forward(self, x):

        out = x.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        kout = self.before(out)
        out = self.dense(kout)

        if transform_type=='combine':
            theta = out.view(-1,2,3)
            multi_mtx = F.affine_grid(theta.to(device), x.size()).to(device)
            output = F.grid_sample(x, multi_mtx)

            return output, kout, theta

        else:
            rotate = out[:,0]
            a_1 = torch.cos(rotate)
            a_2 = torch.sin(rotate)
            a_4 = torch.ones(rotate.shape).to(device)
            a_5 = rotate * 0
            b1 = torch.stack((a_1, a_2), dim=1).squeeze()
            b2 = torch.stack((-a_2, a_1), dim=1).squeeze()
            b3 = torch.stack((a_5, a_5), dim=1).squeeze()
            rotation = torch.stack((b1, b2, b3), dim=2)
            grid_1 = F.affine_grid(rotation.to(device), x.size()).to(device)
            out_r = F.grid_sample(x, grid_1)

            if transform_type=='rs':
                scale_1 = self.scale_limit * nn.Tanh()(out[:, 1]) + 1
                scale_2 = self.scale_limit * nn.Tanh()(out[:, 2]) + 1
                c1 = torch.stack((scale_1, a_5), dim=1).squeeze()
                c2 = torch.stack((a_5, scale_2), dim=1).squeeze()
                c3 = torch.stack((a_5, a_5), dim=1).squeeze()
                scaler = torch.stack((c1, c2, c3), dim=2)
                grid_2 = F.affine_grid(scaler.to(device), x.size()).to(device)
                out_s = F.grid_sample(out_r, grid_2)

                return out_s, kout, rotation, scaler

            elif transform_type=='rt':
                trans_1 = self.translation_limit*nn.Tanh()(out[:,1])
                trans_2 = self.translation_limit*nn.Tanh()(out[:,2])
                d1 = torch.stack((a_4,a_5), dim=1).squeeze()
                d2 = torch.stack((a_5,a_4), dim=1).squeeze()
                d3 = torch.stack((trans_1,trans_2), dim=1).squeeze()
                translation = torch.stack((d1, d2, d3), dim=2)
                grid_2 = F.affine_grid(translation.to(device), x.size()).to(device)
                out_t = F.grid_sample(out_r, grid_2)

                return out_t, kout, rotation, translation

            elif transform_type=='rst':
                scale_1 = self.scale_limit * nn.Tanh()(out[:, 1]) + 1
                scale_2 = self.scale_limit * nn.Tanh()(out[:, 2]) + 1
                trans_1 = self.translation_limit * nn.Tanh()(out[:, 3])
                trans_2 = self.translation_limit * nn.Tanh()(out[:, 4])

                c1 = torch.stack((scale_1, a_5), dim=1).squeeze()
                c2 = torch.stack((a_5, scale_2), dim=1).squeeze()
                c3 = torch.stack((a_5, a_5), dim=1).squeeze()
                scaler = torch.stack((c1, c2, c3), dim=2)

                d1 = torch.stack((a_4, a_5), dim=1).squeeze()
                d2 = torch.stack((a_5, a_4), dim=1).squeeze()
                d3 = torch.stack((trans_1, trans_2), dim=1).squeeze()
                translation = torch.stack((d1, d2, d3), dim=2)

                grid_2 = F.affine_grid(scaler.to(device), x.size()).to(device)
                out_s = F.grid_sample(out_r, grid_2)
                grid_3 = F.affine_grid(translation.to(device), x.size()).to(device)
                out_t = F.grid_sample(out_s, grid_3)

                return out_t, kout, rotation, scaler, translation

            else:
                raise Exception(
                    'the type of affine transformation is invalid, the valid type are: "combine","rst","rs","rt".')





class Decoder(nn.Module):
    def __init__(self,
                 original_step_size,
                 up_list,
                 conv_size):
        """

        :param original_step_size:
        :type original_step_size:
        :param up_list:
        :type up_list:
        :param conv_size:
        :type conv_size:
        """
        super(Decoder, self).__init__()
        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        self.dense = nn.Linear(2, original_step_size[0] * original_step_size[1])
        self.cov2d = nn.Conv2d(1, conv_size, 3, stride=1, padding=1, padding_mode='zeros')
        self.cov2d_1 = nn.Conv2d(conv_size, 1, 3, stride=1, padding=1, padding_mode='zeros')

        blocks = []
        number_of_blocks = len(up_list)
        blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
        blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))
        for i in range(number_of_blocks):
            blocks.append(nn.Upsample(scale_factor=up_list[i], mode='bilinear', align_corners=True))
            original_step_size = [original_step_size[0] * up_list[i], original_step_size[1] * up_list[i]]
            blocks.append(conv_block(t_size=conv_size, n_step=original_step_size))
            blocks.append(identity_block(t_size=conv_size, n_step=original_step_size))

        self.block_layer = nn.ModuleList(blocks)
        self.layers = len(blocks)
        self.output_size_0 = original_step_size[0]
        self.output_size_1 = original_step_size[1]

        self.relu_1 = nn.ReLU()
        self.norm = nn.LayerNorm(2)
        self.softmax = nn.Softmax()
        self.for_k = nn.Linear(20, 2)
        self.num_k_sparse = 1

    # generate k sparsity algorithm to make every channel binary
    def ktop(self, x):
        kout = self.for_k(x)
        kout = self.norm(kout)
        kout = self.softmax(kout)
        k_no = kout.clone()

        k = self.num_k_sparse
        with torch.no_grad():
            if k <= kout.shape[1]:
                for raw in k_no:
                    indices = torch.topk(raw, k)[1].to(device)
                    mask = torch.ones(raw.shape, dtype=bool).to(device)
                    mask[indices] = False
                    raw[mask] = 0
                    raw[~mask] = 1
        return k_no


    def forward(self, x):
        k_out = self.ktop(x)
        out = self.dense(k_out)
        out = out.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = self.relu_1(out)

        #        out = out.view()
        #        out = self.softmax(out)

        return out, k_out


class Joint(nn.Module):
    def __init__(self, encoder, decoder):
        super(Joint, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        #       print(x.shape)
        predicted, kout, rotation, scaler = self.encoder(x)

        identity = torch.tensor([0, 0, 1], dtype=torch.float).reshape(1, 1, 3).repeat(x.shape[0], 1, 1).to(device)

        new_theta_1 = torch.cat((rotation, identity), axis=1).to(device)
        new_theta_2 = torch.cat((scaler, identity), axis=1).to(device)
        #        new_theta_3 = torch.cat((translation,identity),axis=1).to(device)

        inver_theta_1 = torch.linalg.inv(new_theta_1)[:, 0:2].to(device)
        inver_theta_2 = torch.linalg.inv(new_theta_2)[:, 0:2].to(device)
        #        inver_theta_3 = torch.linalg.inv(new_theta_3)[:,0:2].to(device)

        grid_1 = F.affine_grid(inver_theta_1.to(device), x.size()).to(device)
        grid_2 = F.affine_grid(inver_theta_2.to(device), x.size()).to(device)
        #        grid_3 = F.affine_grid(inver_theta_3.to(device), x.size()).to(device)

        predicted_base, k_out = self.decoder(kout)

        #        predicted_t = F.grid_sample(predicted_base, grid_3)
        predicted_s = F.grid_sample(predicted_base, grid_2)
        predicted_input = F.grid_sample(predicted_s, grid_1)

        return predicted, predicted_base, predicted_input, k_out, rotation, scaler