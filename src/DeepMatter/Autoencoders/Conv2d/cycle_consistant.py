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
                 transform_type='combine',
                 scale_limit = 1.0,
                 translation_limit = 1.0,
                 dense_size=20,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        """

        :param original_step_size: the image size trained in the model
        :type original_step_size: list of int
        :param pool_list: the list of the parameters for each maxpool layer, the length of list decides the number of pool layers
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
        :param dense_size: set the output size of the data which goes to the decoder
        :type dense_size: int
        :param device: set the device where the model generated
        :type device: string ('cuda' or 'cpu)

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
        self.transform_type = transform_type
        self.device = device

        if transform_type=='combine':
            embedding_size = 6
        elif transform_type=='rst':
            embedding_size = 5
        elif transform_type=='rs' or transform_type=='rt':
            embedding_size = 3
        else:
            raise Exception('the type of affine transformation is invalid, the valid type are: "combine","rst","rs","rt".')

        self.before = nn.Linear(input_size, dense_size)
        self.dense = nn.Linear(dense_size, embedding_size)

    # generate the transform type, which is useful in the Joint() model
    def check_type(self):
        return self.transform_type

    def forward(self, x):

        out = x.view(-1, 1, self.input_size_0, self.input_size_1)
        out = self.cov2d(out)
        for i in range(self.layers):
            out = self.block_layer[i](out)
        out = self.cov2d_1(out)
        out = torch.flatten(out, start_dim=1)
        kout = self.before(out)
        out = self.dense(kout)

        if self.transform_type=='combine':
            theta = out.view(-1,2,3)
            multi_mtx = F.affine_grid(theta.to(self.device), x.size()).to(self.device)
            output = F.grid_sample(x, multi_mtx)

            return output, kout, theta

        else:
            rotate = out[:,0]
            a_1 = torch.cos(rotate)
            a_2 = torch.sin(rotate)
            a_4 = torch.ones(rotate.shape).to(self.device)
            a_5 = rotate * 0
            b1 = torch.stack((a_1, a_2), dim=1).squeeze()
            b2 = torch.stack((-a_2, a_1), dim=1).squeeze()
            b3 = torch.stack((a_5, a_5), dim=1).squeeze()
            rotation = torch.stack((b1, b2, b3), dim=2)
            grid_1 = F.affine_grid(rotation.to(self.device), x.size()).to(self.device)
            out_r = F.grid_sample(x, grid_1)

            if self.transform_type=='rs':
                scale_1 = self.scale_limit * nn.Tanh()(out[:, 1]) + 1
                scale_2 = self.scale_limit * nn.Tanh()(out[:, 2]) + 1
                c1 = torch.stack((scale_1, a_5), dim=1).squeeze()
                c2 = torch.stack((a_5, scale_2), dim=1).squeeze()
                c3 = torch.stack((a_5, a_5), dim=1).squeeze()
                scaler = torch.stack((c1, c2, c3), dim=2)
                grid_2 = F.affine_grid(scaler.to(self.device), x.size()).to(self.device)
                out_s = F.grid_sample(out_r, grid_2)

                return out_s, kout, rotation, scaler

            elif self.transform_type=='rt':
                trans_1 = self.translation_limit*nn.Tanh()(out[:,1])
                trans_2 = self.translation_limit*nn.Tanh()(out[:,2])
                d1 = torch.stack((a_4,a_5), dim=1).squeeze()
                d2 = torch.stack((a_5,a_4), dim=1).squeeze()
                d3 = torch.stack((trans_1,trans_2), dim=1).squeeze()
                translation = torch.stack((d1, d2, d3), dim=2)
                grid_2 = F.affine_grid(translation.to(self.device), x.size()).to(self.device)
                out_t = F.grid_sample(out_r, grid_2)

                return out_t, kout, rotation, translation

            elif self.transform_type=='rst':
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

                grid_2 = F.affine_grid(scaler.to(self.device), x.size()).to(self.device)
                out_s = F.grid_sample(out_r, grid_2)
                grid_3 = F.affine_grid(translation.to(self.device), x.size()).to(self.device)
                out_t = F.grid_sample(out_s, grid_3)

                return out_t, kout, rotation, scaler, translation

            else:
                raise Exception(
                    'the type of affine transformation is invalid, the valid type are: "combine","rst","rs","rt".')





class Decoder(nn.Module):
    def __init__(self,
                 original_step_size,
                 up_list,
                 conv_size,
                 base_size,
                 dense_size=20,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        """

        :param original_step_size: the size of the image which goes out from the K sparsity
        :type original_step_size: list of int (size should be square)
        :param up_list: the list of parameters for each up sample layer, the length of list decides the number of pool layers
        :type up_list: list of int (the paramters in list multipling the original_step_size should equal to the input
                        image size)
        :param conv_size: the number of filters for each convolutional layer
        :type conv_size: int
        :param base_size: the number of bases generated by the decoder
        :type base_size: int (>=1)
        :param dense_size: the feature size of data which goes to the K sparsity
        :type dense_size: int
        :param device: set the device where the model generated
        :type device: string ('cuda' or 'cpu)

        """
        super(Decoder, self).__init__()
        self.input_size_0 = original_step_size[0]
        self.input_size_1 = original_step_size[1]
        self.dense = nn.Linear(base_size, original_step_size[0] * original_step_size[1])
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
        self.norm = nn.LayerNorm(base_size)
        self.softmax = nn.Softmax()
        self.for_k = nn.Linear(dense_size, base_size)
        self.num_k_sparse = 1
        self.device = device
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
                    indices = torch.topk(raw, k)[1].to(self.device)
                    mask = torch.ones(raw.shape, dtype=bool).to(self.device)
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

        return out, k_out


class Joint(nn.Module):
    def __init__(self,
                 encoder,
                 decoder,
                 device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                 ):
        """

        :param encoder: encoder model which applies affine transformation
        :type encoder: torch model
        :param decoder: decoder model which automatically generate the base
        :type decoder: torch model
        :param device: set the device where the model generated
        :type device: string ('cuda' or 'cpu)

        """
        super(Joint, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        self.transform_type = encoder.check_type()

    def forward(self, x):

        identity = torch.tensor([0, 0, 1], dtype=torch.float).reshape(1, 1, 3).repeat(x.shape[0], 1, 1).to(self.device)

        if self.transform_type=='combine':
            predicted, kout, theta = self.encoder(x)
            predicted_base, k_out = self.decoder(kout)

            new_theta  = torch.cat((theta, identity), axis=1).to(self.device)
            inver_theta = torch.linalg.inv(new_theta)[:, 0:2].to(self.device)
            grid = F.affine_grid(inver_theta.to(self.device), x.size()).to(self.device)
            predicted_input = F.grid_sample(predicted_base, grid)

            return predicted, predicted_base, predicted_input, k_out, theta

        elif self.transform_type=='rs':
            predicted, kout, rotation, scaler = self.encoder(x)
            predicted_base, k_out = self.decoder(kout)

            new_theta_1 = torch.cat((rotation, identity), axis=1).to(self.device)
            new_theta_2 = torch.cat((scaler, identity), axis=1).to(self.device)

            inver_theta_1 = torch.linalg.inv(new_theta_1)[:, 0:2].to(self.device)
            inver_theta_2 = torch.linalg.inv(new_theta_2)[:, 0:2].to(self.device)

            grid_1 = F.affine_grid(inver_theta_1.to(self.device), x.size()).to(self.device)
            grid_2 = F.affine_grid(inver_theta_2.to(self.device), x.size()).to(self.device)

            predicted_s = F.grid_sample(predicted_base, grid_2)
            predicted_input = F.grid_sample(predicted_s, grid_1)

            return predicted, predicted_base, predicted_input, k_out, rotation, scaler

        elif self.transform_type == 'rt':
            predicted, kout, rotation, translation = self.encoder(x)
            predicted_base, k_out = self.decoder(kout)

            new_theta_1 = torch.cat((rotation, identity), axis=1).to(self.device)
            new_theta_2 = torch.cat((translation, identity), axis=1).to(self.device)

            inver_theta_1 = torch.linalg.inv(new_theta_1)[:, 0:2].to(self.device)
            inver_theta_2 = torch.linalg.inv(new_theta_2)[:, 0:2].to(self.device)

            grid_1 = F.affine_grid(inver_theta_1.to(self.device), x.size()).to(self.device)
            grid_2 = F.affine_grid(inver_theta_2.to(self.device), x.size()).to(self.device)

            predicted_t = F.grid_sample(predicted_base, grid_2)
            predicted_input = F.grid_sample(predicted_t, grid_1)

            return predicted, predicted_base, predicted_input, k_out, rotation, translation

        elif self.transform_type == 'rst':
            predicted, kout, rotation, scaler, translation = self.encoder(x)
            predicted_base, k_out = self.decoder(kout)

            new_theta_1 = torch.cat((rotation, identity), axis=1).to(self.device)
            new_theta_2 = torch.cat((scaler, identity), axis=1).to(self.device)
            new_theta_3 = torch.cat((translation,identity),axis=1).to(self.device)

            inver_theta_1 = torch.linalg.inv(new_theta_1)[:, 0:2].to(self.device)
            inver_theta_2 = torch.linalg.inv(new_theta_2)[:, 0:2].to(self.device)
            inver_theta_3 = torch.linalg.inv(new_theta_3)[:,0:2].to(self.device)

            grid_1 = F.affine_grid(inver_theta_1.to(self.device), x.size()).to(self.device)
            grid_2 = F.affine_grid(inver_theta_2.to(self.device), x.size()).to(self.device)
            grid_3 = F.affine_grid(inver_theta_3.to(self.device), x.size()).to(self.device)

            predicted_t = F.grid_sample(predicted_base, grid_3)
            predicted_s = F.grid_sample(predicted_t, grid_2)
            predicted_input = F.grid_sample(predicted_s, grid_1)

            return predicted, predicted_base, predicted_input, k_out, rotation, scaler, translation

        else:
            raise Exception(
                'the type of affine transformation is invalid, the valid type are: "combine","rst","rs","rt".')


class Regularization(nn.Module):
    def __init__(self, model, weight_decay, p=2):
        '''
        :param model: the model used for calculating the regularization loss
        :type model: pytorch model structure
        :param weight_decay: coeifficient of l1 (l2) regulrization.
        :type weight_decay: float, (>=0)
        :param p: p=1 is l1 regularization, p=2 is l2 regularizaiton
        :type p: int (1 or 2)
        '''
        super(Regularization, self).__init__()
        if weight_decay < 0:
            print("param weight_decay can not <0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    def to(self, device):
        '''
        :param device: cude or cpu
        :type device: string
        :return: put the model on particular machine
        :rtype: pytorch model
        '''
        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        """

        :param model: model ectracted the layer weights
        :type model: pytorch model
        :return: regularization loss
        :rtype: tensor float
        """
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        '''
        :param model: model
        :type model: pytorch model
        :return: list of layers needs to be regularized
        :rtype: list of layers
        '''
        weight_list = []
        for name, param in model.named_parameters():
            if 'dec' in name and 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p):
        '''
        :param weight_list: list of layers needs to be regularized
        :type weight_list: list of layers
        :param p: p=1 is l1 regularization, p=2 is l2 regularizaiton
        :type p: int, (1 or 2)
        :param weight_decay: coeifficient of the regularization parameters
        :type weight_decay: float, (>=0)
        :return: loss
        :rtype: tensor float
        '''
        # weight_decay=Variable(torch.FloatTensor([weight_decay]).to(self.device),requires_grad=True)
        # reg_loss=Variable(torch.FloatTensor([0.]).to(self.device),requires_grad=True)
        # weight_decay=torch.FloatTensor([weight_decay]).to(self.device)
        # reg_loss=torch.FloatTensor([0.]).to(self.device)
        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        '''
        :param weight_list: list of layers needs to be regularized
        :type weight_list: list of layers
        :return: list of layers' name needs to be regularized
        :rtype: list of string
        '''
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)

class Entropy_Loss(nn.Module):
    def __init__(self, entroy_coe):
        """

        :param entroy_coe: the coefficient of the entropy loss parameter
        :type entroy_coe: float, (>=0)
        """
        super(Entropy_Loss, self).__init__()
        self.coe = entroy_coe

    def to(self, device):
        """

        :param device: the gpu or cpu used to implement the function
        :type device: string
        :return: the function implemented on the device
        :rtype: pytorch function
        """
        self.device = device
        super().to(device)
        return self

    def forward(self, x):
        """

        :param x: the data for calculating the loss
        :type x: tensor array
        :return: entorpy loss
        :rtype: tensor float
        """
        en_loss = self.entropy_loss(x)
        return en_loss

    def entropy_loss(self, embedding):
        """

        :param embedding: the data for calculating the loss
        :type embedding: tensor array
        :return: entropy loss
        :rtype: tensor float
        """
        N = embedding.shape[0]
        N = torch.tensor(N).type(torch.float32)
        mask = embedding != 0
        mask1 = torch.sum(mask, axis=0)
        loss = 0
        for k in range(mask1.shape[0]):
            if mask1[k]>0:
                p=mask1[k]/N
                loss += -p*torch.log(p)

        return self.coe * loss


def loss_function(join,
                  transform_type,
                  train_iterator,
                  optimizer,
                  coef=0,
                  coef1=0,
                  ln_parm=1,
                  mask_=None,
                  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                  ):

    """

    :param join: the model for calculating the loss
    :type join: pytorch model
    :param transform_type: choose the type of affine transformation,
                                'rst': rotation, scale, translation;
                                'rs': rotation, scale
                                'rt': rotation, translation
                                'combine': a compound matrix that includes every possible affine transformation
    :type transform_type: string
    :param train_iterator: training data set
    :type train_iterator: DataLoader format
    :param optimizer: pytorch optimizer
    :type optimizer: pytorch optimizer
    :param coef: the coefficient of the entropy loss
    :type coef: float
    :param coef1: the coeifficient of the regularization loss
    :type coef1: float
    :param ln_parm: the type of regularization
    :type ln_parm: int, (1 or 2)
    :param mask_: the mask array to select the MSE loss area
    :type mask_: tensor array
    :return: training loss, entropy loss
    :rtype: float, float
    :param device: set the device where the model generated
    :type device: string ('cuda' or 'cpu)
    """
    entroy_coe = coef
    weight_decay = coef1

    # set the train mode
    join.train()

    # loss of the epoch
    train_loss = 0
    Entropy_loss = 0

    for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):

        x = x.squeeze(0).to(device, dtype=torch.float)

        # update the gradients to zero
        optimizer.zero_grad()


        if transform_type == 'combine':
            predicted_x, predicted_base, predicted_input, kout, theta = join(x)

        elif transform_type == 'rt' or transform_type == 'rs':
            predicted_x, predicted_base, predicted_input, kout, theta_1, theta_2 = join(x)

        elif transform_type == 'rst':
            predicted_x, predicted_base, predicted_input, kout, theta_1, theta_2, theta_3 = join(x)

        else:
            raise Exception(
                'the type of affine transformation is invalid, the valid type are: "combine","rst","rs","rt".')

        entropy_loss = Entropy_Loss(entroy_coe).to(device)
        E_loss = entropy_loss(kout)

        if mask_ != None:

            loss = F.mse_loss(predicted_base.squeeze()[:, mask_], predicted_x.squeeze()[:, mask_], reduction='mean') \
                   + F.mse_loss(predicted_input.squeeze()[:, mask_], x.squeeze()[:, mask_], reduction='mean') \
                   + E_loss

            if loss > 1.5:
                loss = F.l1_loss(predicted_base.squeeze()[:, mask_], predicted_x.squeeze()[:, mask_], reduction='mean') \
                       + F.l1_loss(predicted_input.squeeze()[:, mask_], x.squeeze()[:, mask_], reduction='mean') - 1
            if loss > 2:
                loss = 2
        else:
            loss = F.mse_loss(predicted_base.squeeze(), predicted_x.squeeze(), reduction='mean') \
                   + F.mse_loss(predicted_input.squeeze(), x.squeeze(), reduction='mean') \
                   + E_loss

            if loss > 1.5:
                loss = F.l1_loss(predicted_base.squeeze(), predicted_x.squeeze(), reduction='mean') \
                       + F.l1_loss(predicted_input.squeeze(), x.squeeze(), reduction='mean') - 1
            if loss > 2:
                loss = 2

        # backward pass
        train_loss += loss.item()
        Entropy_loss += E_loss

        loss.backward()
        # update the weights
        optimizer.step()

    return train_loss, Entropy_loss


def Train(join,
          encoder,
          decoder,
          train_iterator,
          optimizer,
          epochs,
          file_dir,
          file_name,
          coef=0,
          coef_1=0,
          ln_parm=1,
          mask_=None,
          epoch_=None,
          initial_point = 10,
          device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
          ):
    """

    :param join: the entire autoencoder model
    :type join: pytorch model
    :param encoder: the encoder of the model
    :type encoder: pytorch model
    :param decoder: the decoder of the model
    :type decoder: pytorch model
    :param train_iterator: the input dataset
    :type train_iterator: pytorch DataLoader format
    :param optimizer: the optimizer of the model layers
    :type optimizer: pytorch optimizer
    :param epochs: the total epochs to train
    :type epochs: int
    :param file_dir: the folder to save the weights
    :type file_dir: string
    :param file_name: the name for the saved weights
    :type file_name: string
    :param coef: the coeifficient of the entropy loss
    :type coef: float
    :param coef1: the coeifficient of the regularization loss
    :type coef1: float
    :param ln_parm: the type of regularization
    :type ln_parm: int, (1 or 2)
    :param mask_: the mask array to select the MSE loss area
    :type mask_: tensor array
    :param epoch_: the epoch index to continue training the model
    :type epoch_: int
    :param initial_point: the epoch index to start saving the weights
    :type initial_point: int
    :param device: set the device where the model generated
    :type device: string ('cuda' or 'cpu)

    """

    N_EPOCHS = epochs
    best_train_loss = float('inf')
    transform_type = encoder.check_type()

    if epoch_ == None:
        start_epoch = 0
    else:
        start_epoch = epoch_ + 1

    for epoch in range(start_epoch, N_EPOCHS):
        # This loss function include the entropy loss with increasing coefficient value
        #         if epoch%20 ==0:
        #             coef += 5e-4
        #             best_train_loss = float('inf')

        train = loss_function(join, transform_type, train_iterator,
                              optimizer, coef, coef_1, ln_parm, mask_,device)
        train_loss, Entropy_loss = train
        train_loss /= len(train_iterator)
        Entropy_loss /= len(train_iterator)

        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')

        print('.............................')
        #  schedular.step()
        if best_train_loss > train_loss:
            best_train_loss = train_loss
            patience_counter = 1
            checkpoint = {
                "net": join.state_dict(),
                "encoder": encoder.state_dict(),
                "decoder": decoder.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                'coef_entropy': coef,
            }
            if epoch >= initial_point:
                torch.save(checkpoint,
                           file_dir+'/'+file_name+f'_entropy:{coef:.4f}_entropy_loss:{Entropy_loss:.4f}_epoch:{epoch}_trainloss:{train_loss:.4f}.pkl')
