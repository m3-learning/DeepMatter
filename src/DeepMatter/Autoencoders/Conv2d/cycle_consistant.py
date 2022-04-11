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
