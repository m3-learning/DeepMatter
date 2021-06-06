import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm


class conv2D_Res_block(nn.Module):
    """
    Class that builds a two dimensional convolutional layer with a residual block
    """

    def __init__(self, t_size, n_step):
        """

        Args:
            self:
            t_size: size of the layer
            n_step: input shape from an expected input of size

        Returns:

        """
        super(conv2D_Res_block, self).__init__()
        self.conv2d_1 = nn.Conv2d(t_size, t_size,
                                  3, stride=1,
                                  padding=1,
                                  padding_mode='zeros')

        self.conv2d_2 = nn.Conv2d(t_size, t_size, 3,
                                  stride=1,
                                  padding=1,
                                  padding_mode='zeros')
        self.norm_3 = nn.LayerNorm(n_step)
        self.relu_4 = nn.ReLU()

    def forward(self, x):
        """

        Args:
            self:
            x: input data

        Returns:

        """
        x_input = x
        out = self.conv2d_1(x)
        out = self.conv2d_2(out)
        out = self.norm_3(out)
        out = self.relu_4(out)
        out = out.add(x_input)
        return out


class conv2d_block(nn.Module):
    """
    Single conv2d block with layer norm and relu
    """

    def __init__(self, t_size, n_step):
        """

        Args:
            t_size: size of the layer
            n_step: input shape from an expected input of size
        """

        super(conv2d_block, self).__init__()
        self.conv2d_1 = nn.Conv2d(t_size, t_size, 3, stride=1,
                                  padding=1, padding_mode='zeros')
        self.norm_1 = nn.LayerNorm(n_step)
        self.relu = nn.ReLU()

    def forward(self, x):
        """

        Args:
            x: input data

        Returns:

        """
        out = self.conv2d_1(x)
        out = self.norm_1(out)
        out = self.relu(out)
        return out


class Autoencoder_Conv2D(nn.Module):
    """
    Conv 2D autoencoder constructor
    """

    def __init__(self, kernal_size=[15, 15]):
        """

        Args:
            kernal_size: size of the kernal of the input images
        """
        super().__init__()
        self.kernal_size = kernal_size
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.Encoder = self.Encoder(AE=self).to(device)
        self.Decoder = self.Decoder(AE=self).to(device)

    def forward(self, x):
        """
        forward pass on the autoencoder
        Args:
            x: input data

        Returns:

        """
        embedding = self.Encoder(x)
        predicted = self.Decoder(embedding)
        return predicted

    class Encoder(nn.Module):
        """
        class that defines the encoder
        """

        def __init__(self, AE):
            """

            Args:
                AE: call to higher level class for autoencoder
            """
            super(Autoencoder_Conv2D.Encoder, self).__init__()
            self.kernal_size = AE.kernal_size
            self.cov2d = nn.Conv2d(1, 128, 3, stride=1, padding=1, padding_mode='zeros')
            self.cov2d_1 = nn.Conv2d(128, 1, 3, stride=1, padding=1, padding_mode='zeros')
            self.conv2D_Res_block_1 = conv2D_Res_block(t_size=128, n_step=self.kernal_size)
            self.conv2D_Res_block_2 = conv2D_Res_block(t_size=128, n_step=self.kernal_size)
            self.conv2D_Res_block_3 = conv2D_Res_block(t_size=128, n_step=self.kernal_size)
            self.conv2d_block_1 = conv2d_block(t_size=128, n_step=self.kernal_size)
            self.conv2d_block_2 = conv2d_block(t_size=128, n_step=self.kernal_size)
            self.conv2d_block_3 = conv2d_block(t_size=128, n_step=self.kernal_size)
            self.relu_1 = nn.ReLU()
            self.dense = nn.Linear(225, 64)

        def forward(self, x):
            """

            Args:
                x: input data

            Returns:

            """
            out = x.view(-1, 1, self.kernal_size[0], self.kernal_size[1])
            out = self.cov2d(out)
            out = self.conv2d_block_1(out)
            out = self.conv2D_Res_block_1(out)
            out = self.conv2d_block_2(out)
            out = self.conv2D_Res_block_2(out)
            out = self.conv2d_block_3(out)
            out = self.conv2D_Res_block_3(out)
            out = self.cov2d_1(out)
            out = torch.flatten(out, start_dim=1)
            selection = self.dense(out)
            selection = self.relu_1(selection)
            return selection

    class Decoder(nn.Module):
        def __init__(self, AE):
            """

            Args:
                AE: call to higher level class for autoencoder
            """
            super(Autoencoder_Conv2D.Decoder, self).__init__()
            self.kernal_size = AE.kernal_size
            self.dense = nn.Linear(64, 225)
            self.cov2d = nn.Conv2d(1, 128, 3, stride=1, padding=1, padding_mode='zeros')
            self.cov2d_1 = nn.Conv2d(128, 1, 3, stride=1, padding=1, padding_mode='zeros')
            self.conv2D_Res_block_1 = conv2D_Res_block(t_size=128, n_step=self.kernal_size)
            self.conv2D_Res_block_2 = conv2D_Res_block(t_size=128, n_step=self.kernal_size)
            self.conv2D_Res_block_3 = conv2D_Res_block(t_size=128, n_step=self.kernal_size)
            self.conv2d_block_1 = conv2d_block(t_size=128, n_step=self.kernal_size)
            self.conv2d_block_2 = conv2d_block(t_size=128, n_step=self.kernal_size)
            self.conv2d_block_3 = conv2d_block(t_size=128, n_step=self.kernal_size)

        def forward(self, x):
            """

            Args:
                x: input data

            Returns:

            """
            out = self.dense(x)
            out = out.view(-1, 1, self.kernal_size[0], self.kernal_size[1])
            out = self.cov2d(out)
            out = self.conv2d_block_1(out)
            out = self.conv2D_Res_block_1(out)
            out = self.conv2d_block_2(out)
            out = self.conv2D_Res_block_2(out)
            out = self.conv2d_block_3(out)
            out = self.conv2D_Res_block_3(out)
            out = self.cov2d_1(out)
            out = out.view(-1, self.kernal_size[0], self.kernal_size[1])

            return out


class Regularization(torch.nn.Module):
    """
    module that builds the regularizer for the loss function
    """

    def __init__(self, model, weight_decay, p=1):
        """

        Args:
            model: model that you are training
            weight_decay: sets the rate of weight decay. default is zero
            p: normalization factor of the regularization term.
        """
        super(Regularization, self).__init__()
        if weight_decay < 0:
            print("param weight_decay can not <0")
            exit(0)
        self.model = model
        self.weight_decay = weight_decay
        self.p = p
        self.weight_list = self.get_weight(model)

    def to(self, device):
        """

        Args:
            device: device where the computation will happen

        Returns:

        """

        self.device = device
        super().to(device)
        return self

    def forward(self, model):
        self.weight_list = self.get_weight(model)
        reg_loss = self.regularization_loss(self.weight_list, self.weight_decay, p=self.p)
        return reg_loss

    def get_weight(self, model):
        """

        Args:
            model: model to get weights from

        Returns: list of weights

        """
        weight_list = []
        for name, param in model.named_parameters():
            if 'dec' in name and 'weight' in name:
                weight = (name, param)
                weight_list.append(weight)
        return weight_list

    def regularization_loss(self, weight_list, weight_decay, p):
        """

        Args:
            weight_list: list of the weights
            weight_decay: parameters for the weight decay
            p: exponet of the regularization

        Returns:

        """

        reg_loss = 0
        for name, w in weight_list:
            l2_reg = torch.norm(w, p=p)
            reg_loss = reg_loss + l2_reg

        reg_loss = weight_decay * reg_loss
        return reg_loss

    def weight_info(self, weight_list):
        """
        Function used to print the information about the weights
        Args:
            weight_list: list of weights

        Returns:

        """
        print("---------------regularization weight---------------")
        for name, w in weight_list:
            print(name)


def AE_loss(model,
            train_iterator,
            optimizer,
            coef=0,
            coef1=0,
            ln_parm=1,
            beta=None):
    """

    Function that calculates the loss of the autoencoder

    Args:
        model: model for the autoencoder
        train_iterator: pytorch data generator for autoencoder
        optimizer: optimizer that is used.
        coef: initial regularization beta parameter
        coef1: sets the rate of decay of the regularization parameter
        ln_parm: sets the learning rate
        beta: Beta value for VAE

    Returns:

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    weight_decay = coef
    weight_decay_1 = coef1

    # set the train mode
    model.train()

    # loss of the epoch
    train_loss = 0

    for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):

        reg_loss_2 = Regularization(model, weight_decay_1, p=2).to(device)

        x = x.to(device, dtype=torch.float)

        # update the gradients to zero
        optimizer.zero_grad()

        if beta is None:
            embedding = model.Encoder(x)

        else:
            embedding, sd, mn = model.Encoder(x)

        if weight_decay > 0:
            reg_loss_1 = weight_decay * torch.norm(embedding, ln_parm).to(device)
        else:
            reg_loss_1 = 0.0

        predicted_x = model.decoder(embedding)

        # reconstruction loss
        loss = F.mse_loss(x, predicted_x, reduction='mean')

        loss = loss + reg_loss_1 + reg_loss_2(model)

        if beta is not None:
            vae_loss = beta * 0.5 * torch.sum(torch.exp(sd) + (mn) ** 2 - 1.0 - sd).to(device)
            vae_loss /= (sd.shape[0] * sd.shape[1])
        else:
            vae_loss = 0

        loss = loss + vae_loss

        # backward pass
        train_loss += loss.item()

        loss.backward()

        # update the weights
        optimizer.step()

    return train_loss


def AE_Train(model,
             train_iterator,
             optimizer,
             epochs,
             coef=0,
             coef_1=0,
             ln_parm=1,
             beta=None,
             path='./',
             filename='model'):
    """

    Args:
        model: neural network to train
        train_iterator: data loader
        optimizer: optimizers
        epochs: number of epochs
        coef: initial regularization beta parameter
        coef_1: sets the rate of decay of the regularization parameter
        ln_parm: Set the order of the normalization
        beta: Set the beta value for B VAE
        path: path where data is saved
        filename: filename for file

    Returns:

    """
    N_EPOCHS = epochs
    best_train_loss = float('inf')

    for epoch in range(N_EPOCHS):

        train = AE_loss(model, train_iterator,
                        optimizer, coef, coef_1, ln_parm, beta)
        train_loss = train
        train_loss /= len(train_iterator)
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')
        print('.............................')

        if best_train_loss > train_loss:
            best_train_loss = train_loss
            patience_counter = 1
            checkpoint = {
                "net": model.state_dict(),
                'optimizer': optimizer.state_dict(),
                "epoch": epoch,
                "encoder": model.Encoder.state_dict(),
                'decoder': model.Decoder.state_dict()
            }

            torch.save(checkpoint, path + filename + '.pkl')


def transfer_layer_names(original, updated):

    """
    function that assists in changing the name of a model.

    Args:
        original: Original model with model names
        updated: Updated model where the new names for loading

    Returns:
        original: updated model with new naming convention

    """
    # extracts the updated keys
    new_names = list(updated.keys())

    # copies the original dictionary
    original_ = original.copy()

    for i, name in enumerate(original_):
        original[new_names[i]] = original.pop(name)

    return original