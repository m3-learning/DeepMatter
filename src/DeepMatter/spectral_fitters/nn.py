import torch.nn as nn
import torch

class DensePhysEnc9185(nn.Module):

    def __init__(self,
                 x_vector,
                 model,
                 dense_params=3,
                 verbose=False,
                 device = 'cuda',
                 num_channels=1,
                 **kwargs):

        """

        Args:
            x_vector: The vector of values for x
            model: the empirical function to fit
            dense_params: number of output parameters to the model
            verbose: sets if the model is verbose
            device: device where the model will run
            num_channels: number of channels in the input
        """

        super().__init__()
        self.dense_params = dense_params
        self.x_vector = x_vector
        self.verbose = verbose
        self.num_channels = num_channels
        self.device = device
        self.model_params = kwargs.get('model_params')
        self.model = model(self.x_vector, size=(num_channels, dense_params // self.model_params))


        if torch.cuda.is_available():
            self.cuda()

        # Input block of 1d convolution
        self.hidden_x1 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_channels, out_channels=8, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=8, out_channels=6, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=6, out_channels=4, kernel_size=5),
            nn.SELU(),
        )

        self.hidden_x1_shape = self.hidden_x1(
            torch.zeros(1, self.num_channels,
                        self.x_vector.shape[0])).shape

        # fully connected block
        self.hidden_xfc = nn.Sequential(
            nn.Linear(self.hidden_x1_shape[1] * self.hidden_x1_shape[2], 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU(),
        )  # out of size 20

        self.hidden_xfc_shape = self.hidden_xfc(torch.zeros(1,
                                                            self.hidden_x1_shape[1] * self.hidden_x1_shape[2])).shape

        # 2nd block of 1d-conv layers
        self.hidden_x2 = nn.Sequential(
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(in_channels=1, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4, out_channels=4, kernel_size=5),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=4, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=2, out_channels=2, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
        )

        self.hidden_x2_shape = self.hidden_x2(torch.zeros((self.hidden_xfc_shape[0],
                                                           1,
                                                           self.hidden_x1_shape[1] * self.hidden_x1_shape[2]))).shape

        # Flatten layer
        self.flatten_layer = nn.Flatten()

        # Final embedding block - Output 4 values - linear
        self.hidden_embedding = nn.Sequential(
            nn.Linear(self.hidden_x2_shape[1] * self.hidden_x2_shape[2] + self.hidden_xfc_shape[1], 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, self.dense_params),
        )

    def forward(self, x, n=-1):

        if self.verbose:
            print(f'input shape = {x.shape}')
        x = self.hidden_x1(x)
        if self.verbose:
            print(f'hidden_x1 shape = {x.shape}')
        xfc = torch.reshape(x, (x.shape[0], -1))  # batch size, features
        if self.verbose:
            print(f'reshape shape = {xfc.shape}')
        xfc = self.hidden_xfc(xfc)
        if self.verbose:
            print(f'xfc shape = {xfc.shape}')
        x = torch.reshape(x, (x.shape[0], 1,
                              self.hidden_x1_shape[1] * self.hidden_x1_shape[2]))
        # batch size, (real, imag), timesteps
        if self.verbose:
            print(f'reshape 2 shape = {x.shape}')
        x = self.hidden_x2(x)
        if self.verbose:
            print(f'hidden-x2 shape = {x.shape}')
        cnn_flat = self.flatten_layer(x)
        if self.verbose:
            print(f'cnn_flatten shape = {cnn_flat.shape}')
        encoded = torch.cat((cnn_flat, xfc), 1)  # merge dense and 1d conv.
        if self.verbose:
            print(f'encoded shape = {encoded.shape}')
        embedding = self.hidden_embedding(encoded)  # output is 4 parameters
        if self.verbose:
            print(f'embedding shape = {embedding.shape}')

        out = self.model.compute(embedding,
                                 device=self.device)
        if self.verbose:
            print(f'Computation shape = {out.shape}')

        out = torch.transpose(out, 1, 2)
        out = torch.atleast_3d(out)
        if self.verbose:
            print(f'Ouput shape = {out.shape}')

        return out.to(self.device)
