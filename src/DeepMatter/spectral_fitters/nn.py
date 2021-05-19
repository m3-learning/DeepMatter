import torch.nn as nn
import torch

class DensePhysEnc9185(nn.Module):

    def __init__(self,
                 x_vector,
                 model,
                 num_params=3,
                 verbose=False,
                 device = 'cuda',
                 num_channels=1):

        """

        Args:
            x_vector: The vector of values for x
            model: the empirical function to fit
            num_params: number of output parameters to the model
            verbose: sets if the model is verbose
            device: device where the model will run
            num_channels: number of channels in the input
        """

        super().__init__()
        self.num_params = num_params
        self.x_vector = x_vector
        self.verbose = verbose
        self.num_channels = num_channels
        self.device = device
        self.model = model(self.x_vector)

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

        # fully connected block
        self.hidden_xfc = nn.Sequential(
            nn.Linear(336, 20),
            nn.SELU(),
            nn.Linear(20, 20),
            nn.SELU(),
        )

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

        # Flatten layer
        self.flatten_layer = nn.Flatten()

        # Final embedding block - Output 4 values - linear
        self.hidden_embedding = nn.Sequential(
            nn.Linear(52, 16),
            nn.SELU(),
            nn.Linear(16, 8),
            nn.SELU(),
            nn.Linear(8, num_params),
        )

    def forward(self, x, n=-1):
        if self.verbose:
            print(f'input {x.shape}')
        x = torch.transpose(x, 1, 2)  # output shape - samples, (real, imag), frequency
        if self.verbose:
            print(f'input_swap {x.shape}')
        x = self.hidden_x1(x)
        if self.verbose:
            print(f'hidden_x1 {x.shape}')
        xfc = torch.reshape(x, (x.shape[0], -1))  # batch size, features
        if self.verbose:
            print(f'pre-xfc {xfc.shape}')
        xfc = self.hidden_xfc(xfc)
        if self.verbose:
            print(f'post-xfc {xfc.shape}')
        x = torch.reshape(x, (x.shape[0], 1, 336))  # batch size, (real, imag), timesteps
        if self.verbose:
            print(f'pre-hidden-x2 {x.shape}')
        x = self.hidden_x2(x)
        if self.verbose:
            print(f'hidden-x2 {x.shape}')
        cnn_flat = self.flatten_layer(x)
        if self.verbose:
            print(f'pre-embedding {cnn_flat.shape}')
        encoded = torch.cat((cnn_flat, xfc), 1)  # merge dense and 1d conv.
        if self.verbose:
            print(f'post_encoded {encoded.shape}')
        embedding = self.hidden_embedding(encoded)  # output is 4 parameters
        if self.verbose:
            print(f'post_embedding {embedding.shape}')

        out = self.model.compute(embedding,
                                 device=self.device)

        out = torch.transpose(out, 0, 1)
        out = torch.atleast_3d(out)
        if self.verbose:
            print(f'end_dimensions {out.shape}')

        return out
