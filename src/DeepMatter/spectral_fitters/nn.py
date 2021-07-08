import torch.nn as nn
import torch

class DensePhysLarger(nn.Module):
    
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
        n = 4


        if torch.cuda.is_available():
            self.cuda()

        # Input block of 1d convolution
        self.hidden_x1 = nn.Sequential(
            nn.Conv1d(in_channels=self.num_channels, out_channels=8*n, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=8*n, out_channels=6*n, kernel_size=7),
            nn.SELU(),
            nn.Conv1d(in_channels=6*n, out_channels=4, kernel_size=5),
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
            nn.Conv1d(in_channels=1, out_channels=4*n, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4*n, out_channels=4*n, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4*n, out_channels=4*n, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4*n, out_channels=4*n, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4*n, out_channels=4*n, kernel_size=5),
            nn.SELU(),
            nn.Conv1d(in_channels=4*n, out_channels=4*n, kernel_size=5),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=4*n, out_channels=2*n, kernel_size=3),
            nn.SELU(),
            nn.AvgPool1d(kernel_size=2),
            nn.Conv1d(in_channels=2*n, out_channels=2, kernel_size=3),
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

        x = self.hidden_x1(x)
        xfc = torch.reshape(x, (x.shape[0], -1))  # batch size, features
        xfc = self.hidden_xfc(xfc)
        x = torch.reshape(x, (x.shape[0], 1,
                              self.hidden_x1_shape[1] * self.hidden_x1_shape[2]))
        # batch size, (real, imag), timesteps
        x = self.hidden_x2(x)
        cnn_flat = self.flatten_layer(x)
        encoded = torch.cat((cnn_flat, xfc), 1)  # merge dense and 1d conv.
            
#         print("encoded", encoded.shape)    
            
        embedding = self.hidden_embedding(encoded)  # output is 4 parameters

#         print("embedding", embedding.shape)    
        
        
        embedding = torch.reshape(embedding, (embedding.shape[0], 4, -1))

        embedding[:,0,:] = embedding[:,0,:] * self.model.mean_sd + self.model.mean_mean
        embedding[:,1,:] = embedding[:,1,:] * self.model.sd_sd + self.model.sd_mean
        embedding[:,2,:] = embedding[:,2,:] * self.model.amp_sd + self.model.amp_mean
        embedding[:,3,:] = embedding[:,3,:] * self.model.fraction_sd + self.model.fraction_mean

        
        embedding = torch.reshape(embedding,(embedding.shape[0],-1))
        
        
        self.embed = embedding
        
        
        out = self.model.compute(embedding,
                                 device=self.device)
        out = torch.transpose(out, 1, 2)
        out = torch.atleast_3d(out)

        return out.to(self.device), embedding.to(self.device)
