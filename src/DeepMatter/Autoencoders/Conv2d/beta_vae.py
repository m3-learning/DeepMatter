import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import os
import tensorflow.keras.backend as K
import string
from tqdm import tqdm
from scipy import ndimage
from sklearn.decomposition import DictionaryLearning
from tensorflow.keras.models import Sequential, Model
from mpl_toolkits.axes_grid1 import make_axes_locatable
from tensorflow.keras.layers import (Input,Reshape,Activation,Attention,MaxPool1D,Dense, Conv1D, Convolution2D, GRU, LSTM, Lambda, Bidirectional, TimeDistributed,
                          Dropout, Flatten, LayerNormalization,RepeatVector, Reshape, MaxPooling1D, UpSampling1D, BatchNormalization)
import tensorflow.keras.layers as layers
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.optimizers import Adam
from .Auto_format import *

class generator:
    def __init__(self,
                 model,
                 scaled_data,
                 image,
                 channels=None,
                 color_map='viridis'):
        """

        :param model: model put in generator
        :type model: the dictionary learning or neural network model
        :param scaled_data: the data put in the model
        :type scaled_data: numpy
        :param image: image that has the same size of the output embedding
        :type image: numpy
        :param channels: channels index that used for generating generator movie
        :type channels: list of int
        :param color_map: the type of the color map
        :type color_map: string
        """
        self.model = model
        self.image = image
        # defines the colorlist
        self.cmap = plt.get_cmap(color_map)
        self.modified_model = None

        if isinstance(model, type(DictionaryLearning())):
            def predictor(values):
                return np.dot(values, model.components_)

            self.predict = predictor
            self.vector_length = scaled_data.shape[1]
            self.embeddings = model.transform(scaled_data)
        elif np.atleast_3d(scaled_data).shape[2] == 1:

            def predictor(values):
                return model.decoder_model.predict(np.atleast_2d(values))

            self.embeddings = model.encoder_model.predict(np.atleast_3d(scaled_data))
            self.predict = predictor
            self.vector_length = scaled_data.shape[1]

        elif np.atleast_3d(scaled_data).shape[2] == 2:
            self.modified_model = 1

            def predictor(means, stds):

                return model.decoder_model.predict([np.atleast_2d(means), np.atleast_2d(stds)])

            self.emb_, self.mean, self.std = model.encoder_model.predict(np.atleast_3d(scaled_data))
            self.embeddings_tf = Sampling()([self.mean, self.std])
           # self.embeddings = self.embeddings_tf.numpy()
            self.embeddings = self.embeddings_tf.eval(session=tf.compat.v1.Session())
            self.predict = predictor
            self.vector_length = scaled_data.shape[1]
        else:
            raise Exception('The model is not an included model type ')

        if channels is None:
            self.channels = range(self.embeddings.shape[1])
        else:
            self.channels = channels

    def generator_images(self,
                         folder,
                         ranges=None,
                         number_of_loops=200,
                         averaging_number=100,
                         graph_layout=[3, 3],
                         model_type='dog',
                         y_lim=[-2, 2],
                         y_lim_1=[-2, 2],
                         xlabel='Voltage (V)',
                         ylabel='',
                         xvalues=None
                         ):
        """

        :param folder: folder where to save
        :type folder: string
        :param ranges: range of each embedding value
        :type ranges: list
        :param number_of_loops: embedding range divided by step size of it
        :type number_of_loops: int
        :param averaging_number: number of index which is nearest to the current value
        :type averaging_number: int
        :param graph_layout: format of output graph
        :type graph_layout: list
        :param model_type: the type of the model, 'dog' or 'nn'
        :type model_type: string
        :param y_lim: set the y scale
        :type y_lim: list
        :param xlabel: set the label of x axis
        :type xlabel; string
        :param ylabel: set the label of y axis
        :type ylabel: string
        :param xvalues: set the x axis
        :type xvalues: array

        """
        folder = make_folder(folder)
        for i in tqdm(range(number_of_loops)):
            if model_type == 'dog':
                fig, ax = layout_fig(graph_layout[0] * 3, mod=graph_layout[1])
            else:
                fig, ax = layout_fig(graph_layout[0] * 4, mod=graph_layout[1])
            ax = ax.reshape(-1)

            # loops around all of the embeddings
            for j, channel in enumerate(self.channels):

                # checks if the value is None and if so skips tp next iteration
                if i is None:
                    continue

                if xvalues is None:
                    xvalues = range(self.vector_length)

                if ranges is None:
                    ranges = np.stack((np.min(self.embeddings, axis=0),
                                       np.max(self.embeddings, axis=0)), axis=1)

                # linear space values for the embeddings
                value = np.linspace(ranges[channel][0], ranges[channel][1],
                                    number_of_loops)
                # finds the nearest point to the value and then takes the average
                # average number of points based on the averaging number
                idx = find_nearest(
                    self.embeddings[:, channel],
                    value[i],
                    averaging_number)
                # computes the mean of the selected index

                if self.modified_model is not None:
                    gen_mean = np.mean(self.mean[idx], axis=0)
                    gen_std = np.mean(self.std[idx], axis=0)

                    mn_ranges = np.stack((np.min(self.mean, axis=0),
                                          np.max(self.mean, axis=0)), axis=1)
                    sd_ranges = np.stack((np.min(self.std, axis=0),
                                          np.max(self.std, axis=0)), axis=1)

                    mn_value = np.linspace(mn_ranges[channel][0], mn_ranges[channel][1],
                                           number_of_loops)

                    sd_value = np.linspace(sd_ranges[channel][0], sd_ranges[channel][1],
                                           number_of_loops)

                    gen_mean[channel] = mn_value[i]

                    gen_std[channel] = sd_value[i]
                    generated = self.predict(gen_mean, gen_std).squeeze()

                if self.modified_model is None:
                    gen_value = np.mean(self.embeddings[idx], axis=0)

                    # specifically updates the value of the embedding to visualize based on the
                    # linear spaced vector
                    gen_value[channel] = value[i]

                    # generates the loop based on the model
                    generated = self.predict(gen_value).squeeze()

                # plots the graph
                ax[j].imshow(self.embeddings[:, channel].reshape(self.image.shape[0:2]), clim=ranges[channel])
                ax[j].set_yticklabels('')
                ax[j].set_xticklabels('')
                y_axis, x_axis = np.histogram(self.embeddings[:, channel], number_of_loops)
                if model_type == 'dog':
                    ax[j + len(self.channels)].plot(xvalues, generated,
                                                    color=self.cmap((i + 1) / number_of_loops))
                    # formats the graph
                    ax[j + len(self.channels)].set_ylim(y_lim[0], y_lim[1])
                    #   ax[j+len(self.channels)].set_yticklabels('Piezoresponse (Arb. U.)')
                    ax[j + len(self.channels)].set_ylabel('Amplitude')
                    ax[j + len(self.channels)].set_xlabel(xlabel)
                    ax[j + len(self.channels) * 2].hist(self.embeddings[:, channel], number_of_loops)
                    ax[j + len(self.channels) * 2].plot(x_axis[i], y_axis[i], 'ro')
                    ax[j + len(self.channels) * 2].set_ylabel('Counts')
                    ax[j + len(self.channels) * 2].set_xlabel('Embedding Intensity')
                else:
                    if len(generated.shape) == 1:
                        new_range = int(len(generated) / 2)
                        generated_1 = generated[:new_range].reshape(new_range, 1)
                        generated_2 = generated[new_range:].reshape(new_range, 1)
                        generated = np.concatenate((generated_1, generated_2), axis=1)
                        if len(xvalues) != generated.shape[0]:
                            xvalues = range(int(self.vector_length / 2))

                    ax[j + len(self.channels)].plot(xvalues,
                                                    generated[:, 0] * 7.859902800847493e-05 - 1.0487273116670697e-05
                                                    , color=self.cmap((i + 1) / number_of_loops))
                    # formats the graph
                    ax[j + len(self.channels)].set_ylim(y_lim[0], y_lim[1])
                    ax[j + len(self.channels)].set_ylabel('Piezoresponse (Arb. U.)')
                    ax[j + len(self.channels)].set_xlabel(xlabel)

                    ax[j + len(self.channels) * 2].plot(xvalues,
                                                        generated[:, 1] * 3.1454182388943095 + 1324.800141637855,
                                                        color=self.cmap((i + 1) / number_of_loops))
                    # formats the graph
                    ax[j + len(self.channels) * 2].set_ylim(y_lim_1[0], y_lim_1[1])
                    ax[j + len(self.channels) * 2].set_ylabel('Resonance (kHz)')
                    ax[j + len(self.channels) * 2].set_xlabel(xlabel)
                    ax[j + len(self.channels) * 3].hist(self.embeddings[:, channel], number_of_loops)
                    ax[j + len(self.channels) * 3].plot(x_axis[i], y_axis[i], 'ro')
                    ax[j + len(self.channels) * 3].set_ylabel('Counts')
                    ax[j + len(self.channels) * 3].set_xlabel('Embedding Intensity')

            ax[0].set_ylabel(ylabel)
            fig.tight_layout()
            savefig(pjoin(folder, f'{i:04d}_maps'), printing)

            plt.close(fig)


def embedding_maps_movie(data, image, folder, beta, loss,
                         filename='./embedding_maps', c_lim=None, mod=4, colorbar_shown=True):
    """

    :param data: raw data to plot of embeddings
    :type data: numpy
    :param image: the image of same size with the embeddings
    :type image: numpy
    :param folder: the directory to save the result
    :type folder: string
    :param beta: beta value of the weights
    :type beta: float
    :param loss: loss of the weights
    :type loss: float
    :param filename: name of the file
    :type filename: string
    :param c_lim: the color range to show
    :type c_lim: list of float
    :param mod: number of plots for each row
    :type mod: int
    :param colorbar_shown: decide whether to show the colorbar
    :type colorbar_shown: boolean
    :return: the plots
    :rtype: png
    """

    # creates the figures and axes in a pretty way
    fig, ax = layout_fig(data.shape[1], mod)
    title_name = 'beta=' + beta + '_loss=' + loss
    fig.suptitle(title_name, fontsize=12, y=1)
    for i, ax in enumerate(ax):
        if i < data.shape[1]:
            im = ax.imshow(data[:, i].reshape(image.shape[0], image.shape[1]))
            ax.set_xticklabels('')
            ax.set_yticklabels('')

            # adds the colorbar
        if colorbar_shown == True:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='10%', pad=0.05)
            cbar = plt.colorbar(im, cax=cax, format='%.1e')

            # Sets the scales
            if c_lim is not None:
                im.set_clim(c_lim)

    # plots all of the images

    plt.tight_layout(pad=1)
    fig.set_size_inches(12, 12)

    # saves the figure
    fig.savefig(folder + '/' + filename + '.png', dpi=300)

    return (fig)


def training_images(model,
                    data,
                    image,
                    number_layers,
                    model_folder,
                    beta,
                    printing,
                    folder,
                    file_name):
    """ plots the training images

    :param model: tensorflow object neural network model
    :type model: tensorflow version model
    :param data: data trained by the model
    :type data: numpy array
    :param image: the image with the size of the plots
    :type image: numpy array
    :param number_layers: the index of the embedding layer
    :type number_layers: int
    :param model_folder: the directory of model weights
    :type model_folder: string
    :param beta: beta value of the weights
    :type beta: string
    :param printing: printing format
    :type printing: dictionary
    :param folder: directory to load the plots
    :type folder: string
    :param file_name: the name of the plots
    :type file_name: string
    """

    # makes a copy of the format information to modify
    printing_ = printing.copy()

    # sets to remove the color bars and not to print EPS

    printing_['EPS'] = False

    # simple function to help extract the filename
    def name_extraction(filename):
        filename = file_list[0].split('/')[-1][:-5]
        return filename

    embedding_exported = {}

    # searches the folder and finds the files
    file_list = glob.glob(model_folder + '/phase_shift_only*')
    file_list = natsorted(file_list, key=lambda y: y.lower())

    for i, file_list in enumerate(file_list):
        # load beta and loss value
        loss_ = file_list[-12:-5]

        # loads the weights into the model
        model.load_weights(file_list)

        # Computes the low dimensional layer
        embedding_exported[name_extraction(file_list)] = get_activations(model, data, number_layers)

        # plots the embedding maps
        _ = embedding_maps_movie(embedding_exported[name_extraction(file_list)], image,
                                 folder, beta, loss_, filename='./' + file_name + '_epoch_{0:04}'.format(i))

        # Closes the figure
        plt.close(_)


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class model_builder:

    def __init__(self,
                 input_data,
                 drop_frac=0.2,
                 layer_size=128,
                 num_ident_blocks=3,
                 l1_norm=0,
                 l1_norm_embedding=1e-3,
                 layer_steps=2,
                 embedding=16,
                 VAE=True,
                 coef=1):
        """ developed for dog data and scaled data

        :param input_data: the training dataset
        :type input_data: numpy array
        :param drop_frac: the dropout fraction
        :type drop_frac: float < 1
        :param layer_size: the depth of each LSTM layer
        :type layer_size: int
        :param num_ident_blocks: the number of LSTM ResNet blocks
        :type num_ident_blocks: int
        :param l1_norm: the parameter of l1 regularization
        :type l1_norm: float
        :param l1_norm_embedding: the parameter of l1 for embedding layer
        :type l1_norm_embedding: float
        :param layer_steps: the number of LSTM layer in each block
        :type layer_steps: int
        :param embedding: the number of channels for embedding channel
        :type embedding: int
        :param VAE: whether add the KL divergence in loss function
        :type VAE: boolean
        :param coef: the power parameters of the l1
        :type coef: float
        """

        # Sets self.mean and self.std to use in the loss function;
        #       self.mean = 0
        #       self.std = 0

        # Sets the L1 norm on the decoder/encoder layers
        self.l1_norm = l1_norm

        # Sets the fraction of dropout
        self.drop_frac = drop_frac

        # saves the shape of the input data
        self.data_shape = input_data.shape

        # Sets the number of neurons in the encoder/decoder layers
        self.layer_size = layer_size

        # Sets the number of neurons in the embedding layer
        self.embedding = embedding

        # Bool to set if the model is a VAE
        self.VAE = VAE

        # Set the magnitude of the l1 regularization on the embedding layer.
        self.l1_norm_embedding = l1_norm_embedding

        # sets the number of layers between the residual layer
        self.layer_steps = layer_steps

        self.coef = coef

        # set the number of identity block
        self.num_ident_blocks = num_ident_blocks

        self.model_constructor(input_data)

    def identity_block(self, X, name,
                       block):

        # sets the name of the conv layers
        LSTM_name_base = name + '_LSTM_Res_' + block
        bn_name_base = name + '_layer_norm_' + block

        # output for the residual layer
        X_shortcut = X

        for i in range(self.layer_steps):
            # bidirectional LSTM
            X = layers.Bidirectional(LSTM(self.layer_size,
                                          return_sequences=True,
                                          dropout=0.0,
                                          activity_regularizer=l1(self.l1_norm)),
                                     input_shape=(self.data_shape[1], self.data_shape[2]))(X)


            X = layers.Activation('relu')(X)

        X = layers.add([X, X_shortcut])
        X = layers.LayerNormalization(axis=1, name=bn_name_base + '_res_end')(X)
        X = layers.Activation('relu')(X)

        return X

    def model_constructor(self, input_data):
        # defines the input
        encoder_input = layers.Input(shape=(self.data_shape[1:]))

        X = encoder_input

        for i in range(self.num_ident_blocks):
            X = self.identity_block(X, 'encoder', string.ascii_uppercase[i + 1])

        # This is in preparation for the embedding layer
        X = layers.Bidirectional(LSTM(self.layer_size,
                                      return_sequences=False,
                                      dropout=0.0,
                                      activity_regularizer=l1(self.l1_norm)),
                                 input_shape=(self.data_shape[1],
                                              self.data_shape[2]))(X)

        X = layers.BatchNormalization(axis=1, name='last_encode')(X)
        X = layers.Activation('relu')(X)

        if self.VAE:
            X = layers.Dense(self.embedding, name="embedding_pre")(X)
            X = layers.Activation('relu')(X)
            X = layers.ActivityRegularization(l1=self.l1_norm_embedding * 10 ** (self.coef))(X)
            z_mean = layers.Dense(self.embedding, name="z_mean")(X)
            z_log_var = layers.Dense(self.embedding, name="z_log_var")(X)
            sampling = Sampling()((z_mean, z_log_var))
            # update the self.mean and self.std:
        #            self.mean = z_mean
        #            self.std = z_log_var

        self.encoder_model = Model(inputs=encoder_input, outputs=sampling, name='LSTM_encoder')

        decoder_input = layers.Input(shape=(self.embedding,), name="z_sampling")

        z = layers.Dense(self.embedding, name="embedding")(decoder_input)
        z = layers.Activation('relu')(z)
        z = layers.ActivityRegularization(l1=self.l1_norm_embedding * 10 ** (self.coef))(z)

        X = layers.RepeatVector(self.data_shape[1])(z)

        X = layers.Bidirectional(LSTM(self.layer_size, return_sequences=True,
                                      dropout=0.0,
                                      activity_regularizer=l1(self.l1_norm)))(X)

        # X = layers.BatchNormalization(axis = 1, name = 'fires_decode')(X)
        X = layers.Activation('relu')(X)

        for i in range(self.num_ident_blocks):
            X = self.identity_block(X, 'decoder', string.ascii_uppercase[i + 1])

        X = layers.LayerNormalization(axis=1, name='batch_normal')(X)
        X = layers.TimeDistributed(Dense(1, activation='linear'))(X)

        self.decoder_model = Model(inputs=decoder_input, outputs=X, name='LSTM_encoder')

        outputs = self.decoder_model(sampling)

        self.vae = tf.keras.Model(inputs=encoder_input, outputs=outputs, name="vae")

        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.vae.add_loss(self.coef * kl_loss)


class model_builder_combine:

    def __init__(self,
                 input_data,
                 drop_frac=0.0,
                 layer_size=128,
                 num_ident_blocks=3,
                 l1_norm=0,
                 l1_norm_embedding=1e-3,
                 layer_steps=2,
                 embedding=16,
                 VAE=True,
                 coef=1):
        """ Developed for the combined piezoresponse and resonance loop

        :param input_data: the training dataset
        :type input_data: numpy array
        :param drop_frac: the dropout fraction
        :type drop_frac: float < 1
        :param layer_size: the depth of each LSTM layer
        :type layer_size: int
        :param num_ident_blocks: the number of LSTM ResNet blocks
        :type num_ident_blocks: int
        :param l1_norm: the parameter of l1 regularization
        :type l1_norm: float
        :param l1_norm_embedding: the parameter of l1 for embedding layer
        :type l1_norm_embedding: float
        :param layer_steps: the number of LSTM layer in each block
        :type layer_steps: int
        :param embedding: the number of channels for embedding channel
        :type embedding: int
        :param VAE: whether add the KL divergence in loss function
        :type VAE: boolean
        :param coef: the power parameters of the l1
        :type coef: float
        """
        # Sets self.mean and self.std to use in the loss function;
        #       self.mean = 0
        #       self.std = 0

        # Sets the L1 norm on the decoder/encoder layers
        self.l1_norm = l1_norm

        # Sets the fraction of dropout
        self.drop_frac = drop_frac

        # saves the shape of the input data
        self.data_shape = input_data.shape

        # Sets the number of neurons in the encoder/decoder layers
        self.layer_size = layer_size

        # Sets the number of neurons in the embedding layer
        self.embedding = embedding

        # Bool to set if the model is a VAE
        self.VAE = VAE

        # Set the magnitude of the l1 regularization on the embedding layer.
        self.l1_norm_embedding = l1_norm_embedding

        # sets the number of layers between the residual layer
        self.layer_steps = layer_steps

        self.coef = coef

        # set the number of identity block
        self.num_ident_blocks = num_ident_blocks

        self.model_constructor(input_data)

    def identity_block(self, X, name,
                       block):

        # sets the name of the conv layers
        LSTM_name_base = name + '_LSTM_Res_' + block
        bn_name_base = name + '_layer_norm_' + block

        # output for the residual layer
        X_shortcut = X

        for i in range(self.layer_steps):
            # bidirectional LSTM
            X = layers.Bidirectional(LSTM(self.layer_size,
                                          return_sequences=True,
                                          dropout=0.0,
                                          activity_regularizer=l1(self.l1_norm)),
                                     input_shape=(self.data_shape[1] * 2, 1))(X)


            X = layers.Activation('relu')(X)

        X = layers.add([X, X_shortcut])
        #    X = layers.LayerNormalization(axis = 1, name = bn_name_base + '_res_end')(X)
        X = layers.Activation('relu')(X)

        return X

    def model_constructor(self, input_data):
        # defines the input
        encoder_input = layers.Input(shape=(self.data_shape[1:]))
        X = layers.Flatten()(encoder_input)
        X = layers.RepeatVector(1)(X)
        X = layers.Permute((2, 1))(X)

        #      X = encoder_input

        for i in range(self.num_ident_blocks):
            X = self.identity_block(X, 'encoder', string.ascii_uppercase[i + 1])

        # This is in preparation for the embedding layer
        X = layers.Bidirectional(LSTM(self.layer_size,
                                      return_sequences=False,
                                      dropout=0.0,
                                      activity_regularizer=l1(self.l1_norm)),
                                 input_shape=(self.data_shape[1] * 2,
                                              1))(X)

        #     X = layers.BatchNormalization(axis=1, name='last_encode')(X)
        X = layers.Activation('relu')(X)

        #    if self.VAE:

        #    if self.VAE:
        X = layers.Dense(self.embedding, name="embedding_pre")(X)
        X = layers.Activation('relu')(X)
        Embedding_out = layers.ActivityRegularization(l1=self.l1_norm_embedding * 10 ** (self.coef))(X)
        z_mean = layers.Dense(self.embedding, name="z_mean")(Embedding_out)
        z_log_var = layers.Dense(self.embedding, name="z_log_var")(Embedding_out)

        # update the self.mean and self.std:
        #            self.mean = z_mean
        #            self.std = z_log_var

        self.encoder_model = Model(inputs=encoder_input,
                                   outputs=[Embedding_out, z_mean, z_log_var], name='LSTM_encoder')

        #      decoder_input = layers.Input(shape=(self.embedding,), name="z_sampling")
        decoder_mean = layers.Input(shape=(self.embedding,), name="z_mean")
        decoder_log = layers.Input(shape=(self.embedding,), name="z_log")
        sampling = Sampling()((decoder_mean, decoder_log))

        #         self.encoder_model = Model(inputs=encoder_input,
        #                                outputs=sampling, name='LSTM_encoder')

        z = layers.Dense(self.embedding, name="embedding")(sampling)
        z = layers.Activation('relu')(z)
        z = layers.ActivityRegularization(l1=self.l1_norm_embedding * 10 ** (self.coef))(z)

        X = layers.RepeatVector(self.data_shape[1])(z)

        X = layers.Bidirectional(LSTM(self.layer_size, return_sequences=True,
                                      dropout=0.0,
                                      activity_regularizer=l1(self.l1_norm)))(X)

        # X = layers.BatchNormalization(axis = 1, name = 'fires_decode')(X)
        X = layers.Activation('relu')(X)

        for i in range(self.num_ident_blocks):
            X = self.identity_block(X, 'decoder', string.ascii_uppercase[i + 1])

        #     X = layers.LayerNormalization(axis=1, name='batch_normal')(X)
        X = layers.TimeDistributed(Dense(2, activation='linear'))(X)

        self.decoder_model = Model(inputs=[decoder_mean, decoder_log], outputs=X, name='LSTM_encoder')

        outputs = self.decoder_model([z_mean, z_log_var])

        self.vae = tf.keras.Model(inputs=encoder_input, outputs=outputs, name="vae")

        # Add KL divergence regularization loss.
        kl_loss = -0.5 * tf.reduce_mean(z_log_var - tf.square(z_mean) - tf.exp(z_log_var) + 1)
        self.vae.add_loss(self.coef * kl_loss)


def Train(epochs, initial_epoch, epoch_per_increase, initial_beta, beta_per_increase
          , new_data, folder_, ith_epoch=None, file_path=None, batch_size=300):
    """

    :param epochs: total epochs training
    :type epochs: int
    :param epoch_per_increase: number of epochs of each beta increase
    :type epoch_per_increase: int
    :param initial_beta: initial beta value
    :type initial_beta: float
    :param beta_per_increase: beta increase for each epoch_per_increase
    :type beta_per_increase: float
    :param new_data: input data set
    :type new_data: array
    :param folder_: folder save the weights
    :type folder_: string
    :param ith_epoch: training from the ith epoch
    :type ith_epoch: int
    :param file_path: weights dictionary from ith epoch
    :type file_path: string
    :param batch_size: batch size for training
    :type batch_size: int

    """
    best_loss = float('inf')
    iteration = (epochs // epoch_per_increase) + 1
    model = []
    # filepath =folder + '/if_appear_means_bug_happens.hdf5'
    if ith_epoch is None:
        list_ = [0, iteration]
    else:
        list_ = [ith_epoch, iteration]

    for i in range(list_[0], list_[1]):

        if i == iteration - 1:
            training_epochs = epochs - epoch_per_increase * (iteration - 1)
            if training_epochs <= 0:
                break
        elif i == 0:
            training_epochs = initial_epoch
        else:
            training_epochs = epoch_per_increase

        beta = initial_beta + beta_per_increase * i
        print(beta)
        del model
        model = model_builder(np.atleast_3d(new_data), embedding=16,
                              VAE=True, l1_norm_embedding=1e-5, coef=beta)
        beta = format(beta, '.4f')
        run_id = 'beta=' + beta + '_beta_step_size=' + str(beta_per_increase) + '_' + np.str(
            model.embedding) + '_layer_size_' + np.str(
            model.layer_size) + '_l1_norm_' + np.str(model.l1_norm) + '_l1_norm_' + np.str(
            model.l1_norm_embedding) + '_VAE_' + np.str(model.VAE)
        folder = folder_ + '/' + run_id
        make_folder(folder)

        if i == ith_epoch:
            filepath = file_path

        if i > 0:
            print(filepath)
            model.vae.load_weights(filepath)

        optimizer = tf.keras.optimizers.Adam(learning_rate=3e-5)
        model.vae.compile(optimizer, loss=tf.keras.losses.MeanSquaredError())

        # sets the file path
        epoch_begin = i * epoch_per_increase
        if i > 0:
            filepath = folder + '/phase_shift_only' + beta + '_epochs_begin_' + str(initial_epoch) + '+' + np.str(
                epoch_begin) + '+{epoch:04d}' + '-{loss:.5f}.hdf5'
        else:
            filepath = folder + '/phase_shift_only' + beta + '_epochs_begin_' + np.str(
                epoch_begin) + '+{epoch:04d}' + '-{loss:.5f}.hdf5'

        # callback for saving checkpoints. Checkpoints are only saved when the model improves
        checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath, monitor='loss',
                                                     verbose=0, save_best_only=True,
                                                     save_weights_only=True, mode='min')

        hist = model.vae.fit(np.atleast_3d(new_data),
                             np.atleast_3d(new_data),
                             batch_size, epochs=training_epochs, callbacks=[checkpoint])

        min_loss = np.min(hist.history['loss'])
        user_input = folder
        directory = os.listdir(user_input)
        searchString = format(min_loss, '.5f')
        for fname in directory:  # change directory as needed
            if searchString in fname:
                f = fname
                filepath = user_input + '/' + str(f)


def get_activations(model, X=[], i=[], mode='test'):
    """    function to get the activations of a specific layer
           this function can take either a model and compute the activations or can load previously
           generated activations saved as an numpy array

    :param model: tensorflow keras model, object
    :type model: tensorflow keras model
    :param X: input data
    :type X: numpy array
    :param i: index of the layer to extract
    :type i: int
    :param mode: test or train, changes the model behavior to scale the network
    :type mode: string
    :return: array containing the output from layer i of the network
    :rtype: float
    """

    # if a string is passed loads the activations from a file
    if isinstance(model, str):
        activation = np.load(model)
        print(f'activations {model} loaded from saved file')
    else:
        # computes the output of the ith layer
        activation = get_ith_layer_output(model, np.atleast_3d(X), i, mode)

    return activation
