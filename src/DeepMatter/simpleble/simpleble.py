import logging
import binascii
import time
import os
import sys
import torchvision.transforms as transforms
from skimage.color import gray2rgb
import numpy as np
import os
import sys
import json
import torch
from torch.utils.data import DataLoader, Dataset
from torch import nn

from PIL import Image
import numpy as np

import matplotlib.pyplot as plt
import numpy as np
import scipy
from torch.autograd import Variable

from fnmatch import fnmatch
from PIL import Image
from skimage.transform import rescale, resize, downscale_local_mean
from tqdm import tqdm
device = 'cuda' if torch.cuda.is_available() else 'cpu'

class SimpleBleClient(object):
    """This is a class implementation of a simple BLE client.
    :param iface: The Bluetooth interface on which to make the connection. On Linux, 0 means `/dev/hci0`, 1 means `/dev/hci1` and so on., defaults to 0
    :type iface: int, optional
    :param scanCallback: A function handle of the form ``callback(client, device, isNewDevice, isNewData)``, where ``client`` is a handle to the :class:`simpleble.SimpleBleClient` that invoked the callback and ``device`` is the detected :class:`simpleble.SimpleBleDevice` object. ``isNewDev`` is `True` if the device (as identified by its MAC address) has not been seen before by the scanner, and `False` otherwise. ``isNewData`` is `True` if new or updated advertising data is available, defaults to None
    :type scanCallback: function, optional
    :param notificationCallback: A function handle of the form ``callback(client, characteristic, data)``, where ``client`` is a handle to the :class:`simpleble.SimpleBleClient` that invoked the callback, ``characteristic`` is the notified :class:`bluepy.blte.Characteristic` object and data is a `bytearray` containing the updated value. Defaults to None
    :type notificationCallback: function, optional
    """

    def __init__(self, iface=0, scanCallback=None, notificationCallback=None):
        """Constructor method
        """

        self._scanner = Scanner(iface) if(scanCallback is None)\
            else Scanner().withDelegate(SimpleBleScanDelegate(scanCallback, self))
        self._iface = iface
        self._discoveredDevices = []
        self._characteristics = []
        self._connected = False
        self._connectedDevice = None
        self._notificationCallback = None

    def setScanCallback(self, callback):
        """Set the callback function to be executed when a device is detected by the client.
        :param callback: A function handle of the form ``callback(client, device, isNewDevice, isNewData)``, where ``client`` is a handle to the :class:`simpleble.SimpleBleClient` that invoked the callback and ``device`` is the detected :class:`simpleble.SimpleBleDevice` object. ``isNewDev`` is `True` if the device (as identified by its MAC address) has not been seen before by the scanner, and `False` otherwise. ``isNewData`` is `True` if new or updated advertising data is available.
        :type callback: function
        """

        self._scanner.withDelegate(
            SimpleBleScanDelegate(callback, client=self))

    def setNotificationCallback(self, callback):
        """Set the callback function to be executed when a device sends a notification to the client.
        :param callback: A function handle of the form ``callback(client, characteristic, data)``, where ``client`` is a handle to the :class:`simpleble.SimpleBleClient` that invoked the callback, ``characteristic`` is the notified :class:`bluepy.blte.Characteristic` object and data is a `bytearray` containing the updated value. Defaults to None
        :type callback: function, optional
        """
        #if(self._connectedDevice is not None):
        #    self._connectedDevice.setNotificationCallback(callback)
        #self._notificationCallback = callback

    def scan(self, timeout=10.0):
        """Scans for and returns detected nearby devices
        :param timeout: Specify how long (in seconds) the scan should last, defaults to 10.0
        :type timeout: float, optional
        :return: List of :class:`simpleble.SimpleBleDevice` objects
        :rtype: list
        """

#        #self._discoveredDevices = []
#        #scanEntries = self._scanner.scan(timeout)
#        #for scanEntry in scanEntries:
#            self._discoveredDevices.append(
#                SimpleBleDevice(
#                    client=self,
#                    addr=scanEntry.addr,
#                    iface=scanEntry.iface,
#                    data=scanEntry.getScanData(),
#                    rssi=scanEntry.rssi,
#                    connectable=scanEntry.connectable, updateCount=scanEntry.updateCount
#                )
#            )
#        return self._discoveredDevices

#    def connect(self, device):
#        """Attempts to connect client to a given :class:`simpleble.SimpleBleDevice` object and returns a bool indication of the result.
#        :param device: An instance of the device to which we want to connect. Normally acquired by calling :meth:`simpleble.SimpleBleClient.scan` or :meth:`simpleble.SimpleBleClient.searchDevice`
#        :type device: SimpleBleDevice
#        :return: `True` if connection was successful, `False` otherwise
#        :rtype: bool
#        """
#        self._connected = device.connect()
#        if(self._connected):
#            self._connectedDevice = device
#            if(self._notificationCallback is not None):
#                self._connectedDevice.setNotificationCallback(
#                    self._notificationCallback)
#        return self._connected

    def disconnect(self):
        """Drops existing connection.
        Note that the current version of the project assumes that the client can be connected to at most one device at a time.
        """
        self._connectedDevice.disconnect()
        try:
            self._scanner.stop()
        except:
            pass
        self._connectedDevice = None
        self._connected = False

    def isConnected(self):
        """Check to see if client is connected to a device
        :return: `True` if connected, `False` otherwise
        :rtype: bool
        """
        return self._connected

    def getCharacteristics(self, startHnd=1, endHnd=0xFFFF, uuids=None):
        """Returns a list containing :class:`bluepy.btle.Characteristic` objects for the peripheral. If no arguments are given, will return all characteristics. If startHnd and/or endHnd are given, the list is restricted to characteristics whose handles are within the given range.
        :param startHnd: Start index, defaults to 1
        :type startHnd: int, optional
        :param endHnd: End index, defaults to 0xFFFF
        :type endHnd: int, optional
        :param uuids: a list of UUID strings, defaults to None
        :type uuids: list, optional
        :return: List of returned :class:`bluepy.btle.Characteristic` objects
        :rtype: list
        """
        self._characteristics = self._connectedDevice.getCharacteristics(
            startHnd, endHnd, uuids
        )
        return self._characteristics

    def readCharacteristic(self, characteristic=None, uuid=None):
        """Reads the current value of the characteristic identified by either a :class:`bluepy.btle.Characteristic` object ``characteristic``, or a UUID string ``uuid``. If both are provided, then the characteristic will be read on the basis of the ``characteristic`` object. A :class:`bluepy.btle.BTLEException.GATT_ERROR` is raised if no inputs are specified or the requested characteristic was not found.
        :param characteristic: A :class:`bluepy.btle.Characteristic` object, defaults to None
        :type characteristic: :class:`bluepy.btle.Characteristic`, optional
        :param uuid: A given UUID string, defaults to None
        :type uuid: string, optional
        :raises: :class:`bluepy.btle.BTLEException.GATT_ERROR`: If no inputs are specified or the requested characteristic was not found.
        :return: The value read from the characteristic
        :rtype: bytearray
        """
        if(characteristic is None and uuid is not None):
            characteristic = self._connectedDevice.getCharacteristic(
                uuids=[uuid])[0]
        if(characteristic is None):
            raise BTLEException(
                BTLEException.GATT_ERROR, "Characteristic was either not found, given the UUID, or not specified")
        return self._connectedDevice.readCharacteristic(
            characteristic.getHandle())

    def writeCharacteristic(self, val, characteristic=None,
                            uuid=None, withResponse=False):
        """Writes the data val (of type str on Python 2.x, byte on 3.x) to the characteristic identified by either a :class:`bluepy.btle.Characteristic` object ``characteristic``, or a UUID string ``uuid``. If both are provided, then the characteristic will be read on the basis of the ``characteristic`` object. A :class:`bluepy.btle.BTLEException.GATT_ERROR` is raised if no inputs are specified or the requested characteristic was not found. If ``withResponse`` is `True`, the client will await confirmation that the write was successful from the device.
        :param val: Value to be written in characteristic
        :type val: str on Python 2.x, byte on 3.x
        :param characteristic: A :class:`bluepy.btle.Characteristic` object, defaults to None
        :type characteristic: :class:`bluepy.btle.Characteristic`, optional
        :param uuid: A given UUID string, defaults to None
        :type uuid: string, optional
        :param withResponse: If ``withResponse`` is `True`, the client will await confirmation that the write was successful from the device, defaults to False
        :type withResponse: bool, optional
        :raises: :class:`bluepy.btle.BTLEException.GATT_ERROR`: If no inputs are specified or the requested characteristic was not found.
        :return: `True` or `False` indicating success or failure of write operation, in the case that ``withResponce`` is `True`
        :rtype: bool
        """
        if(characteristic is None and uuid is not None):
            characteristic = device.getCharacteristic(uuids=[uuid])
        if(characteristic is None):
            raise BTLEException(
                BTLEException.GATT_ERROR, "Characteristic was either not found, given the UUID, or not specified")
        return self._connectedDevice.writeCharacteristic(characteristic.getHandle(), val, withResponse)

    def searchDevice(self, name=None, mac=None, timeout=10.0):
        """Searches for and returns, given it exists, a :class:`simpleble.SimpleBleDevice` device objects, based on the provided ``name`` and/or ``mac`` address. If both a ``name`` and a ``mac`` are provided, then the client will only return a device that matches both conditions.
        :par    am name: The "Complete Local Name" Generic Access Attribute (GATT) of the device, defaults to None
        :type name: str, optional
        :param mac: The MAC address of the device, defaults to None
        :type mac: str, optional
        :param timeout: Specify how long (in seconds) the scan should last, defaults to 10.0. Internally, it serves as an input to the invoked :meth:`simpleble.SimpleBleClient.scan` method.
        :type timeout: float, optional
        :raises AssertionError: If neither a ``name`` nor a ``mac`` inputs have been provided
        :return: A :class:`simpleble.SimpleBleDevice` object if search was succesfull, None otherwise
        :rtype: :class:`simpleble.SimpleBleDevice` | None
        """
        try:
            check = not (name is None)
            chekc = not (mac is None)
            assert check or chekc
        except AssertionError as e:
            print("Either a name or a mac address must be provided to find a device!")
            raise e
        mode = 0
        if(name is not None):
            mode += 1
        if(mac is not None):
            mode += 1
        # Perform initial detection attempt
        self.scan(timeout)
        for device in self._discoveredDevices:
            found = 0
            if (device.addr == mac):
                found += 1
            for (adtype, desc, value) in device.data:
                if (adtype == 9 and value == name):
                    found += 1
            if(found >= mode):
                return device
        return None

    def printFoundDevices(self):
        """Print all devices discovered during the last scan. Should only be called after a :meth:`simpleble.SimpleBleClient.scan` has been called first.
        """
        for device in self._discoveredDevices:
            print("Device %s (%s), RSSI=%d dB" %
                  (device.addr, device.addrType, device.rssi))
            for (adtype, desc, value) in device.data:
                print("  %s = %s" % (desc, value))




class select_model(object):
    """This is the class that let you choose how to train with the reuslt of the input data.
    
    :param data_loader: input data loader
    :type data_loader: torch.utils.data.DataLoader
    :param batch_size: number of data for everytime training
    :type batch_size: int
    :param images: same size of the input data
    :type images: numpy array
    """

    def __init__(self,data_loader,images,batch_size=5):
        """Construct method
        
        """
        self.input = data_loader
        self.images = images
        self.batch_size = batch_size
        
      
    def vgg_model(self):
        """Generate the result of Vgg model
        
        :return: training result
        :rtype: numpy arrray
        """
#    if model_type == 'vgg' or model_type=='both':
        vgg_model = torch.hub.load('pytorch/vision:v0.6.0', 'vgg16', pretrained=True)
        vgg_model.classifier = nn.Sequential(*[vgg_model.classifier[i] for i in range(4)])
        vgg_model.to(device)
        vgg_model.eval()
        vgg_out=np.zeros([len(self.images),4096])
        for i, data in enumerate(self.input):
             with torch.no_grad():
                print(i)
                value = data
                test_value = Variable(value.to(device))
                test_value = test_value.float()
                out_ = vgg_model(test_value)
                out_ = out_.to('cpu')
                out_ = out_.detach().numpy()
                vgg_out[i*self.batch_size:i*self.batch_size+len(out_)] = out_
#        if model_type == 'vgg':
#            print(vgg_out.shape)
        return vgg_out
            
    def resnet_model(self):
        """Generate the result of ResNet model
        
        :return: training result
        :rtype: numpy arrray
        """
#    if model_type  == 'resnet' or model_type=='both':
        res_model = torch.load('symmodel')
        res_model.to(device)
        res_out = np.zeros([len(self.images),512])
        for i, data in enumerate(self.input):
             with torch.no_grad():
                print(i)
                value = data
                test_value = Variable(value.to(device))
                test_value = test_value.float()
                out_ = res_model(test_value)
                out_ = out_.to('cpu')
                out_ = out_.detach().numpy()
                res_out[i*self.batch_size:i*self.batch_size+len(out_)] = out_

#        if model_type == 'resnet':
#            print(res_out.shape)
        return res_out
        
    def loss_function(self,model,train_iterator,optimizer):
        """Calculate the loss of the input model
        
        :param model: deep learning model
        :type model: nn.Module
        :param train_iterator: input data loader
        :type train_iterator: torch.utils.data.DataLoader
        :param optimizer: optimizer of the model
        :type optimizer: torch.optim
        
        
        :return: mse loss of autoencoder model
        :rtype: float
        """
        model.train()
        train_loss = 0
        for x in tqdm(train_iterator, leave=True, total=len(train_iterator)):
            x = x.to(device, dtype=torch.float)
            optimizer.zero_grad()
            predicted_x = model(x)
            # reconstruction loss
            loss = F.mse_loss(x, predicted_x, reduction='mean')
            # backward pass
            train_loss += loss.item()
            loss.backward()
            # update the weights
            optimizer.step()

        return train_loss

    def combine(self,N_EPOCHS=5):
        """Combine the result of Vgg and ResNet model, the Vgg should go through the Autoencoder and concatenate with
            ResNet result.
           
        :param N_EPOCHS: number of epochs for autoencoder model training
        :type N_EPOCHS: int
        :return: training result
        :rtype: numpy arrray
        """
        self.vgg_out = self.vgg_model()
        self.res_out = self.resnet_model()
        

        encoder = Encoder().to(device)
        decoder = Decoder().to(device)
        auto_model = Auto(encoder, decoder).to(device)
        optimizer = optim.Adam(auto_model.parameters(), lr=1e-4)
        train_iterator = torch.utils.data.DataLoader(self.vgg_out,
                                                     batch_size = self.batch_size,
                                                     shuffle = False)
                                                     
        for epoch in range(N_EPOCHS):

            train_loss = self.loss_function(auto_model,train_iterator,optimizer)
            train_loss /= len(train_iterator)
            print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}')
            print('.............................')

        auto_out = np.zeros([len(self.vgg_out),512])
        for i, x in enumerate(train_iterator):
            with torch.no_grad():
                value = x
                test_value = Variable(value.to(device))
                test_value = test_value.float()
                embedding = encoder(test_value)
                embedding1 = embedding.to('cpu')
                embedding1 = embedding1.detach().numpy()
        #        print(np.mean(embedding1))
                auto_out[i*self.batch_size:i*self.batch_size+len(embedding1)] = embedding1
                print(i)

        combine_out = np.concatenate((auto_out,self.res_out),axis=1)
        print(combine_out.shape)
        return combine_out
    
#    else:
#
#        raise NameError('Please insert valid model, like "vgg","resnet" or "both".')

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.dense = nn.Linear(4096,512)

    def forward(self,x):
        out = self.dense(x)
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder,self).__init__()
        self.dense = nn.Linear(512,4096)

    def forward(self,x):
        out = self.dense(x)
        return out

class Auto(nn.Module):
    def __init__(self,enc,dec):
        super().__init__()
        self.encoder = enc
        self.decoder = dec

    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
