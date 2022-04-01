import numpy as np
import math
import torch
from torch.autograd import Variable

import os.path
from os import path

from random import randrange


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(2, 32);
        self.hidden2 = torch.nn.Linear(32 , 16 );
        self.hidden3 = torch.nn.Linear(16 , 8 );
        self.predict = torch.nn.Linear(8 , 2);

    def forward(self, x):
        x = torch.tanh(self.hidden1(x));
        x = torch.tanh(self.hidden2(x));
        x = torch.tanh(self.hidden3(x));
        x = self.predict(x);   # linear output
        return x



def main():

    global_start_years = [];
    global_end_years = [];
    global_means = [];
    global_std_devs = [];

    if path.exists('span_data.txt'):
        file = open('span_data.txt');
    else:
        print("Could not find file span_data.txt")
        exit()

    while(1):

        t = file.readline();
        t = t.strip();

        if(len(t) == 0):
            break;

        tokens = list();
        tokens = t.split();
        
        global_start_years.append(float(tokens[0]));
        global_end_years.append(float(tokens[1]));
        global_means.append(float(tokens[2]));
        global_std_devs.append(float(tokens[3]));



    batch = np.zeros((len(global_start_years), 2), np.float32);
    gt = np.zeros((len(global_start_years), 2), np.float32);

    for i in range(len(global_start_years)):
        batch[i][0] = global_start_years[i];
        batch[i][1] = global_end_years[i];
        gt[i][0] = global_means[i];
        gt[i][1] = global_std_devs[i];

    print("Randomizing")

    for i in range(500000):
        index_a = randrange(len(global_start_years));
        index_b = randrange(len(global_start_years));

        temp_batchi0 = batch[index_a][0];
        temp_batchi1 = batch[index_a][1];
        temp_gti0 = gt[index_a][0];
        temp_gti1 = gt[index_a][1];

        batch[index_a][0] = batch[index_b][0];
        batch[index_a][1] = batch[index_b][1];
        gt[index_a][0] = gt[index_b][0];
        gt[index_a][1] = gt[index_b][1];

        batch[index_b][0] = temp_batchi0;
        batch[index_b][1] = temp_batchi1;
        gt[index_b][0] = temp_gti0;
        gt[index_b][1] = temp_gti1;

    print("Done randomizing")

    print("Read " + str(len(global_start_years)) + " span data."  );

    # only use a portion of the span data for training
    batch = batch[0:(len(global_start_years) - 1) // 2];
    gt = gt[0:(len(global_start_years) - 1) // 2];

    print("Using " + str(batch.shape[0]) + " span data."  );





    num_epochs = 10000;

    net = Net()

    if path.exists('weights_' + str(1) + '_' + str(num_epochs) + '.pth'):
        net.load_state_dict(torch.load('weights_' + str(1) + '_' + str(num_epochs) + '.pth'))
        print("loaded file successfully")
    else:
        print("training...")

        #optimizer = torch.optim.SGD(net.parameters(), lr=0.0001, momentum = 1)
        optimizer = torch.optim.Adam(net.parameters(), lr=0.000001);
        loss_func = torch.nn.MSELoss();

        for epoch in range(num_epochs):

          print(epoch);

          x = Variable(torch.tensor(batch));
          y = Variable(torch.tensor(gt));

          prediction = net(x)
          loss = loss_func(prediction, y)

          #if epoch % 500 == 0:
          print(epoch, loss);
  
          optimizer.zero_grad()   # clear gradients for next train
          loss.backward()         # backpropagation, compute gradients
          optimizer.step()        # apply gradients

        torch.save(net.state_dict(), 'weights_' + str(1) + '_' + str(num_epochs) + '.pth')

    batch = np.zeros((1, 2), np.float32);
    batch[0][0] = 1850;
    batch[0][1] = 2020;
    prediction = net(torch.tensor(batch)).detach();
    print("1850 - 2020 temperature anomaly trend (tenths of degree per year, or equivalently degrees per decade)");
    print(prediction)

    batch[0][0] = 1980;
    batch[0][1] = 2020;
    prediction = net(torch.tensor(batch)).detach();
    print("1980 - 2020 temperature anomaly trend (tenths of degree per year, or equivalently degrees per decade)");
    print(prediction)

    batch[0][0] = 2020;
    batch[0][1] = 2040;
    prediction = net(torch.tensor(batch)).detach();
    print("2020 - 2040 temperature anomaly trend (tenths of degree per year, or equivalently degrees per decade)");
    print(prediction)




main();



