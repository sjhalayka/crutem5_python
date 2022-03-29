import numpy as np
import math
import torch
from torch.autograd import Variable


import os.path
from os import path



num_components = 1
threshold = 4.0
num_epochs = 10000


def traditional_mul(in_a, in_b):

  out = np.zeros([num_components], np.float32)

  out[0] = in_a[0] * in_b[0]
    
  return out.T;


def ground_truth(batch):
  truth = np.zeros([batch.shape[0],num_components],np.float32);
  for i in range(batch.shape[0]):
    a = batch[i, 0:num_components]
    b = batch[i, num_components:num_components*2]
    truth[i,:] = traditional_mul(a, b);
  return truth;

class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden1 = torch.nn.Linear(num_components*2, 32*num_components)
        self.hidden2 = torch.nn.Linear(32*num_components, 16*num_components) 
        self.hidden3 = torch.nn.Linear(16*num_components, 8*num_components)
        self.predict = torch.nn.Linear(8*num_components, num_components)

    def forward(self, x):
        x = torch.tanh(self.hidden1(x))      
        x = torch.tanh(self.hidden2(x))
        x = torch.tanh(self.hidden3(x))
        x = self.predict(x)             # linear output
        return x


class year_data:
    station_id = 0;
    year = 0;
    jan = 0.0; feb = 0.0; mar = 0.0;
    apr = 0.0; may = 0.0; jun = 0.0;
    jul = 0.0; aug = 0.0; sep = 0.0;
    oct = 0.0; nov = 0.0; dec = 0.0;



def regline_slope(x, y):
    slope, intercept = np.polyfit(x, y, 1);
    return slope;



if path.exists('stat4.postqc.CRUTEM.5.0.1.0-202109.txt'):
    file = open('stat4.postqc.CRUTEM.5.0.1.0-202109.txt');
else:
    print("Could not find file...")
    exit()




num_stations_read = 0;
min_samples_per_station = 12 * 20; # require a minimum of 20 years of data
min_year = 0;
max_year = 2022;

trends = list();

while(1):

    s = file.readline();
    s = s.strip();

    if(len(s) == 0):
        break;
    
    station_id = int(s[0:6]);
    first_year = int(s[56:60]);
    last_year = int(s[60:64]);

    num_years = 1 + last_year - first_year;

    year_data_list = list();

    for j in range(num_years):

        t = file.readline();
        t = t.strip();

        if(len(t) == 0):
            break;
        
        year_tokens = list();
        year_tokens = t.split();

        y = year_data();
        y.station_id = station_id;
        y.year = int(year_tokens[0]);

        y.jan = float(year_tokens[1]);  y.feb = float(year_tokens[2]);  y.mar = float(year_tokens[3]);
        y.apr = float(year_tokens[4]);  y.may = float(year_tokens[5]);  y.jun = float(year_tokens[6]);
        y.jul = float(year_tokens[7]);  y.aug = float(year_tokens[8]);  y.sep = float(year_tokens[9]); 
        y.oct = float(year_tokens[10]); y.nov = float(year_tokens[11]); y.dec = float(year_tokens[12]);

        year_data_list.append(y);

    num_stations_read = num_stations_read + 1

    if(num_stations_read % 1000 == 0):
        print(num_stations_read);


    # done loading data from file

    x = list()
    y = list()

    for i in range(len(year_data_list)):

        if((year_data_list[i].year < min_year) or (year_data_list[i].year > max_year)):
            continue;

        if(year_data_list[i].jan != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].jan);

        if(year_data_list[i].feb != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].feb);

        if(year_data_list[i].mar != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].mar);

        if(year_data_list[i].apr != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].apr);

        if(year_data_list[i].may != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].may);

        if(year_data_list[i].jun != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].jun);

        if(year_data_list[i].jul != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].jul);

        if(year_data_list[i].aug != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].aug);

        if(year_data_list[i].sep != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].sep);

        if(year_data_list[i].oct != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].oct);

        if(year_data_list[i].nov != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].nov);

        if(year_data_list[i].dec != -999):
            x.append(year_data_list[i].year); 
            y.append(year_data_list[i].dec);


    if(len(x) == 0):
       # print("No valid records in date range found");
        continue;


    # Go to next station if this one hasn't enough valid xy data
    if(len(x) < min_samples_per_station):
       #print("Not enough data for station " + str(station_id))
        continue;
    else:
        # Save this station's trend
        trends.append(regline_slope(x, y));  #c(trends, coefficients(lm(y~x))[[2]])


print(str(num_stations_read) + " stations processed altogether.");
print(str(len(trends)) + " stations used.");
print(str(np.mean(trends)) + " +/-" + str(np.std(trends)));


exit()




net = Net()

if path.exists('weights_' + str(num_components) + '_' + str(num_epochs) + '.pth'):
    net.load_state_dict(torch.load('weights_' + str(num_components) + '_' + str(num_epochs) + '.pth'))
    print("loaded file successfully")
else:
    print("training...")

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005)
    loss_func = torch.nn.MSELoss()

    for epoch in range(num_epochs):

      batch = threshold * (torch.rand((100,num_components*2),dtype=torch.float32) * 2 - 1)

      gt = ground_truth(batch.numpy())
      x = Variable(batch)
      y = Variable(torch.from_numpy(gt))

      prediction = net(x)     
      loss = loss_func(prediction, y)

      if epoch % 500 == 0:
        print(epoch,loss)
  
      optimizer.zero_grad()   # clear gradients for next train
      loss.backward()         # backpropagation, compute gradients
      optimizer.step()        # apply gradients

    torch.save(net.state_dict(), 'weights_' + str(num_components) + '_' + str(num_epochs) + '.pth')


batch = threshold*(torch.rand((10, num_components*2),dtype=torch.float32) * 2 - 1)
gt = ground_truth(batch.numpy())
prediction = net(batch).detach().numpy()

print(gt)
print("\n")
print(prediction)


