import numpy as np
import math
import torch
from torch.autograd import Variable


import os.path
from os import path



num_components = 1
threshold = 4.0
num_epochs = 10000


def gt_function(in_a, in_b, min_samples_per_station, stations):

    trends = [];

    if(in_a > in_b):
        temp = in_a;
        in_a = in_b;
        in_b = temp;
    
    print(str(in_a) + " " + str(in_b));

    for i in range(len(stations)):
        useful, trend = get_trend(in_a, in_b, min_samples_per_station, stations[i])
    
        if(useful):
            trends.append(trend);

    if(len(trends) == 0):
        return math.nan, math.nan;

    return np.mean(trends), np.std(trends);
   

def ground_truth(batch, min_samples_per_station, stations):

    means = [];
    stddevs = [];

    for i in range(batch.shape[0]):
        mean, stddev = gt_function(batch[i][0], batch[i][1], min_samples_per_station, stations);

        means.append(mean);
        stddevs.append(stddev);
 
    return means, stddevs;
    




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


class year_data_class:

    def __init__(self):
        self.year = 0;
        self.jan = 0.0; self.feb = 0.0; self.mar = 0.0;
        self.apr = 0.0; self.may = 0.0; self.jun = 0.0;
        self.jul = 0.0; self.aug = 0.0; self.sep = 0.0;
        self.oct = 0.0; self.nov = 0.0; self.dec = 0.0;

        year = 0;
        jan = 0.0; feb = 0.0; mar = 0.0;
        apr = 0.0; may = 0.0; jun = 0.0;
        jul = 0.0; aug = 0.0; sep = 0.0;
        oct = 0.0; nov = 0.0; dec = 0.0;

class station_data_class:

    def __init__(self):
        self.station_id = 0;
        self.year_data_list = [];

    station_id = 0;
    year_data_list = [];


def regline_slope(x, y):
    slope, intercept = np.polyfit(x, y, 1);
    return slope;



def get_trend(min_year, max_year, min_samples_per_station, sd):

    x = list()
    y = list()

    for i in range(len(sd.year_data_list)):

        if((sd.year_data_list[i].year < min_year) or (sd.year_data_list[i].year > max_year)):
            continue;

        if(sd.year_data_list[i].jan != -999):
            x.append(sd.year_data_list[i].year); 
            y.append(sd.year_data_list[i].jan);

        if(sd.year_data_list[i].feb != -999):
            x.append(sd.year_data_list[i].year); 
            y.append(sd.year_data_list[i].feb);

        if(sd.year_data_list[i].mar != -999):
            x.append(sd.year_data_list[i].year);  
            y.append(sd.year_data_list[i].mar);

        if(sd.year_data_list[i].apr != -999):
            x.append(sd.year_data_list[i].year);  
            y.append(sd.year_data_list[i].apr);

        if(sd.year_data_list[i].may != -999):
            x.append(sd.year_data_list[i].year);   
            y.append(sd.year_data_list[i].may);

        if(sd.year_data_list[i].jun != -999):
            x.append(sd.year_data_list[i].year);  
            y.append(sd.year_data_list[i].jun);

        if(sd.year_data_list[i].jul != -999):
            x.append(sd.year_data_list[i].year);   
            y.append(sd.year_data_list[i].jul);

        if(sd.year_data_list[i].aug != -999):
            x.append(sd.year_data_list[i].year);   
            y.append(sd.year_data_list[i].aug);

        if(sd.year_data_list[i].sep != -999):
            x.append(sd.year_data_list[i].year);   
            y.append(sd.year_data_list[i].sep);

        if(sd.year_data_list[i].oct != -999):
            x.append(sd.year_data_list[i].year);   
            y.append(sd.year_data_list[i].oct);

        if(sd.year_data_list[i].nov != -999):
            x.append(sd.year_data_list[i].year);  
            y.append(sd.year_data_list[i].nov);

        if(sd.year_data_list[i].dec != -999):
            x.append(sd.year_data_list[i].year);   
            y.append(sd.year_data_list[i].dec);


    if(len(x) == 0):
        #print("No valid records in date range found");
        return 0, 0.0;
    #else:
        #print(len(x));

    # Go to next station if this one hasn't enough valid xy data
    if(len(x) < min_samples_per_station):
       #print("Not enough data for station " + str(sd.station_id))
       return 0, 0.0;

    # Save this station's trend

    return 1, regline_slope(x, y);




if path.exists('stat4.postqc.CRUTEM.5.0.1.0-202109.txt'):
    file = open('stat4.postqc.CRUTEM.5.0.1.0-202109.txt');
else:
    print("Could not find file...")
    exit()




num_stations_read = 0;
min_samples_per_station = 12 * 20; # require a minimum of 20 years of data


stations = [];
trends = [];
min_year = 9999;
max_year = 0;

while(1):

    s = file.readline();
    s = s.strip();

    if(len(s) == 0):
        break;
    
    sd = station_data_class();

    sd.station_id = int(s[0:6]);
    first_year = int(s[56:60]);
    last_year = int(s[60:64]);

    num_years = 1 + last_year - first_year;

    for j in range(num_years):

        t = file.readline();
        t = t.strip();

        if(len(t) == 0):
            break;
        
        year_tokens = list();
        year_tokens = t.split();

        y = year_data_class();

        y.year = int(year_tokens[0]);

        if(y.year < min_year):
            min_year = y.year;

        if(y.year > max_year):
            max_year = y.year;

        y.jan = float(year_tokens[1]);  y.feb = float(year_tokens[2]);  y.mar = float(year_tokens[3]);
        y.apr = float(year_tokens[4]);  y.may = float(year_tokens[5]);  y.jun = float(year_tokens[6]);
        y.jul = float(year_tokens[7]);  y.aug = float(year_tokens[8]);  y.sep = float(year_tokens[9]); 
        y.oct = float(year_tokens[10]); y.nov = float(year_tokens[11]); y.dec = float(year_tokens[12]);

        sd.year_data_list.append(y);
        
    stations.append(sd);

    num_stations_read = num_stations_read + 1

    if(num_stations_read % 1000 == 0):
        break;#print(num_stations_read);


for i in range(len(stations)):
    useful, trend = get_trend(min_year, max_year, min_samples_per_station, stations[i])
    
    if(useful):
        trends.append(trend);


print(str(num_stations_read) + " stations processed altogether.");
print(str(len(trends)) + " stations used.");
print(str(np.mean(trends)) + " +/-" + str(np.std(trends)));


torch.manual_seed(123);


net = Net()

if 0:#path.exists('weights_' + str(num_components) + '_' + str(num_epochs) + '.pth'):
    net.load_state_dict(torch.load('weights_' + str(num_components) + '_' + str(num_epochs) + '.pth'))
    print("loaded file successfully")
else:
    print("training...")

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0005);
    loss_func = torch.nn.MSELoss();

    max_training_samples = 10;

    for epoch in range(num_epochs):

      batch = torch.randint(min_year, max_year + 1, (max_training_samples, num_components*2));
      batch = batch.float();
      means, stddevs = ground_truth(batch, min_samples_per_station, stations);

      gt = torch.zeros(max_training_samples, num_components*2, dtype=torch.float32);

      valid_count = 0;

      for i in range(gt.shape[0]):
          if(math.isnan(means[i]) == False and math.isnan(stddevs[i]) == False):
            valid_count = valid_count + 1;

      #print(valid_count);

      if(valid_count == 0):
        continue;

      batch_trimmed = torch.zeros(valid_count, num_components*2, dtype=torch.float32);
      gt_trimmed = torch.zeros(valid_count, num_components*2, dtype=torch.float32);
      
      index = 0;

      for i in range(gt.shape[0]):
          if(math.isnan(means[i]) == False and math.isnan(stddevs[i]) == False):
            gt_trimmed[index][0] = means[i];
            gt_trimmed[index][1] = stddevs[i];
            batch_trimmed[index] = batch[i];
            index = index + 1;

      x = Variable(batch_trimmed);
      y = Variable(gt_trimmed);

      prediction = net(x)    
      loss = loss_func(prediction, y)

      #if epoch % 500 == 0:
      print(epoch,loss);
  
      optimizer.zero_grad()   # clear gradients for next train
      loss.backward()         # backpropagation, compute gradients
      optimizer.step()        # apply gradients

    torch.save(net.state_dict(), 'weights_' + str(num_components) + '_' + str(num_epochs) + '.pth')



"""
batch = threshold*(torch.rand((10, num_components*2),dtype=torch.float32) * 2 - 1)
gt = ground_truth(batch.numpy())
prediction = net(batch).detach().numpy()

print(gt)
print("\n")
print(prediction)
"""
