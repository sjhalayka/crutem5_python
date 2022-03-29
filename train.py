import numpy as np
import math
import torch
from torch.autograd import Variable

import os.path
from os import path




num_components = 1
threshold = 4.0
num_epochs = 1000




def gt_function(in_a, in_b, min_samples_per_station, p_years, p_jans, p_febs, p_mars, p_aprs, p_mays, p_juns, p_juls, p_augs, p_seps, p_octs, p_novs, p_decs):

    trends = [];

    if(in_a > in_b):
        temp = in_a;
        in_a = in_b;
        in_b = temp;
    
    print(str(in_a) + " " + str(in_b));

    for i in range(len(global_station_ids)):
        useful, trend = get_trend(in_a, in_b, min_samples_per_station, i, p_years, p_jans, p_febs, p_mars, p_aprs, p_mays, p_juns, p_juls, p_augs, p_seps, p_octs, p_novs, p_decs)
    
        if(useful):
            trends.append(trend);

    if(len(trends) == 0):
        return math.nan, math.nan;

    return np.mean(trends), np.std(trends);
   


def ground_truth(batch, min_samples_per_stations, p_years, p_jans, p_febs, p_mars, p_aprs, p_mays, p_juns, p_juls, p_augs, p_seps, p_octs, p_novs, p_decs):

    means = [];
    stddevs = [];

    for i in range(batch.shape[0]):
        mean, stddev = gt_function(batch[i][0], batch[i][1], min_samples_per_station, p_years, p_jans, p_febs, p_mars, p_aprs, p_mays, p_juns, p_juls, p_augs, p_seps, p_octs, p_novs, p_decs);

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


def get_trend(min_year, max_year, min_samples_per_station, station_index, p_years, p_jans, p_febs, p_mars, p_aprs, p_mays, p_juns, p_juls, p_augs, p_seps, p_octs, p_novs, p_decs):

    x = list()
    y = list()

    for i in range(len(p_years[station_index])):

        if((p_years[station_index][i] < min_year) or (p_years[station_index][i] > max_year)):
            continue;

        if(p_jans[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_jans[station_index][i]);

        if(p_febs[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_febs[station_index][i]);

        if(p_mars[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_mars[station_index][i]);

        if(p_aprs[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_aprs[station_index][i]);

        if(p_mays[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_mays[station_index][i]);

        if(p_juns[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_juns[station_index][i]);

        if(p_juls[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_juls[station_index][i]);

        if(p_augs[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_augs[station_index][i]);

        if(p_seps[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_seps[station_index][i]);

        if(p_octs[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_octs[station_index][i]);

        if(p_novs[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_novs[station_index][i]);

        if(p_decs[station_index][i] != -999):
            x.append(p_years[station_index][i]);
            y.append(p_decs[station_index][i]);

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

    #slope, intercept = np.polyfit(x, y, 1);

    x_mean = 0;
    y_mean = 0;

    for i in range(len(x)):
        x_mean = x_mean + x[i];
        y_mean = y_mean + y[i];

    x_mean = x_mean / len(x);
    y_mean = y_mean / len(x);

    covariance = 0.0;
    variance = 0.0;

    for i in range(len(x)):
        z = x[i] - x_mean;
        covariance = covariance + z*(y[i] - y_mean);
        variance = variance + z*z;

    covariance = covariance / len(x);
    variance = variance / len(x);

    return 1, covariance / variance;




if path.exists('stat4.postqc.CRUTEM.5.0.1.0-202109.txt'):
    file = open('stat4.postqc.CRUTEM.5.0.1.0-202109.txt');
else:
    print("Could not find file...")
    exit()






global_station_ids = [0];
global_years = [[0]];
global_jans = [[0]];
global_febs = [[0]];
global_mars = [[0]];
global_aprs = [[0]];
global_mays = [[0]];
global_juns = [[0]];
global_juls = [[0]];
global_augs = [[0]];
global_seps = [[0]];
global_octs = [[0]];
global_novs = [[0]];
global_decs = [[0]];


num_stations_read = 0;
min_samples_per_station = 12 * 20; # require a minimum of 20 years of data


#stations = [];
trends = [];
min_year = 9999;
max_year = 0;

while(1):

    s = file.readline();
    s = s.strip();

    if(len(s) == 0):
        break;

    local_station_id = int(s[0:6]);
    first_year = int(s[56:60]);
    last_year = int(s[60:64]);

    num_years = 1 + last_year - first_year;

    local_years = [];
    local_jans = [];
    local_febs = [];
    local_mars = [];
    local_aprs = [];
    local_mays = [];
    local_juns = [];
    local_juls = [];
    local_augs = [];
    local_seps = [];
    local_octs = [];
    local_novs = [];
    local_decs = [];

    for j in range(num_years):

        t = file.readline();
        t = t.strip();

        if(len(t) == 0):
            break;
        
        year_tokens = list();
        year_tokens = t.split();

        year = int(year_tokens[0]);

        if(year < min_year):
            min_year = year;

        if(year > max_year):
            max_year = year;

        jan = float(year_tokens[1]);  feb = float(year_tokens[2]);  mar = float(year_tokens[3]);
        apr = float(year_tokens[4]);  may = float(year_tokens[5]);  jun = float(year_tokens[6]);
        jul = float(year_tokens[7]);  aug = float(year_tokens[8]);  sep = float(year_tokens[9]); 
        oct = float(year_tokens[10]); nov = float(year_tokens[11]); dec = float(year_tokens[12]);

        local_years.append(year);
        local_jans.append(jan);
        local_febs.append(feb);
        local_mars.append(mar);
        local_aprs.append(apr);
        local_mays.append(may);
        local_juns.append(jun);
        local_juls.append(jul);
        local_augs.append(aug);
        local_seps.append(sep);
        local_octs.append(oct);
        local_novs.append(nov);
        local_decs.append(dec);
        
    global_station_ids.append(local_station_id);
    global_years.append(local_years);
    global_jans.append(local_jans);
    global_febs.append(local_febs);
    global_mars.append(local_mars);
    global_aprs.append(local_aprs);
    global_mays.append(local_mays);
    global_juns.append(local_juns);
    global_juls.append(local_juls);
    global_augs.append(local_augs);
    global_seps.append(local_seps);
    global_octs.append(local_octs);
    global_novs.append(local_novs);
    global_decs.append(local_decs);



    num_stations_read = num_stations_read + 1



    if(num_stations_read % 1000 == 0):
       break;#print(num_stations_read);



"""
print(len(global_station_ids));


for i in range(len(global_station_ids)):

    print(i);

    useful, trend = get_trend(min_year, max_year, min_samples_per_station, i, global_years, global_jans, global_febs, global_mars, global_aprs, global_mays, global_juns, global_juls, global_augs, global_seps, global_octs, global_novs, global_decs )
    
    if(useful):
        trends.append(trend);


print(str(num_stations_read) + " stations processed altogether.");
print(str(len(trends)) + " stations used.");
print(str(np.mean(trends)) + " +/-" + str(np.std(trends)));


exit();
"""




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
      means, stddevs = ground_truth(batch, min_samples_per_station, global_years, global_jans, global_febs, global_mars, global_aprs, global_mays, global_juns, global_juls, global_augs, global_seps, global_octs, global_novs, global_decs);

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
