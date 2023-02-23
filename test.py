import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

hotel_1 = pd.read_csv('H1.csv',parse_dates=True,index_col='ReservationStatusDate')
hotel_2 = pd.read_csv('H2.csv',parse_dates=True,index_col='ReservationStatusDate')

hotel_1 = hotel_1.replace(to_replace = '       NULL', 
                 value =np.NAN) 
hotel_2 = hotel_2.replace(to_replace = '       NULL', 
                 value =np.NAN) 

# Drop Company and Agent from both hotel_1 & hotel_2 datasets
hotel_1 = hotel_1.drop(['Agent','Company'],axis=1)
hotel_2 = hotel_2.drop(['Agent','Company'],axis=1)

# Fill NA values using Most frequently occuring value in that column
hotel_1['Country'] = hotel_1['Country'].fillna(hotel_1['Country'].mode()[0])

hotel_2['Country'] = hotel_2['Country'].fillna(hotel_2['Country'].mode()[0])
hotel_2['Children'] = hotel_2['Children'].fillna(hotel_2['Children'].mode()[0])

# drop arrival date month
hotel_1 = hotel_1.drop(['ArrivalDateMonth'],axis=1)
hotel_2 = hotel_2.drop(['ArrivalDateMonth'],axis=1)

# drop arrival date day of month
hotel_1 = hotel_1.drop(['ArrivalDateDayOfMonth'],axis=1)
hotel_2 = hotel_2.drop(['ArrivalDateDayOfMonth'],axis=1)

# drop reservation status
hotel_1 = hotel_1.drop(['ReservationStatus'],axis=1)
hotel_2 = hotel_2.drop(['ReservationStatus'],axis=1)

hotel_1['AssignNewRoom'] = 0
# check if the reserved room type is different from assigned room type
hotel_1.loc[hotel_1['ReservedRoomType'] != hotel_1['AssignedRoomType'], 'AssignNewRoom'] = 1
# drop older features
hotel_1 = hotel_1.drop(['AssignedRoomType', 'ReservedRoomType'], axis=1)

# replacing 1 by True and 0 by False for treatment and outcome features
hotel_1['AssignNewRoom'] = hotel_1['AssignNewRoom'].replace({1: True, 0: False})
hotel_1['IsCanceled'] = hotel_1['IsCanceled'].replace({1: True, 0: False})

t = torch.tensor(hotel_1['AssignNewRoom'].values, dtype=torch.float32).reshape(-1,1)
y = torch.tensor(hotel_1['IsCanceled'].values, dtype=torch.float32).reshape(-1,1)
concat_true = torch.cat((y,t),1)
w = torch.tensor(hotel_1['BookingChanges'].values, dtype=torch.float32).reshape(-1,1)

from models import tarnet

ate = tarnet.estimate(
    treatment = t,
    outcome = y,
    confounders = w,
)
print(ate)