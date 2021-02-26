import os
import pandas as pd
import numpy as np
import simpy
import time
import random
from datetime import datetime

from SimComponents_rev import Source, Resource, Process, Sink, Monitor, Part
start_run = time.time()

## Pre-Processing
# DATA INPUT
data_all = pd.read_excel('../data/master_planning.xlsx')
data = data_all[['PROJECTNO', 'ACTIVITYCODE', 'LOCATIONCODE', 'PLANSTARTDATE', 'PLANFINISHDATE', 'PLANDURATION']]

# DATA PRE-PROCESSING
data = data[data['PLANSTARTDATE'].dt.year >= 2018]
data = data[data['LOCATIONCODE'] != 'OOO']

initial_date = data['PLANSTARTDATE'].min()

data['PLANSTARTDATE'] = data['PLANSTARTDATE'].apply(lambda x: (x - initial_date).days)
data['PLANFINISHDATE'] = data['PLANFINISHDATE'].apply(lambda x: (x - initial_date).days)
data['ACTIVITY'] = data['ACTIVITYCODE'].apply(lambda x: x[5:])
data['BLOCKCODE'] = data['PROJECTNO'] + ' ' + data['LOCATIONCODE']

process_list = list(data.drop_duplicates(['ACTIVITY'])['ACTIVITY'])
block_list = list(data.drop_duplicates(['BLOCKCODE'])['BLOCKCODE'])

df_part = pd.DataFrame(block_list, columns=["part"])

# 각 블록별 activity 개수
activity_num = []
for block_code in block_list:
    temp = data[data['BLOCKCODE'] == block_code]
    temp_1 = temp.sort_values(by=['PLANSTARTDATE'], axis=0, inplace=False)
    temp = temp_1.reset_index(drop=True, inplace=False)
    activity_num.append(len(temp))

## 최대 activity 개수
max_num_of_activity = np.max(activity_num)

# S-Module에 넣어 줄 dataframe(중복된 작업시간 처리)
columns = pd.MultiIndex.from_product([[i for i in range(max_num_of_activity + 1)], ['start_time', 'process_time', 'process']])
df = pd.DataFrame([], columns=columns)
idx = 0  # df에 저장된 block 개수

for block_code in block_list:
    temp = data[data['BLOCKCODE'] == block_code]
    temp_1 = temp.sort_values(by=['PLANSTARTDATE'], axis=0, inplace=False)
    temp = temp_1.reset_index(drop=True)
    df.loc[block_code] = [None for _ in range(len(df.columns))]
    n = 0  # 저장된 공정 개수
    for i in range(0, len(temp)):
        activity = temp['ACTIVITY'][i]
        df.loc[block_code][(n, 'start_time')] = temp['PLANSTARTDATE'][i]
        df.loc[block_code][(n, 'process_time')] = temp['PLANDURATION'][i]
        df.loc[block_code][(n, 'process')] = activity
        n += 1

    df.loc[block_code][(n, 'process')] = 'Sink'

print(df)
df.sort_values(by=[(0, 'start_time')], axis=0, inplace=True)

parts = []
for i in range(len(df)):
    parts.append(Part(df.index[i], df.iloc[i]))

env = simpy.Environment()
model = {}
server_num = np.full(len(process_list), 1)

Monitor = Monitor('../result/event_log_master_plan_with_tp.csv')

# network -> distance data
network_dist = pd.read_excel('../network/master_plan_distance.xlsx')
network_dist = network_dist.set_index('Unnamed: 0', drop=True)
# Resource
tp_info = {}
tp_num = 10
for i in range(tp_num):
    tp_info["TP_{0}".format(i+1)] = {"capa": 100, "v_loaded": 0.5, "v_unloaded": 1.0}
Resource = Resource(env, model, Monitor, tp_info=tp_info, network=network_dist)

source = Source(env, parts, model, Monitor)
for i in range(len(process_list) + 1):
    if i == len(process_list):
        model['Sink'] = Sink(env, Monitor)
    else:
        model[process_list[i]] = Process(env, process_list[i], server_num[i], model, Monitor, resource=Resource, network=network_dist, transporter=True)

start = time.time()
env.run()
finish = time.time()
print('#' * 80)
print("Results of simulation")
print('#' * 80)


# 코드 실행 시간
print("data pre-processing : ", start - start_run)
print("simulation execution time :", finish - start)
print("total time : ", finish - start_run)

event_tracer = Monitor.save_event_tracer()
print("#" * 80)
print("# Master Plan using 10 Transporters")
print("# Total simulation time : ", model['Sink'].last_arrival)
print('# Total moving time and utilization of each transporter')
tp_df = pd.DataFrame([], columns=['distance', 'Utilization'])
for process in process_list:
    if len(model[process].tp_store.items) > 0:
        for tp in model[process].tp_store.items:
            tp_df.loc[tp.name] = [tp.distance, tp.moving_time/model['Sink'].last_arrival]
tp_df = tp_df.sort_index()
# tp_df = tp_df.reindex(index=['TP_1', 'TP_2', 'TP_3', 'TP_4', 'TP_5', 'TP_6', 'TP_7', 'TP_8', 'TP_9', 'TP_10']
print(tp_df)

