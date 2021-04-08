import pandas as pd
import numpy as np
import simpy
import time
import datetime
from collections import OrderedDict
from SimComponents_Layout import Part, Source, Process, Sink, Monitor, Assembly

start_running = time.time()

# input data
data_all = pd.read_excel('../data/Layout_Activity_A0001.xlsx')
block_assembly = pd.read_excel('../data/Layout_BOM_A0001.xlsx')

''' ## Activity data pre-processing ## '''
''' 
convert integer to datetime
and conver datetime to integer for timeout function and calculating process time
'''
data_all['시작일'] = pd.to_datetime(data_all['시작일'], format='%Y%m%d')
data_all['종료일'] = pd.to_datetime(data_all['종료일'], format='%Y%m%d')
initial_date = data_all['시작일'].min()

data_all['시작일'] = data_all['시작일'].apply(lambda x: (x - initial_date).days)
data_all['종료일'] = data_all['종료일'].apply(lambda x: (x - initial_date).days)

'''
define variable such as process time, block code..etc. for simulating
'''
data_all['process time'] = list(data_all['종료일'] - data_all['시작일'] + 1)
data_all['호선'] = data_all['호선'].apply(lambda x: str(x))
data_all['블록'] = data_all['블록'].apply(lambda x: str(x))
data_all['block code'] = data_all['호선'] + '_' + data_all['블록']
description_list = []
for i in range(len(data_all)):
    temp = data_all.iloc[i]
    description_list.append(temp['ACT설명'][len(temp['블록']) + 1:])
data_all['data description'] = description_list
print('data pre-processing is done at ', time.time() - start_running)

'''
define shops and number of machines
'''
PE_Shelter = ['1도크쉘터', '2도크쉘터', '3도크쉘터', '의장쉘터', '특수선쉘터', '선행의장1공장쉘터', '선행의장2공장쉘터',
               '선행의장3공장쉘터', '대조립쉘터', '뉴판넬PE장쉘터', '대조립부속1동쉘터', '대조립2공장쉘터', '선행의장6공장쉘터',
               '화공설비쉘터', '판넬조립5부쉘터', '총조립SHOP쉘터', '대조5부쉘터']
server_PE_Shelter = [1, 2, 1, 2, 2, 9, 7, 1, 3, 3, 1, 2, 2, 2, 2, 1, 4, 2]

convert_to_process = {'가공소조립부 1야드' : '선각공장', '가공소조립부 2야드': '2야드 중조공장', '대조립1부': '대조립 1공장',
                      '대조립2부': '대조립 2공장', '대조립3부': '2야드 대조립공장', '의장생산부': '해양제관공장',
                      '판넬조립1부': '선각공장', '판넬조립2부': '2야드 판넬공장', '건조1부': ['1도크', '2도크'], '건조2부': '3도크',
                      '건조3부': ['8도크', '9도크'], '선행도장부': ['도장 1공장', '도장 2공장', '도장 3공장', '도장 4공장',
                                                        '도장 5공장', '도장 6공장', '도장 7공장', '도장 8공장',
                                                        '2야드 도장 1공장', '2야드 도장 2공장', '2야드 도장 3공장',
                                                        '2야드 도장 5공장', '2야드 도장 6공장'],
                      '선실생산부': '선실공장', '선행의장부': PE_Shelter, '기장부': PE_Shelter, '의장1부': PE_Shelter,
                      '의장3부': PE_Shelter, '도장1부': '도장1부', '도장2부': '도장2부', '발판지원부': '발판지원부', '외부': '외부'}

shop_list = []
for shop in convert_to_process.values():
    if type(shop) == str:
        if shop not in shop_list:
            shop_list.append(shop)
    else:  # type(process) == list
        for i in range(len(shop)):
            if shop[i] not in shop_list:
                shop_list.append(shop[i])

machine_dict = {}
for i in range(len(PE_Shelter)):
    machine_dict[PE_Shelter[i]] = server_PE_Shelter[i]

for shop in shop_list:
    if '쉘터' not in shop:
        if '도크' not in shop:
            machine_dict[shop] = 10
        elif shop == '2도크':
            machine_dict[shop] = 2
        else:
            machine_dict[shop] = 1
print('defining converting process and number of machines is done at ', time.time() - start_running)

'''
assemble block data sorting by block code
'''
block_list = list(data_all.drop_duplicates(['block code'])['block code'])

# 각 블록별 activity 개수
activity_num = []
for block_code in block_list:
    temp = data_all[data_all['block code'] == block_code]
    temp_1 = temp.sort_values(by=['시작일'], axis=0, inplace=False)
    temp = temp_1.reset_index(drop=True, inplace=False)
    activity_num.append(len(temp))

# 최대 activity 개수
max_num_of_activity = np.max(activity_num)
print('activity 개수 :', max_num_of_activity)

# SimComponents에 넣어 줄 dataframe(중복된 작업시간 처리)
columns = pd.MultiIndex.from_product([[i for i in range(max_num_of_activity + 1)],
                                      ['start_time', 'process_time', 'process', 'description', 'activity']])  # assemble을 고려할 activity

data = pd.DataFrame([], columns=columns)
idx = 0  # df에 저장된 block 개수

for block_code in block_list:
    temp = data_all[data_all['block code'] == block_code]
    temp_1 = temp.sort_values(by=['시작일'], axis=0, inplace=False)
    temp = temp_1.reset_index(drop=True)
    data.loc[block_code] = [None for _ in range(len(data.columns))]
    n = 0  # 저장된 공정 개수
    for i in range(0, len(temp)):
        activity = temp['작업부서'][i]
        data.loc[block_code][(n, 'start_time')] = temp['시작일'][i]
        data.loc[block_code][(n, 'process_time')] = temp['process time'][i]
        data.loc[block_code][(n, 'process')] = activity
        data.loc[block_code][(n, 'description')] = temp['data description'][i]
        data.loc[block_code][(n, 'activity')] = temp['공정공종'][i]
        n += 1

    data.loc[block_code][(n, 'process')] = 'Sink'
print('reassembling data is done at ', time.time() - start_running)
data.sort_values(by=[(0, 'start_time')], axis=0, inplace=True)

''' ## input data from dataframe to Part class ## '''
parts = OrderedDict()
for i in range(len(data)):
    parts[data.index[i]] = (Part(data.index[i], data.iloc[i]))

''' ## BOM data pre-processing'''
block_assembly['호선'] = block_assembly['호선'].apply(lambda x: str(x))
block_assembly['블록'] = block_assembly['블록'].apply(lambda x: str(x))
block_assembly['상위블록'] = block_assembly['상위블록'].apply(lambda x: str(x))
block_assembly['block code'] = block_assembly['호선'] + '_' + block_assembly['블록']
block_assembly['upper block code'] = block_assembly['호선'] + '_' + block_assembly['상위블록']
assembly_list = list(block_assembly.drop_duplicates(['block code'])['block code'])
assembly_upper_list = list(block_assembly.drop_duplicates(['upper block code'])['upper block code'])

'''
adding information about lower block in Part class 
it can contain multiple blocks
'''
for block_code in assembly_upper_list:
    if block_code in block_list:
        temp = block_assembly[block_assembly['upper block code'] == block_code]
        for i in range(len(temp)):
            lower_block = temp.iloc[i]['block code']
            if lower_block in block_list:
                parts[block_code].lower_block_list.append(lower_block)

'''
adding information about upper block in Part class 
'''
upper_block_data = {}
for block_code in assembly_list:
    if block_code in block_list:
        temp = block_assembly[block_assembly['block code'] == block_code]
        for i in range(len(temp)):
            upper_block = temp.iloc[i]['upper block code']
            if upper_block in block_list:
                parts[block_code].upper_block = upper_block
            if upper_block in parts:
                upper_block_part = parts.pop(block_code)
                upper_block_data[block_code] = upper_block_part
print(len(assembly_upper_list) == len(upper_block_data))

''' ## modeling ## '''
env = simpy.Environment()
model = {}

monitor = Monitor('../result/event_log_Layout_without_GIS.csv')

source = Source(env, parts, model, monitor, convert_dict=convert_to_process)
for i in range(len(shop_list) + 1):
    if i == len(shop_list):
        model['Sink'] = Sink(env, monitor)
    else:
        model[shop_list[i]] = Process(env, shop_list[i], machine_dict[shop_list[i]], model, monitor, convert_dict=convert_to_process)

model['Assembly'] = Assembly(env, upper_block_data, source, monitor)

print('modeling is done at ', time.time() - start_running)

start_simulation = time.time()
env.run()
finish_simulation = time.time()

print('#' * 80)
print("Results of simulation")
print('#' * 80)


# 코드 실행 시간
print("data pre-processing : ", start_simulation - start_running)
print("simulation execution time :", finish_simulation - start_simulation)
print("total time : ", finish_simulation - start_running)

event_tracer = monitor.save_event_tracer()
