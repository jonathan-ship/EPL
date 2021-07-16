import simpy
import pandas as pd
import numpy as np
import time
import random
from SimComponents_KSOE_Layout import Resource, Part, Sink, StockYard, Monitor, Process
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import math
import os

save_path = './result/Series40_and_adding_stock'
if not os.path.exists(save_path):
   os.makedirs(save_path)

start = time.time()
print("## Start preprocessing... ")

# 적치장
# stocks = ['E7', 'E8', 'E9', 'E4', 'E6', 'E5', 'Y81', 'Y9', 'Y4', 'Y7', 'Y5', 'Y2', 'Y1', 'Y3', 'Y6', 'Y8']
## 가상 적치장 Y1V
stocks = ['E7', 'E8', 'E9', 'E4', 'E6', 'E5', 'Y81', 'Y9', 'Y4', 'Y7', 'Y5', 'Y2', 'Y1', 'Y3', 'Y6', 'Y8', 'Y1V']
stock_area_data = pd.read_excel('./data/Stockyard_area.xlsx')
stock_area = {}
for i in range(len(stock_area_data)):
    temp = stock_area_data.iloc[i]
    stock_area[temp['Stock']] = temp['area']
stock_area['Y9'] = 3200  # 최소값
# 1야드 내, 가상의 적치장
stock_area['Y1V'] = 22650

# 쉘터
PE_Shelter = ['1도크PE', '2도크PE', '3도크PE', '의장쉘터', '특수선쉘터', '선행의장1공장쉘터', '선행의장2공장쉘터',
               '선행의장3공장쉘터', '대조립쉘터', '뉴판넬PE장쉘터', '대조립부속1동쉘터', '대조립2공장쉘터', '선행의장6공장쉘터',
               '화공설비쉘터', '판넬조립5부쉘터', '총조립SHOP쉘터', '대조립5부쉘터', '8도크PE', '9도크PE']

Shelter_area_data = pd.read_excel('./data/Shelter_area.xlsx')
shelter_area = {}
for i in range(len(Shelter_area_data)):
    temp = Shelter_area_data.iloc[i]
    shelter_area[temp['Shelter']] = temp['area']
shelter_area['1도크PE'] = 4675  # 평균값
shelter_area['2도크PE'] = 4675
shelter_area['3도크PE'] = 4675
shelter_area['8도크PE'] = 4675
shelter_area['9도크PE'] = 4675

# server_PE_Shelter = [1, 2, 1, 2, 2, 9, 7, 1, 3, 3, 1, 2, 2, 2, 2, 1, 4, 2]

# data - real matching
convert_to_process = {'가공소조립부 1야드' : '선각공장', '가공소조립부 2야드': '2야드 중조공장', '대조립1부': '대조립 1공장',
                      '대조립2부': '대조립 2공장', '대조립3부': '2야드 대조립공장', '의장생산부': '해양제관공장',
                      '판넬조립1부': '선각공장', '판넬조립2부': '2야드 판넬공장', '건조1부': ['1도크', '2도크'], '건조2부': '3도크',
                      '건조3부': ['8도크', '9도크'], '선행도장부': ['도장 1공장', '도장 2공장', '도장 3공장', '도장 4공장',
                                                        '도장 5공장', '도장 6공장', '도장 7공장', '도장 8공장',
                                                        '2야드 도장 1공장', '2야드 도장 2공장', '2야드 도장 3공장',
                                                        '2야드 도장 5공장', '2야드 도장 6공장'],
                      '선실생산부': '선실공장', '선행의장부': PE_Shelter, '기장부': PE_Shelter, '의장1부': PE_Shelter,
                      '의장2부': PE_Shelter, '의장3부': PE_Shelter, '외부': '외부'}

# 각 공장의 입출구, list[0] : 입구 // list[1] : 출구
process_inout = {}
mapping_table = pd.read_excel('./data/process_gis_mapping_table.xlsx')
for i in range(len(mapping_table)):
    temp = mapping_table.iloc[i]
    process_inout[temp['LOC']] = [temp['IN'], temp['OUT']]
process_inout['Virtual'] = ['Virtual', 'Virtual']
process_inout['Painting'] = ['Painting', 'Painting']
process_inout['Shelter'] = ['Shelter', 'Shelter']
process_inout['Y1V'] = ['Y1V', 'Y1V']

# 호선 - 도크
dock_mapping = pd.read_excel('./data/호선도크.xlsx')
dock_mapping = dict(dock_mapping.transpose())

process_list = list(process_inout.keys())

# yard data
# yard = pd.read_excel('./data/호선도크.xlsx')


# block_data
data_all = pd.read_excel('./data/Layout_data_series40.xlsx')
bom_all = pd.read_excel('./data/Layout_BOM.xlsx')
bom_all['child code'] = bom_all['호선'] + '_' + bom_all['블록']
bom_all.loc[:, '상위블록'] = bom_all['상위블록'].apply(lambda x: str(x))
bom_all['parent code'] = bom_all['호선'] + '_' + bom_all['상위블록']
block_group = data_all.groupby(data_all['series_block_code'])
bom_group_parent = bom_all.groupby(bom_all['parent code'])
bom_group_child = bom_all.groupby(bom_all['child code'])
block_list = list(data_all.drop_duplicates(['series_block_code'])['series_block_code'])

print("## Finish reading data at", time.time() - start)

columns = pd.MultiIndex.from_product([[i for i in range(8)], ['start_time', 'process_time', 'process', 'work']])
block_info = {}

for block_code in block_list:
    block_info[block_code] = {}
    block = block_group.get_group(block_code)
    block = block.sort_values(by=['start_date'], ascending=True)
    block = block.reset_index(drop=True)
    # size and area
    block_info[block_code]['size'] = block['size'][0]
    block_info[block_code]['area'] = block['area'][0]
    # information about process
    data = [None for _ in range(32)]
    for idx in range(len(block) + 1):
        if idx < len(block):
            temp = block.iloc[idx]
            data[4*idx] = temp['start_date']
            proc_time = temp['finish_date'] - temp['start_date']
            data[4 * idx + 1] = proc_time if proc_time > 0 else 0
            data[4*idx + 2] = temp['location']
            data[4*idx + 3] = temp['process_code']
        else:
            data[4*idx + 2] = 'Sink'
    block_info[block_code]['data'] = pd.Series(data, index=columns)
    # information about Assembly
    series = block_code.split('_')[0]
    code = block_code.split('_')[1]
    ## parent block
    if block_code in list(bom_all['child code']):
        parent = bom_group_child.get_group(block_code)  # child code = block code인 애들
        temp = list(parent['parent code'])
        if temp[0] in block_list:
            # 상위 블록이 있으면
            block_info[block_code]['parent_block'] = temp[0]
        else: block_info[block_code]['parent_block'] = None
    else:
        block_info[block_code]['parent_block'] = None

    ## child block
    if block_code in list(bom_all['parent code']):

        child = bom_group_parent.get_group(block_code)  # parent code = block code인 애들
        temp = list(child['child code'])
        if len(temp):
            child_temp = []
            for child_code in temp:
                if child_code in block_list:
                    child_temp.append(child_code)

        if len(child_temp) == 0:
            block_info[block_code]['child_block'] = None
        else:
            block_info[block_code]['child_block'] = child_temp
    else:
        block_info[block_code]['child_block'] = None

print("## Finish integrating Activity data and Bom data, and Start reading Network(distance) data at", time.time() - start)

# network data
network = {}
#for i in range(12, 41):
# from_to_matrix = pd.read_excel('./network/distance_above_{0}_meters.xlsx'.format(i), index_col=0)
from_to_matrix = pd.read_excel('./network/distance_above_12_meters.xlsx', index_col=0)
# Virtual Stockyard까지의 거리 추가 --> 거리 = 0 (가상의 공간이므로)
from_to_matrix.loc['Virtual'] = 0.0
from_to_matrix.loc['Source'] = 0.0
from_to_matrix.loc['Sink'] = 0.0
from_to_matrix.loc['Painting'] = 0.0
from_to_matrix.loc['Shelter'] = 0.0
from_to_matrix.loc['Y1V'] = 0.0
from_to_matrix['Virtual'] = 0.0
from_to_matrix['Source'] = 0.0
from_to_matrix['Sink'] = 0.0
from_to_matrix['Painting'] = 0.0
from_to_matrix['Shelter'] = 0.0
from_to_matrix['Y1V'] = 0.0
network[12] = from_to_matrix

print("## Finish data preprocessing and Start modeling at", time.time() - start)

monitor = Monitor(save_path+'/result_series40_adding_stock.csv', network)

env = simpy.Environment()
parts = {}
processes = {}
stock_yard = {}

tp_info = {}
tp_num = 30
v_loaded = 3 * 1000 * 24  # m / day
v_unloaded = 10 * 1000 * 24
for i in range(tp_num):
    tp_info["TP_{0}".format(i+1)] = {"v_loaded": v_loaded, "v_unloaded": v_unloaded}
resource = Resource(env, processes, stock_yard, monitor, tp_info=tp_info, network=network, inout=process_inout)


# Block modeling
for block_code in block_info:
    series = block_code[:5]
    if series[3] == 0:  # 한 자리수 호선
        dock = dock_mapping[int(series[4]) - 1]['도크']
    else:  # 두 자리수 호선
        dock = dock_mapping[int(series[-2:]) - 1]['도크']
    parts[block_code] = Part(block_code, env, block_info[block_code]['data'], processes, monitor, resource=resource,
                             from_to_matrix=network, size=block_info[block_code]['size'],
                             area=block_info[block_code]['area'], child=block_info[block_code]['child_block'],
                             parent=block_info[block_code]['parent_block'], stocks=stock_yard, Inout=process_inout,
                             convert_to_process=convert_to_process, dock=dock)

Painting_process = ['도장 1공장', '도장 2공장', '도장 3공장', '도장 4공장', '도장 5공장', '도장 6공장', '도장 7공장', '도장 8공장',
                    '2야드 도장 1공장', '2야드 도장 2공장', '2야드 도장 3공장', '2야드 도장 5공장', '2야드 도장 6공장']

# Process modeling
for process in process_list:
    if process in PE_Shelter:
        processes[process] = Process(env, process, 5000, processes, parts, monitor, resource=resource,
                                     convert_dict=convert_to_process, area=shelter_area[process])
    elif process in Painting_process:
        processes[process] = Process(env, process, 5000, processes, parts, monitor, resource=resource,
                                     convert_dict=convert_to_process, area=3000)
    else:
        processes[process] = Process(env, process, 5000, processes, parts, monitor, resource=resource,
                                     convert_dict=convert_to_process)
processes['Sink'] = Sink(env, processes, parts, monitor)
processes['Painting'] = Process(env, 'Painting', 5000, processes, parts, monitor, resource=resource,
                                convert_dict=convert_to_process)
processes['Shelter'] = Process(env, 'Shelter', 5000, processes, parts, monitor, resource=resource,
                               convert_dict=convert_to_process)


# StockYard modeling
for stock in stocks:
    stock_yard[stock] = StockYard(env, stock, parts, monitor, capacity=stock_area[stock])

stock_yard['Virtual'] = StockYard(env, 'Virtual', parts, monitor, capacity=float('inf'))

start_sim = time.time()
print("## Preprocessing and Modeling are ended, and Start to run at", start_sim-start)
env.run()
finish_sim = time.time()
print("Execution time:", finish_sim-start_sim)
monitor.save_event_tracer()
monitor.save_road_info()

from matplotlib import font_manager, rc

print("## Start post-processing at", time.time() - start)

print("number of part created = ", monitor.created)
print("number of completed = ", monitor.completed)


font_name = font_manager.FontProperties(fname="C:\Windows\Fonts\H2GTRM.TTF").get_name()
rc('font', family=font_name)
save_path_stock = save_path + '/Stock'
if not os.path.exists(save_path_stock):
   os.makedirs(save_path_stock)

stocks.append('Virtual')
for stock in stocks:
     each_stock_yard = stock_yard[stock]
     stock_area = each_stock_yard.capacity
     event_area = each_stock_yard.event_area
     if len(event_area) > 0:
         event_time = each_stock_yard.event_time
         fig, ax = plt.subplots()
         if stock == 'Virtual':
             line = ax.plot(event_time, event_area, color="blue", marker="o")
             ax.set_ylabel("Area")
             ax.set_ylim([0, max(event_area) * 1.2])
             max_area_unit = math.ceil(max(event_area) / 10)
             area_digit_num = len(str(max_area_unit)) - 1
             area_digit = math.ceil(max_area_unit / math.pow(10, area_digit_num)) * math.pow(10, area_digit_num)
             ax.yaxis.set_major_locator(ticker.MultipleLocator(area_digit))
         else:
             line = ax.plot(event_time, event_area, color="blue", marker="o")
             ax.set_ylabel("Ratio")
             ax.set_ylim([0, stock_area*1.05])
             area_unit = math.ceil(stock_area/10)
             ax.yaxis.set_major_locator(ticker.MultipleLocator(area_unit))

         ax.set_title("{0} occupied area".format(stock), fontsize=13, fontweight="bold")
         ax.set_xlabel("Time")

         filepath = save_path_stock + '/{0}.png'.format(stock)
         plt.savefig(filepath, dpi=600, transparent=True)
         plt.show()
         print("### {0} ###".format(stock))

save_path_painting = save_path + '/Painting'
if not os.path.exists(save_path_painting):
   os.makedirs(save_path_painting)

Painting_process.append("Painting")
for paint in Painting_process:
    each_painting_process = processes[paint]
    stock_area = each_painting_process.area
    event_area = each_painting_process.event_area
    if len(event_area) > 0:
        # area_ratio = list(map(lambda x: x / stock_area, event_area))
        event_time = each_painting_process.event_time
        event_block_num = each_painting_process.event_block_num
        fig, ax1 = plt.subplots()  # ax1: area - bar graph / ax2: # of blocks - line graph
        ax2 = ax1.twinx()
        if paint != "Painting":
            bar = ax1.bar(event_time, event_area, color="orange", label="occupied area", width=5)
            ax1.set_ylim([0, stock_area*1.05])
            area_unit = math.ceil(stock_area / 10)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(area_unit))
        else:
            bar = ax1.bar(event_time, event_area, color="orange", label="occupied area", width=5)
            ax1.set_ylim([0, max(event_area) * 1.2])
            max_area_unit = math.ceil(max(event_area) / 10)
            area_digit_num = len(str(max_area_unit)) - 1
            area_digit = math.ceil(max_area_unit / math.pow(10, area_digit_num)) * math.pow(10, area_digit_num)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(area_digit))

        block_num_unit = math.ceil(max(event_block_num) / 10)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(block_num_unit))

        line = ax2.plot(event_time, event_block_num, color="cornflowerblue", label="# of Blocks", marker="o")
        ax1.set_title("{0} Results".format(paint), fontsize=13, fontweight="bold")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Area")
        ax2.set_ylabel("Number")

        ax2.set_ylim([0, max(event_block_num)*1.2])

        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, shadow=True, fancybox=True)

        filepath = save_path_painting + '/{0}.png'.format(paint)
        plt.savefig(filepath, dpi=600, transparent=True)
        plt.show()

        print("### {0} ###".format(paint))


save_path_shelter = save_path + '/Shelter'
if not os.path.exists(save_path_shelter):
   os.makedirs(save_path_shelter)

PE_Shelter.append("Shelter")
for shelter in PE_Shelter:
    each_shelter = processes[shelter]
    shelter_area = each_shelter.area
    event_area = each_shelter.event_area
    if len(event_area) > 0:
        event_time = each_shelter.event_time
        event_block_num = each_shelter.event_block_num
        fig, ax1 = plt.subplots()  # ax1: area - bar graph / ax2: # of blocks - line graph
        ax2 = ax1.twinx()
        ax2.set_ylim([0, max(event_block_num) * 1.2])
        if shelter != "Shelter":
            bar = ax1.bar(event_time, event_area, color="orange", label="occupied area", width=5)
            ax1.set_ylim([0, shelter_area])
            area_unit = math.ceil(shelter_area / 10)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(area_unit))
        else:
            bar = ax1.bar(event_time, event_area, color="orange", label="occupied area", width=5)
            ax1.set_ylim([0, max(event_area) * 1.2])
            max_area_unit = math.ceil(max(event_area) / 10)
            area_digit_num = len(str(max_area_unit)) - 1
            area_digit = math.ceil(max_area_unit / math.pow(10, area_digit_num)) * math.pow(10, area_digit_num)
            ax1.yaxis.set_major_locator(ticker.MultipleLocator(area_digit))

        block_num_unit = math.ceil(max(event_block_num) / 10)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(block_num_unit))
        line = ax2.plot(event_time, event_block_num, color="cornflowerblue", label="# of Blocks", marker="o")
        ax1.set_title("{0} Results".format(shelter), fontsize=13, fontweight="bold")
        ax1.set_xlabel("Time")
        ax1.set_ylabel("Area")
        ax2.set_ylabel("Number")

        fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax1.transAxes, shadow=True, fancybox=True)

        filepath = save_path_shelter + '/{0}.png'.format(shelter)
        plt.savefig(filepath, dpi=600, transparent=True)
        plt.show()

        print("### {0} ###".format(shelter))


part_moving_distance = {}
## block 이동 거리
for part in parts:
    each_part = parts[part]
    part_moving_distance[part] = {}
    if len(each_part.moving_distance_w_TP) > 0:
        part_moving_distance[part]['mean_distance'] = np.mean(each_part.moving_distance_w_TP)
        part_moving_distance[part]['total_distance'] = sum(each_part.moving_distance_w_TP)
    else:
        part_moving_distance[part]['mean_distance'] = 0.0
        part_moving_distance[part]['total_distance'] = 0.0

info_part_moving = pd.DataFrame(part_moving_distance)
info_part_moving = info_part_moving.transpose()
info_part_moving.to_excel(save_path + '/part_distance.xlsx')


tp_time = [i for i in range(math.ceil(processes['Sink'].last_arrival)+1)]
tp_unloaded_distance = [0 for _ in range(len(tp_time))]  # 상차 이동 거리
tp_loaded_distance = tp_unloaded_distance[:]  # 하차 이동 거리
tp_used_time_loaded = tp_unloaded_distance[:]  # 날짜 별 사용 시간 (상차)
tp_used_time_unloaded = tp_unloaded_distance[:]  # 날짜 별 사용 시간 (하차)
for tp in resource.tp_post_processing.keys():
    each_tp = resource.tp_post_processing[tp]
    ## loaded
    loaded_time = each_tp['loaded']['moving_time']
    for i in range(len(loaded_time)):
        start_time = int(round(loaded_time[i][0]))
        finish_time = int(round(loaded_time[i][1]))
        if start_time == finish_time:
            tp_loaded_distance[start_time] += each_tp['loaded']['moving_distance'][i]
            tp_used_time_loaded[start_time] += loaded_time[i][1] - loaded_time[i][0]
        else:  # 시간 비례하여 각 날짜에 포함
            day_1 = finish_time - loaded_time[i][0]  # 첫째날에 이동한 총 시간
            tp_loaded_distance[start_time] += day_1 * 3 * 1000 * 24
            tp_used_time_loaded[start_time] += day_1
            day_2 = loaded_time[i][1] - finish_time  # 둘째날에 이동한 총 시간
            tp_loaded_distance[finish_time] += day_2 * 3 * 1000 * 24
            tp_used_time_loaded[finish_time] += day_2

    ## unloaded
    unloaded_time = each_tp['unloaded']['moving_time']
    for i in range(len(unloaded_time)):
        start_time = int(round(unloaded_time[i][0]))
        finish_time = int(round(unloaded_time[i][1]))
        if start_time == finish_time:
            tp_unloaded_distance[start_time] += each_tp['unloaded']['moving_distance'][i]
            tp_used_time_unloaded[start_time] += unloaded_time[i][1] - unloaded_time[i][0]
        else:  # 시간 비례하여 각 날짜에 포함
            day_1 = finish_time - unloaded_time[i][0]  # 첫째날에 이동한 총 시간
            tp_unloaded_distance[start_time] += day_1 * 10 * 1000 * 24
            tp_used_time_loaded[start_time] += day_1
            day_2 = unloaded_time[i][1] - finish_time  # 둘째날에 이동한 총 시간
            tp_unloaded_distance[finish_time] += day_2 * 10 * 1000 * 24
            tp_used_time_loaded[finish_time] += day_2

## ax1: unloaded time / ax2: loaded_time
fig, ax = plt.subplots()
tp_used_time_loaded_hour = list(map(lambda x: x*24, tp_used_time_loaded))
tp_used_time_unloaded_hour = list(map(lambda x: x*24, tp_used_time_unloaded))
unloaded_line_time = ax.plot(tp_time, tp_used_time_unloaded_hour, color="blue", marker=".", label="Unloaded")
loaded_bar_time = ax.bar(tp_time, tp_used_time_loaded_hour, color="red", width=5, label="Loaded")
ax.set_title("T/P time", fontsize=13, fontweight="bold")
ax.set_xlabel("Time")
ax.set_ylabel("Used Time [hr]")
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, shadow=True, fancybox=True)
filepath = save_path + '/TP_time.png'
plt.savefig(filepath, dpi=600, transparent=True)
plt.show()

fig, ax = plt.subplots()
tp_used_distance_loaded_km = list(map(lambda x: x*0.001, tp_loaded_distance))
tp_used_distance_unloaded_km = list(map(lambda x: x*0.001, tp_unloaded_distance))
unloaded_line_distance = ax.plot(tp_time, tp_used_distance_unloaded_km, color="blue", marker=".", label="Unloaded")
loaded_bar_distance = ax.bar(tp_time, tp_used_distance_loaded_km, color="red", width=5, label="Loaded")
ax.set_title("T/P Distance", fontsize=13, fontweight="bold")
ax.set_xlabel("Time")
ax.set_ylabel("Distance [km]")
fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, shadow=True, fancybox=True)
filepath = save_path + '/TP_distance.png'
plt.savefig(filepath, dpi=600, transparent=True)
plt.show()

print("number of part created = ", monitor.created)
print("number of completed = ", monitor.completed)