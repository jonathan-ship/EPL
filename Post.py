import json, math, os, sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import font_manager, rc

font_name = font_manager.FontProperties(fname="C:\Windows\Fonts\H2GTRM.TTF").get_name()
rc('font', family=font_name)


def tp_index(event_tracer, tp_info, result_path):  # get get tp moving distance and time each day
    tp_time = [i for i in range(math.ceil(max(event_tracer['Time'])) + 1)]
    tp_unloaded_distance = [0 for _ in range(len(tp_time))]  # 상차 이동 거리
    tp_loaded_distance = tp_unloaded_distance[:]  # 하차 이동 거리
    tp_used_time_loaded = tp_unloaded_distance[:]  # 날짜 별 사용 시간 (상차)
    tp_used_time_unloaded = tp_unloaded_distance[:]  # 날짜 별 사용 시간 (하차)
    for tp in tp_info.keys():
        each_tp = tp_info[tp]
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
    tp_used_time_loaded_hour = list(map(lambda x: x * 24, tp_used_time_loaded))
    tp_used_time_unloaded_hour = list(map(lambda x: x * 24, tp_used_time_unloaded))
    unloaded_line_time = ax.plot(tp_time, tp_used_time_unloaded_hour, color="blue", marker=".", label="Unloaded")
    loaded_bar_time = ax.bar(tp_time, tp_used_time_loaded_hour, color="red", width=5, label="Loaded")
    ax.set_title("T/P time", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Used Time [hr]")
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, shadow=True, fancybox=True)
    filepath = result_path + '/TP_time.png'
    plt.savefig(filepath, dpi=600, transparent=True)
    # plt.show()

    fig, ax = plt.subplots()
    tp_used_distance_loaded_km = list(map(lambda x: x * 0.001, tp_loaded_distance))
    tp_used_distance_unloaded_km = list(map(lambda x: x * 0.001, tp_unloaded_distance))
    unloaded_line_distance = ax.plot(tp_time, tp_used_distance_unloaded_km, color="blue", marker=".", label="Unloaded")
    loaded_bar_distance = ax.bar(tp_time, tp_used_distance_loaded_km, color="red", width=5, label="Loaded")
    ax.set_title("T/P Distance", fontsize=13, fontweight="bold")
    ax.set_xlabel("Time")
    ax.set_ylabel("Distance [km]")
    fig.legend(loc=1, bbox_to_anchor=(1, 1), bbox_transform=ax.transAxes, shadow=True, fancybox=True)
    filepath = result_path + '/TP_distance.png'
    plt.savefig(filepath, dpi=600, transparent=True)
    # plt.show()


def calculate_stock_occupied_area(result_path, event_tracer, input_data, stock_capacity):
    save_path_stock = result_path + '/Stock'
    if not os.path.exists(save_path_stock):
        os.makedirs(save_path_stock)

    stock_event = event_tracer[event_tracer['Process_Type'] == 'Stock']
    stock_list = list(np.unique(list(stock_event['Process'].dropna())))

    if input_data['stock_virtual']:
        stock_list.append("Stock")
        stock_capacity["Stock"] = float("inf")

    for stock in stock_list:
        each_stock_event = stock_event[stock_event['Process'] == stock]
        stock_area = stock_capacity[stock]
        event_area = list(each_stock_event['Area'])
        if len(event_area) > 0:
            event_time = list(each_stock_event['Time'])
            fig, ax = plt.subplots()
            if stock == 'Stock':
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
                ax.set_ylim([0, stock_area * 1.05])
                area_unit = math.ceil(stock_area / 10)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(area_unit))

            ax.set_title("{0} occupied area".format(stock), fontsize=13, fontweight="bold")
            ax.set_xlabel("Time")

            filepath = save_path_stock + '/{0}.png'.format(stock)
            plt.savefig(filepath, dpi=600, transparent=True)
            # plt.show()
            print("### {0} ###".format(stock))


def calculate_painting_occupied_area(result_path, event_tracer, input_data, process_capacity):
    save_path_painting = result_path + '/Painting'
    if not os.path.exists(save_path_painting):
        os.makedirs(save_path_painting)

    painting_event = event_tracer[event_tracer['Process_Type'] == 'Painting']
    painting_list = list(np.unique(list(painting_event['Process'].dropna())))

    if input_data['painting_virtual']:
        painting_list.append("Painting")
        process_capacity["Painting"] = float("inf")

    for painting in painting_list:
        each_painting_event = painting_event[painting_event['Process'] == painting]
        painting_area = process_capacity[painting]
        event_area = list(each_painting_event['Area'])
        if len(event_area) > 0:
            event_time = list(each_painting_event['Time'])
            fig, ax = plt.subplots()
            if painting == 'Painting':
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
                ax.set_ylim([0, painting_area * 1.05])
                area_unit = math.ceil(painting_area / 10)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(area_unit))

            ax.set_title("{0} occupied area".format(painting), fontsize=13, fontweight="bold")
            ax.set_xlabel("Time")

            filepath = save_path_painting + '/{0}.png'.format(painting)
            plt.savefig(filepath, dpi=600, transparent=True)
            # plt.show()
            print("### {0} ###".format(painting))


def calculate_shelter_occupied_area(result_path, event_tracer, input_data, process_capacity):
    save_path_shelter = result_path + '/Shelter'
    if not os.path.exists(save_path_shelter):
        os.makedirs(save_path_shelter)

    shelter_event = event_tracer[event_tracer['Process_Type'] == 'Shelter']
    shelter_list = list(np.unique(list(shelter_event['Process'].dropna())))

    if input_data['shelter_virtual']:
        shelter_list.append("Shelter")
        process_capacity["Shelter"] = float("inf")

    for shelter in shelter_list:
        each_shelter_event = shelter_event[shelter_event['Process'] == shelter]
        shelter_area = process_capacity[shelter]
        event_area = list(each_shelter_event['Area'])
        if len(event_area) > 0:
            event_time = list(each_shelter_event['Time'])
            fig, ax = plt.subplots()
            if shelter == 'Shelter':
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
                ax.set_ylim([0, shelter_area * 1.05])
                area_unit = math.ceil(shelter_area / 10)
                ax.yaxis.set_major_locator(ticker.MultipleLocator(area_unit))

            ax.set_title("{0} occupied area".format(shelter), fontsize=13, fontweight="bold")
            ax.set_xlabel("Time")

            filepath = save_path_shelter + '/{0}.png'.format(shelter)
            plt.savefig(filepath, dpi=600, transparent=True)
            # plt.show()
            print("### {0} ###".format(shelter))


def post_main(result_path):
    with open(result_path, 'r') as f:
        result_path = json.load(f)

    event_tracer = pd.read_csv(result_path['event_tracer'])

    with open(result_path['input_path'], 'r') as f:
        input_data = json.load(f)

    with open(result_path['tp_info'], 'r') as f:
        tp_info = json.load(f)

    # parts = list(np.unique(list(self.event_tracer['Part'].dropna())))

    process_capacity = pd.read_excel(input_data['path_process_area'])
    process_capacity = {process_capacity.iloc[i]['name']: process_capacity.iloc[i]['area'] for i in
                        range(len(process_capacity))}

    stock_capacity = pd.read_excel(input_data['path_stock_area'])
    stock_capacity = {stock_capacity.iloc[i]['name']: stock_capacity.iloc[i]['area'] for i in
                      range(len(stock_capacity))}

    tp_index(event_tracer, tp_info, input_data['default_result'])
    calculate_stock_occupied_area(input_data['default_result'], event_tracer, input_data, stock_capacity)
    calculate_painting_occupied_area(input_data['default_result'], event_tracer, input_data, process_capacity)
    calculate_shelter_occupied_area(input_data['default_result'], event_tracer, input_data, process_capacity)

