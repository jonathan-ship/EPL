import pandas as pd

data_all = pd.read_excel('./data/Layout_data_series40.xlsx')
dock_mapping = pd.read_excel('./data/호선도크.xlsx')
dock_mapping = dict(dock_mapping.transpose())

block_group = data_all.groupby(data_all['series_block_code'])
block_list = list(data_all.drop_duplicates(['series_block_code'])['series_block_code'])

block_yard1 = 0.0
block_yard2 = 0.0

for block_code in block_list:
    block = block_group.get_group(block_code)
    block = block.sort_values(by=['start_date'], ascending=True)
    block = block.reset_index(drop=True)
    block_area = block['area'][0]

    series_num = int(block_code[3:5])
    yard = dock_mapping[series_num - 1]['야드']

    if yard == 1:
        block_yard1 += block_area
    else:
        block_yard2 += block_area

print("#"*10, "Area of blocks", "#"*10)
print("total block area in yard 1 = ", block_yard1)
print("total block area in yard 2 = ", block_yard2)
print("total block area in entire yard = ", block_yard1 + block_yard2)
print("total block area in yard 1 / total block in yard 2 = ", block_yard1/block_yard2)


event_data = pd.read_csv('./result/Series40_and_adding_stock/result_series40_adding_stock.csv', low_memory = False)

# 적치장 역물류
event_data_stock_in = event_data[event_data['Event'] == 'Stock_in']

block_group_stock = event_data_stock_in.groupby(event_data_stock_in['Part'])
block_list_stock = list(event_data_stock_in.drop_duplicates(['Part'])['Part'])

block_go_to_stock_in_yard2_but_be_in_yard1 = 0.0  # 1야드 물량이지만 2야드 적치장으로 간 경우
block_go_to_stock_in_yard1_but_be_in_yard2 = 0.0  # 2야드 물량이지만 1야드 적치장으로 간 경우
block_go_2_but_in_1_count_only_1 = 0.0
block_go_1_but_in_2_count_only_1 = 0.0

stock_in_yard2 = ['Y1', 'Y2', 'Y3', 'Y4', 'Y5', 'Y6', 'Y7']
part_name_stock = []
yard_num_stock = []
times_stock = []

for part in block_list_stock:
    series_num = int(part[3:5])
    yard = dock_mapping[series_num - 1]['야드']

    block_info = block_group.get_group(part)
    block_info = block_info.reset_index(drop=True)
    part_area = block_info['area'][0]

    part_group = block_group_stock.get_group(part)
    time = 0
    for i in range(len(part_group)):
        temp = part_group.iloc[i]
        where_stock = 2 if temp['Process'] in stock_in_yard2 else 1
        if where_stock != yard:
            if yard == 1:  # 1야드 물량인데 2야드로 간 경우
                block_go_to_stock_in_yard2_but_be_in_yard1 += part_area
                if time == 0:
                    block_go_2_but_in_1_count_only_1 += part_area
                time += 1
            else:  # 2야드 물량인데 1야드로 간 경우
                block_go_to_stock_in_yard1_but_be_in_yard2 += part_area
                if time == 0:
                    block_go_1_but_in_2_count_only_1 += part_area
                time += 1
    part_name_stock.append(part)
    yard_num_stock.append(yard)
    times_stock.append(time)

print("#"*10, "Inverse Logistics of Stock Yard", "#"*10)
print("Count multiple")
print("Part is in yard 1 but it went to stock in yard 2 = ", block_go_to_stock_in_yard2_but_be_in_yard1)
print("Part is in yard 2 but it went to stock in yard 1 = ", block_go_to_stock_in_yard1_but_be_in_yard2)
print("Count only once")
print("Part is in yard 1 but it went to stock in yard 2 = ", block_go_2_but_in_1_count_only_1)
print("Part is in yard 2 but it went to stock in yard 1 = ", block_go_1_but_in_2_count_only_1)

inverse_logistics_stock = pd.DataFrame(columns=['Part', 'Yard', 'Times'])
inverse_logistics_stock['Part'] = part_name_stock
inverse_logistics_stock['Yard'] = yard_num_stock
inverse_logistics_stock['Times'] = times_stock

inverse_logistics_stock.to_excel('./Inverse_logistics_stocks.xlsx')

# 도장 역물류
event_data_paint = event_data[event_data['Reason'] == 'Paint']

block_group_paint = event_data_paint.groupby(event_data_paint['Part'])
block_list_paint = list(event_data_paint.drop_duplicates(['Part'])['Part'])

block_go_paint_in_2_but_in_1 = 0.0  # 1야드 물량이지만 2야드 적치장으로 간 경우
block_go_paint_in_1_but_in_2 = 0.0  # 2야드 물량이지만 1야드 적치장으로 간 경우
block_go_paint_in_2_but_in_1_count_once = 0.0
block_go_paint_in_1_but_in_2_count_once = 0.0

paint_in_yard2 = ['2야드 도장 1공장', '2야드 도장 2공장', '2야드 도장 3공장', '2야드 도장 5공장', '2야드 도장 6공장']
part_name_paint = []
yard_num_paint = []
times_paint = []

for part in block_list_paint:
    series_num = int(part[3:5])
    yard = dock_mapping[series_num - 1]['야드']

    block_info = block_group.get_group(part)
    block_info = block_info.reset_index(drop=True)
    part_area = block_info['area'][0]

    part_group = block_group_paint.get_group(part)
    time = 0
    for i in range(len(part_group)):
        temp = part_group.iloc[i]
        where_paint = 2 if temp['Process'] in paint_in_yard2 else 1
        if where_paint != yard:
            if yard == 1:  # 1야드 물량인데 2야드로 간 경우
                block_go_paint_in_2_but_in_1 += part_area
                if time == 0:
                    block_go_paint_in_2_but_in_1_count_once += part_area
                time += 1
            else:  # 2야드 물량인데 1야드로 간 경우
                block_go_paint_in_1_but_in_2 += part_area
                if time == 0:
                    block_go_paint_in_1_but_in_2_count_once += part_area
                time += 1
    part_name_paint.append(part)
    yard_num_paint.append(yard)
    times_paint.append(time)

print("#"*10, "Inverse Logistics of Paint", "#"*10)
print("Count multiple")
print("Part is in yard 1 but it went to paint in yard 2 = ", block_go_paint_in_2_but_in_1)
print("Part is in yard 2 but it went to paint in yard 1 = ", block_go_paint_in_1_but_in_2)
print("Count only once")
print("Part is in yard 1 but it went to paint in yard 2 = ", block_go_paint_in_2_but_in_1_count_once)
print("Part is in yard 2 but it went to paint in yard 1 = ", block_go_paint_in_1_but_in_2_count_once)

inverse_logistics_paint = pd.DataFrame(columns=['Part', 'Yard', 'Times'])
inverse_logistics_paint['Part'] = part_name_paint
inverse_logistics_paint['Yard'] = yard_num_paint
inverse_logistics_paint['Times'] = times_paint

inverse_logistics_paint.to_excel('./Inverse_logistics_paints.xlsx')

# 쉘터 역물류
event_data_shelter = event_data[event_data['Reason'] == 'Shelter']

block_group_shelter = event_data_shelter.groupby(event_data_shelter['Part'])
block_list_shelter = list(event_data_shelter.drop_duplicates(['Part'])['Part'])

block_go_shelter_in_2_but_in_1 = 0.0  # 1야드 물량이지만 2야드 적치장으로 간 경우
block_go_shelter_in_1_but_in_2 = 0.0  # 2야드 물량이지만 1야드 적치장으로 간 경우
block_go_shelter_in_2_but_in_1_count_once = 0.0
block_go_shelter_in_1_but_in_2_count_once = 0.0

shelter_in_yard2 = ['대조립5부쉘터', '중조립SHOP쉘터', '판넬조립5부쉘터', '8도크PE', '9도크PE']
part_name_shelter = []
yard_num_shelter = []
times_shelter = []

for part in block_list_shelter:
    series_num = int(part[3:5])
    yard = dock_mapping[series_num - 1]['야드']

    block_info = block_group.get_group(part)
    block_info = block_info.reset_index(drop=True)
    part_area = block_info['area'][0]

    part_group = block_group_shelter.get_group(part)
    time = 0
    for i in range(len(part_group)):
        temp = part_group.iloc[i]
        where_shelter = 2 if temp['Process'] in shelter_in_yard2 else 1
        if where_shelter != yard:
            if yard == 1:  # 1야드 물량인데 2야드로 간 경우
                block_go_shelter_in_2_but_in_1 += part_area
                if time == 0:
                    block_go_shelter_in_2_but_in_1_count_once += part_area
                time += 1
            else:  # 2야드 물량인데 1야드로 간 경우
                block_go_shelter_in_1_but_in_2 += part_area
                if time == 0:
                    block_go_shelter_in_1_but_in_2_count_once += part_area
                time += 1
    part_name_shelter.append(part)
    yard_num_shelter.append(yard)
    times_shelter.append(time)

print("#"*10, "Inverse Logistics of Shelter", "#"*10)
print("Count multiple")
print("Part is in yard 1 but it went to shelter in yard 2 = ", block_go_shelter_in_2_but_in_1)
print("Part is in yard 2 but it went to shelter in yard 1 = ", block_go_shelter_in_1_but_in_2)
print("Count only once")
print("Part is in yard 1 but it went to shelter in yard 2 = ", block_go_shelter_in_2_but_in_1_count_once)
print("Part is in yard 2 but it went to shelter in yard 1 = ", block_go_shelter_in_1_but_in_2_count_once)

inverse_logistics_shelter = pd.DataFrame(columns=['Part', 'Yard', 'Times'])
inverse_logistics_shelter['Part'] = part_name_paint
inverse_logistics_shelter['Yard'] = yard_num_paint
inverse_logistics_shelter['Times'] = times_paint

inverse_logistics_shelter.to_excel('./Inverse_logistics_shelter.xlsx')