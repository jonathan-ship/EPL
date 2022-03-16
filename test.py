def _find_stock(self, next_process):
    stock_list = list(self.stocks.keys())
    if self.dock != 3:
        stock_list.remove('E6')
        stock_list.remove('E7')
        stock_list.remove('E8')
        stock_list.remove('E9')
    else:
        stock_list = ['E6', 'E7', 'E8', 'E9', 'Stock']
    stock_list.remove('Stock')
    next_stock = None
    shortest_path = 1e8
    if (self.dock == 3) and (self.env.now >= 470):
        print(0)
    for idx in range(len(stock_list)):
        temp_stock = self.stocks[stock_list[idx]]
        if temp_stock.capacity_unit == 'm2':
            if_to_be_capacity = temp_stock.capacity - temp_stock.capacity_dict['m2'][
                'preemptive'] - self.block.area
        elif temp_stock.capacity_unit == 'EA':
            if_to_be_capacity = temp_stock.capacity - temp_stock.capacity_dict['EA']['preemptive'] - 1
        else:
            if_to_be_capacity = temp_stock.capacity - temp_stock.capacity_dict['ton'][
                'preemptive'] - self.block.weight
        if if_to_be_capacity > 0:
            if next_stock is None:  # No stock to compare
                next_stock = stock_list[idx]
                shortest_path = self.network_distance[next_process][self.inout[stock_list[idx]][0]]
            else:
                compared_path = self.network_distance[next_process][self.inout[stock_list[idx]][0]]
                if shortest_path > compared_path:  # replaced
                    next_stock = stock_list[idx]
                    shortest_path = compared_path
                elif shortest_path == compared_path:
                    shortest_stock_len = len(self.stocks[next_stock].stock_yard.items)
                    compared_stock_len = len(self.stocks[stock_list[idx]].stock_yard.items)
                    next_stock = stock_list[idx] if compared_stock_len < shortest_stock_len else next_stock

    next_stock = next_stock if next_stock is not None else 'Stock'
    self.stocks[next_stock].capacity_dict = record_capacity(self.stocks[next_stock].capacity_dict,
                                                            self.stocks[next_stock].capacity_unit,
                                                            self.blocks[self.name], type='preemptive',
                                                            inout='in')

    return next_stock


def _convert_process(self, step):
    present_process = self.data[(step, 'process')]
    if (present_process in ['선행도장부', '선행의장부', '기장부', '의장1부', '의장2부', '의장3부']) and (self.dock == 3):
        present_process = '3도크 ' + present_process
    # 1:1 대응
    if present_process not in self.convert_to_process.keys():
        return present_process

    if present_process not in ['선행도장부', '선행의장부', '기장부', '의장1부', '의장3부', '의장2부', '3도크 선행도장부',
                               '3도크 선행의장부', '3도크 기장부', '3도크 의장1부', '3도크 의장2부', '3도크 의장3부']:
        if present_process in ['건조1부', '건조2부', '건조3부']:
            dock_num = self.dock
            dock = "{0}도크".format(dock_num)
            return dock
        elif present_process == "Sink":
            return "Sink"
        else:
            converted_process = self.convert_to_process[present_process]
            return converted_process

    # 그냥 process인 경우 + 경우의 수가 여러 개인 경우
    else:
        if step == 0:
            previous_process = self.source_location
        else:
            previous_process = self.data[(step - 1, 'process')]

        process_convert_by_dict = self.convert_to_process[present_process]
        if present_process in ['3도크 선행의장부', '3도크 기장부', '3도크 의장1부', '3도크 의장2부', '3도크 의장3부']:
            print(0)
        distance = []
        pre_choice = process_convert_by_dict[:]
        if previous_process is None:
            converted_pro_process = self.convert_to_process[self.data[(step + 1, 'process')]]
            if converted_pro_process == "Dock":
                converted_pro_process = '{0}도크'.format(self.dock)
            if '판넬' in converted_pro_process:  # 판넬 물량이면
                previous_process = '판넬선각공장' if self.yard == 1 else "2야드 판넬공장"
            elif self.name[6] == "H":  ## 선실 물량이면
                previous_process = "선실공장"
            else:
                previous_process = "대조립1공장" if self.yard == 1 else "2야드 대조립공장"
        else:
            previous_process = self.convert_to_process[
                previous_process] if previous_process in self.convert_to_process.keys() else previous_process
            if previous_process == 'Dock':
                previous_process = '{0}도크'.format(self.dock)

        compared_process = self.inout[previous_process][1]
        for process in process_convert_by_dict:
            process_temp = self.inout[process][0]
            if (process_temp in self.network_distance.keys()) and (
                    compared_process in self.network_distance.keys()) and \
                    (self.network_distance[process_temp][compared_process] is not None) and \
                    (self.network_distance[process_temp][compared_process] < 100000):
                temp_process = self.processes[process]
                if temp_process.capacity_unit == 'm2':
                    if_to_be_capacity = temp_process.capacity - temp_process.capacity_dict['m2'][
                        'preemptive'] - self.block.area
                elif temp_process.capacity_unit == 'EA':
                    if_to_be_capacity = temp_process.capacity - temp_process.capacity_dict['EA']['preemptive'] - 1
                else:
                    if_to_be_capacity = temp_process.capacity - temp_process.capacity_dict['ton'][
                        'preemptive'] - self.block.weight

                if if_to_be_capacity > 0:
                    distance.append(self.network_distance[process_temp][compared_process])
                else:
                    pre_choice.remove(process)
            else:
                pre_choice.remove(process)
        if len(distance) > 0:
            process_idx = distance.index(min(distance))
            process = pre_choice[process_idx]
        else:
            if present_process == '선행도장부' or present_process == '3도크 선행도장부':
                process = 'Painting'
            else:
                process = 'Shelter'

        self.data[(step, 'process')] = process
        self.processes[process].capacity_dict = record_capacity(self.processes[process].capacity_dict,
                                                                self.processes[process].capacity_unit,
                                                                self.blocks[self.name], type='preemptive',
                                                                inout='in')
        return process