import scipy.io
import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
from tools.utility import get_central_vertices, smooth_ployline, get_intersection_point
import pandas as pd
from datetime import datetime
import xlsxwriter

illustration_needed = False
print_needed = False
save_data_needed = True
# load data
mat = scipy.io.loadmat('./data/NDS_data_fixed.mat')
# full interaction information
inter_info = mat['interaction_info']
case_number = len(inter_info)
'''
inter_info:
0-1: [position x] [position y]
2-3: [velocity x] [velocity y]
4: [heading]
5: [velocity overall = sqrt(vx^2+xy^2)]
6: [curvature] (only for left-turn vehicles)
dt = 0.12s 
'''
# the number of go-straight vehicles that interact with the left-turn vehicle
inter_num = mat['interact_agent_num']


def find_inter_od(case_id):
    """
    find the starting and end frame of each FC agent that interacts with LT agent
    :param case_id:
    :return:
    """
    case_info = inter_info[case_id]
    lt_info = case_info[0]

    # find co-present gs agents (not necessarily interacting)
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    # find interacting gs agent
    init_id = 0
    inter_o = np.zeros(np.size(gs_info_multi, 0))
    inter_d = np.zeros(np.size(gs_info_multi, 0))
    for i in range(np.size(gs_info_multi, 0)):
        gs_agent_temp = gs_info_multi[i]
        solid_frame = np.nonzero(gs_agent_temp[:, 0])[0]
        solid_range = range(solid_frame[0], solid_frame[-1])
        inter_frame = solid_frame[0] + np.array(
            np.where(gs_agent_temp[solid_range, 1] - lt_info[solid_range, 1] < 0)[0])

        # find start and end frame with each gs agent
        if inter_frame.size > 1:
            if i == init_id:
                inter_o[i] = inter_frame[0]
            else:
                inter_o[i] = max(inter_frame[0], inter_d[i - 1])
            inter_d[i] = max(inter_frame[-1], inter_d[i - 1])
        else:
            init_id += 1
    return inter_o, inter_d


def analyze_ipv_in_nds(case_id, fig=False):
    file_name = 'data/ipv_estimation/' + str(case_id) + '.xlsx'
    file = pd.ExcelFile(file_name)
    num_sheet = len(file.sheet_names)
    # print(num_sheet)
    start_x = 0
    crossing_id = -1

    for i in range(num_sheet):
        "get ipv data from excel"
        df_ipv_data = pd.read_excel(file_name, sheet_name=i)
        ipv_data_temp = df_ipv_data.values
        ipv_value_lt, ipv_value_gs = ipv_data_temp[:, 0], ipv_data_temp[:, 7]
        ipv_error_lt, ipv_error_gs = ipv_data_temp[:, 1], ipv_data_temp[:, 8]

        # find cross event
        x_lt = ipv_data_temp[:, 2]
        x_gs = ipv_data_temp[:, 9]
        delta_x = x_lt - x_gs  # x position of LT is larger than that of interacting FC
        if len(delta_x) > 0:
            if np.max(delta_x) > 0 and crossing_id == -1:
                crossing_id = i

        "draw ipv value and error bar"

        if fig:
            x = start_x + np.arange(len(ipv_value_lt))
            start_x = start_x + len(ipv_value_lt)
            print(start_x)

            if len(x) > 6:
                # left turn
                smoothed_ipv_value_lt, _ = smooth_ployline(np.array([x, ipv_value_lt]).T)
                smoothed_ipv_error_lt, _ = smooth_ployline(np.array([x, ipv_error_lt]).T)
                plt.plot(smoothed_ipv_value_lt[:, 0], smoothed_ipv_value_lt[:, 1],
                         color='blue')
                plt.fill_between(smoothed_ipv_value_lt[:, 0], smoothed_ipv_value_lt[:, 1] - smoothed_ipv_error_lt[:, 1],
                                 smoothed_ipv_value_lt[:, 1] + smoothed_ipv_error_lt[:, 1],
                                 alpha=0.4,
                                 color='blue')

                # go straight
                smoothed_ipv_value_gs, _ = smooth_ployline(np.array([x, ipv_value_gs]).T)
                smoothed_ipv_error_gs, _ = smooth_ployline(np.array([x, ipv_error_gs]).T)
                plt.plot(smoothed_ipv_value_gs[:, 0], smoothed_ipv_value_gs[:, 1],
                         color='red')
                plt.fill_between(smoothed_ipv_value_gs[:, 0], smoothed_ipv_value_gs[:, 1] - smoothed_ipv_error_gs[:, 1],
                                 smoothed_ipv_value_gs[:, 1] + smoothed_ipv_error_gs[:, 1],
                                 alpha=0.4,
                                 color='red')

            else:  # too short to be fitted
                # left turn
                plt.plot(x, ipv_value_lt,
                         color='red')
                plt.fill_between(x, ipv_value_lt - ipv_error_lt,
                                 ipv_value_lt + ipv_error_lt,
                                 alpha=0.4,
                                 color='red',
                                 label='estimated lt IPV')

                # go straight
                plt.plot(x, ipv_value_gs,
                         color='blue')
                plt.fill_between(x, ipv_value_gs - ipv_error_gs,
                                 ipv_value_gs + ipv_error_gs,
                                 alpha=0.4,
                                 color='blue',
                                 label='estimated gs IPV')
            # plt.pause(1)
    plt.show()

    # save ipv during the crossing event
    case_data_crossing = []
    case_data_non_crossing = []
    if not crossing_id == -1:
        df_data = pd.read_excel(file_name, sheet_name=crossing_id)
        case_data_crossing = df_data.values[:, :]

    for sheet_id in range(num_sheet):
        if not sheet_id == crossing_id:
            df_data = pd.read_excel(file_name, sheet_name=sheet_id)
            case_data_non_crossing.append(df_data.values[:, :])

    return crossing_id, case_data_crossing, case_data_non_crossing


def cal_pet(trj_a, trj_b, type_cal):
    """
    calculate the PET of two given trajectory
    :param trj_a:
    :param trj_b:
    :param type_cal: PET or APET
    :return:
    """

    "find the conflict point"
    conflict_point_str = get_intersection_point(trj_a, trj_b)
    conflict_point = np.array(conflict_point_str)
    if conflict_point_str.is_empty:  # there is no intersection between given polylines
        min_dis = 99
        min_dis2cv_index_a = None
        min_dis2cv_index_b = 0
        for i in range(np.size(trj_b, 0)):
            point_b = trj_b[i, :]
            dis2cv_lt_temp = np.linalg.norm(trj_a - point_b, axis=1)
            min_dis2cv_temp = np.amin(dis2cv_lt_temp)
            min_dis2cv_index_temp = np.where(min_dis2cv_temp == dis2cv_lt_temp)
            if min_dis2cv_temp < min_dis:
                min_dis = min_dis2cv_temp
                min_dis2cv_index_a = min_dis2cv_index_temp[0]
                min_dis2cv_index_b = i
        conflict_point = (trj_a[min_dis2cv_index_a[0], :] + trj_b[min_dis2cv_index_b, :]) / 2

    "find the point that most closed to cp in each trajectory"
    smoothed_trj_a, smoothed_progress_a = smooth_ployline(trj_a, point_num=100)
    cp2trj_a = np.linalg.norm(smoothed_trj_a - conflict_point, axis=1)
    min_dcp2trj_a = np.amin(cp2trj_a)
    cp_index_a = np.where(min_dcp2trj_a == cp2trj_a)

    smoothed_trj_b, smoothed_progress_b = smooth_ployline(trj_b, point_num=100)
    cp2trj_b = np.linalg.norm(smoothed_trj_b - conflict_point, axis=1)
    min_dcp2trj_b = np.amin(cp2trj_b)
    cp_index_b = np.where(min_dcp2trj_b == cp2trj_b)

    "calculate time to cp"
    seg_len_a = np.linalg.norm(trj_a[1:, :] - trj_a[:-1, :], axis=1)
    seg_len_b = np.linalg.norm(trj_b[1:, :] - trj_b[:-1, :], axis=1)
    vel_a = seg_len_a / 0.12
    vel_b = seg_len_b / 0.12
    seg_len_a = np.concatenate([np.array([0]), seg_len_a])
    seg_len_b = np.concatenate([np.array([0]), seg_len_b])
    longi_progress_a = np.cumsum(seg_len_a)
    longi_progress_b = np.cumsum(seg_len_b)

    dis2conf_a = -(longi_progress_a - smoothed_progress_a[cp_index_a])
    dis2conf_b = -(longi_progress_b - smoothed_progress_b[cp_index_b])

    ttcp_a = dis2conf_a[:-1] / vel_a
    ttcp_b = dis2conf_b[:-1] / vel_b

    solid_len = min(np.size(ttcp_a[ttcp_a > 0], 0), np.size(ttcp_b[ttcp_b > 0], 0))

    "PET and APET"
    apet = np.abs(ttcp_a[:solid_len] - ttcp_b[:solid_len])

    pet = max(ttcp_a[solid_len - 1], ttcp_b[solid_len - 1]) - min(ttcp_a[solid_len - 1], ttcp_b[solid_len - 1])

    if type_cal == 'pet':

        return pet, conflict_point

    elif type_cal == 'apet':

        return apet, ttcp_a, ttcp_b
