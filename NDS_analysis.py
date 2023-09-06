import scipy.io
import math
import numpy as np
from matplotlib import pyplot as plt
from agent import Agent
from tools.utility import get_central_vertices, smooth_ployline, get_intersection_point, draw_rectangle
import pandas as pd
import time
from datetime import datetime
import xlsxwriter

illustration_needed = False
print_needed = False
save_data_needed = False
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
    寻找一个（剑河仙霞数据集）驾驶数据片段中的有效交互片段
    :param case_id: 数据片段序号
    :return: 交互的起终点
    """
    case_info = inter_info[case_id]
    lt_info = case_info[0]

    # find co-present gs agents (not necessarily interacting)
    # 读取所有与左转车同时存在于场景中的直行车（这些直行车不一定与左转车存在交互）
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    # find interacting gs agent
    # 遍历所有同时存在过的直行车，判断交互存在性及何时存在交互
    init_id = 0
    inter_o = np.zeros(np.size(gs_info_multi, 0))
    inter_d = np.zeros(np.size(gs_info_multi, 0))
    for i in range(np.size(gs_info_multi, 0)):
        gs_agent_temp = gs_info_multi[i]
        solid_frame = np.nonzero(gs_agent_temp[:, 0])[0]
        solid_range = range(solid_frame[0], solid_frame[-1])  # 同时存在的时间片段
        inter_frame = solid_frame[0] + np.array(
            np.where(gs_agent_temp[solid_range, 1] - lt_info[solid_range, 1] < 0)[0])  # 交互片段

        # find start and end frame with each gs agent
        # 记录交互起终点
        if inter_frame.size > 1:
            if i == init_id:
                inter_o[i] = inter_frame[0]
            else:
                inter_o[i] = max(inter_frame[0], inter_d[i - 1])
            inter_d[i] = max(inter_frame[-1], inter_d[i - 1])
        else:
            init_id += 1
    return inter_o, inter_d


def get_ipv_in_nds(case_id, fig=False):
    """
    从excel读取给定驾驶数据片段中的交互事件片段
    Parameters
    ----------
    case_id: 驾驶数据片段序号
    fig: 是否绘制该片段中个体的ipv(主要用于调试)

    """
    file_name = 'data/ipv_estimation/' + str(case_id) + '.xlsx'
    file = pd.ExcelFile(file_name)
    num_sheet = len(file.sheet_names)
    # print(num_sheet)
    start_x = 0
    crossing_id = -1

    # 遍历该片段中的所有直行车辆
    for i in range(num_sheet):
        "get ipv data from excel"
        "读取IPV"
        df_ipv_data = pd.read_excel(file_name, sheet_name=i)
        ipv_data_temp = df_ipv_data.values
        ipv_value_lt, ipv_value_gs = ipv_data_temp[:, 0], ipv_data_temp[:, 7]
        ipv_error_lt, ipv_error_gs = ipv_data_temp[:, 1], ipv_data_temp[:, 8]

        # find cross event
        # 寻找让行于左转车的直行车
        x_lt = ipv_data_temp[:, 2]
        x_gs = ipv_data_temp[:, 9]
        delta_x = x_lt - x_gs  # x position of LT is larger than that of interacting FC
        if len(delta_x) > 0:  # 如果直行车让行于左转车，则将存在某一帧满足：左转车的x坐标大于直行车的x坐标
            if np.max(delta_x) > 0 and crossing_id == -1:
                crossing_id = i  # 记录让行直行车的id

        "draw ipv value and error bar"
        "绘制交互双方的ipv"
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
    # 区分直行车是否让行，分别保存交互事件数据
    case_data_crossing = []
    case_data_non_crossing = []

    # 保存让行直行车的交互数据
    if not crossing_id == -1:
        df_data = pd.read_excel(file_name, sheet_name=crossing_id)
        case_data_crossing = df_data.values[:, :]

    # 保存抢行直行车的交互数据
    for sheet_id in range(num_sheet):
        if not sheet_id == crossing_id:
            df_data = pd.read_excel(file_name, sheet_name=sheet_id)
            case_data_non_crossing.append(df_data.values[:, :])

    return crossing_id, case_data_crossing, case_data_non_crossing


def cal_pet(trj_a, trj_b, type_cal):
    """
    Calculate the PET of two given trajectory
    计算两条给定轨迹的PET或APET
    Parameters
    ----------
    trj_a
    trj_b
    type_cal: 'pet'或'apet'


    """

    "find the conflict point"
    "寻找两条轨迹的冲突点"
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
    if not np.size(conflict_point) == 2:
        conflict_point = conflict_point[0, :]

    "find the point that most closed to cp in each trajectory"
    "寻找两条轨迹各自距离冲突点最近的轨迹点"
    smoothed_trj_a, smoothed_progress_a = smooth_ployline(trj_a, point_num=100)
    cp2trj_a = np.linalg.norm(smoothed_trj_a - conflict_point, axis=1)
    min_dcp2trj_a = np.amin(cp2trj_a)
    cp_index_a = np.where(min_dcp2trj_a == cp2trj_a)

    smoothed_trj_b, smoothed_progress_b = smooth_ployline(trj_b, point_num=100)
    cp2trj_b = np.linalg.norm(smoothed_trj_b - conflict_point, axis=1)
    min_dcp2trj_b = np.amin(cp2trj_b)
    cp_index_b = np.where(min_dcp2trj_b == cp2trj_b)

    "calculate time to cp"
    "计算达到冲突点的时间差"
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

    ttcp_a = dis2conf_a[:-1] / vel_a  # ttcp:time to conflict point
    ttcp_b = dis2conf_b[:-1] / vel_b

    solid_len = min(np.size(ttcp_a[ttcp_a > 0], 0), np.size(ttcp_b[ttcp_b > 0], 0))
    if solid_len == 0:
        solid_len = 1
    "PET and APET"
    apet = np.abs(ttcp_a[:solid_len] - ttcp_b[:solid_len])

    pet = max(ttcp_a[solid_len - 1], ttcp_b[solid_len - 1]) - min(ttcp_a[solid_len - 1], ttcp_b[solid_len - 1])

    if type_cal == 'pet':

        return pet, conflict_point

    elif type_cal == 'apet':

        return apet, ttcp_a, ttcp_b


def estimate_ipv_in_nds(case_id):
    """
    估计自然驾驶数据片段中个体的交互倾向（IPV）
    Parameters
    ----------
    case_id：（剑河仙霞）驾驶数据数据片段序号

    """
    output_path = 'outputs/ipv_estimation/'

    # 计算当前驾驶数据片段中的交互事件起终点
    inter_o, inter_d = find_inter_od(case_id)
    case_info = inter_info[case_id]
    lt_info = case_info[0]

    # find co-present gs agents (not necessarily interacting)
    # 读取所有与左转车同时存在于场景中的直行车（这些直行车不一定与左转车存在交互）
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    # initialize IPV
    start_time = 0
    ipv_collection = np.zeros_like(lt_info[:, 0:2])
    ipv_error_collection = np.ones_like(lt_info[:, 0:2])

    # set figure
    if illustration_needed:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=[16, 8])

    inter_id = 0
    inter_id_save = inter_id
    file_name = output_path + str(case_id) + '.xlsx'

    for t in range(np.size(lt_info, 0)):
        print(t)

        "find current interacting agent"
        "寻找t时刻的交互直行车"
        flag = 0
        for i in range(np.size(gs_info_multi, 0)):
            if inter_o[i] <= t < inter_d[i]:  # switch to next interacting agent
                # update interaction info
                flag = 1
                inter_id = i
                if print_needed:
                    print('inter_id', inter_id)
                start_time = max(int(inter_o[inter_id]), t - 10)

        # save data of last one
        # 如果与某一个直行车的交互结束，则保存一次IPV估计结果
        if save_data_needed:
            if inter_id_save < inter_id or t == inter_d[-1]:
                # if inter_d[inter_id_save] - inter_o[inter_id_save] > 3:
                '''
                inter_id_save < inter_id：  interacting agent changed
                t == inter_d[-1]:  end frame of the last agent
                inter_d[inter_id_save]-inter_o[inter_id_save] > 3：  interacting period is long enough
                '''
                # save data into an excel with the format of:
                # 0-ipv_lt | ipv_lt_error | lt_px | lt_py  | lt_vx  | lt_vy  | lt_heading  |...
                # 7-ipv_gs | ipv_gs_error | gs_px | gs_py  | gs_vx  | gs_vy  | gs_heading  |

                df_ipv_lt = pd.DataFrame(ipv_collection[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 0],
                                         columns=["ipv_lt"])
                df_ipv_lt_error = pd.DataFrame(
                    ipv_error_collection[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 0],
                    columns=["ipv_lt_error"])
                df_motion_lt = pd.DataFrame(lt_info[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 0:5],
                                            columns=["lt_px", "lt_py", "lt_vx", "lt_vy", "lt_heading"])

                df_ipv_gs = pd.DataFrame(ipv_collection[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 1],
                                         columns=["ipv_gs"])
                df_ipv_gs_error = pd.DataFrame(
                    ipv_error_collection[int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 1],
                    columns=["ipv_gs_error"])
                df_motion_gs = pd.DataFrame(gs_info_multi[inter_id_save]
                                            [int(inter_o[inter_id_save]): int(inter_d[inter_id_save]), 0:5],
                                            columns=["gs_px", "gs_py", "gs_vx", "gs_vy", "gs_heading"])

                if inter_id_save == 0:
                    with pd.ExcelWriter(file_name) as writer:
                        df_ipv_lt.to_excel(writer, startcol=0, index=False, sheet_name=str(inter_id_save))
                        df_ipv_lt_error.to_excel(writer, startcol=1, index=False, sheet_name=str(inter_id_save))
                        df_motion_lt.to_excel(writer, startcol=2, index=False, sheet_name=str(inter_id_save))

                        df_ipv_gs.to_excel(writer, startcol=7, index=False, sheet_name=str(inter_id_save))
                        df_ipv_gs_error.to_excel(writer, startcol=8, index=False, sheet_name=str(inter_id_save))
                        df_motion_gs.to_excel(writer, startcol=9, index=False, sheet_name=str(inter_id_save))
                else:
                    with pd.ExcelWriter(file_name, mode="a", if_sheet_exists="overlay") as writer:
                        df_ipv_lt.to_excel(writer, startcol=0, index=False, sheet_name=str(inter_id_save))
                        df_ipv_lt_error.to_excel(writer, startcol=1, index=False, sheet_name=str(inter_id_save))
                        df_motion_lt.to_excel(writer, startcol=2, index=False, sheet_name=str(inter_id_save))

                        df_ipv_gs.to_excel(writer, startcol=7, index=False, sheet_name=str(inter_id_save))
                        df_ipv_gs_error.to_excel(writer, startcol=8, index=False, sheet_name=str(inter_id_save))
                        df_motion_gs.to_excel(writer, startcol=9, index=False, sheet_name=str(inter_id_save))

                inter_id_save = inter_id

        "IPV estimation process"
        "对t时刻的IPV进行估计"
        if flag and (t - start_time > 3):

            "====simulation-based method===="
            # generate two agents
            # 基于t时刻的状态，实例化两个交互对象
            init_position_lt = lt_info[start_time, 0:2] - [13, 7.8]
            init_velocity_lt = lt_info[start_time, 2:4]
            init_heading_lt = lt_info[start_time, 4]
            agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt_nds')
            lt_track = lt_info[start_time:t + 1, 0:2]
            lt_track = lt_track - np.repeat([[13, 7.8]], np.size(lt_track, 0), axis=0)

            init_position_gs = gs_info_multi[inter_id][start_time, 0:2] - [13, 7.8]
            init_velocity_gs = gs_info_multi[inter_id][start_time, 2:4]
            init_heading_gs = gs_info_multi[inter_id][start_time, 4]
            agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs_nds')
            gs_track = gs_info_multi[inter_id][start_time:t + 1, 0:2]
            gs_track = gs_track - np.repeat([[13, 7.8]], np.size(gs_track, 0), axis=0)

            # estimate ipv
            # IPV估计
            agent_lt.estimate_self_ipv(lt_track, gs_track)
            ipv_collection[t, 0] = agent_lt.ipv
            ipv_error_collection[t, 0] = agent_lt.ipv_error

            agent_gs.estimate_self_ipv(gs_track, lt_track)
            ipv_collection[t, 1] = agent_gs.ipv
            ipv_error_collection[t, 1] = agent_gs.ipv_error

            if print_needed:
                print('left turn', agent_lt.ipv, agent_lt.ipv_error)
                print('go straight', agent_gs.ipv, agent_gs.ipv_error)
            "====end of simulation-based method===="

            "====cost-based method===="
            # load observed trajectories
            # lt_track_observed = lt_info[start_time:t + 1, 0:2]
            # gs_track_observed = gs_info_multi[inter_id][start_time:t + 1, 0:2]
            #
            # # cost results in observation
            # interior_cost_lt = cal_interior_cost([], lt_track_observed, 'lt_nds')
            # interior_cost_gs = cal_interior_cost([], gs_track_observed, 'gs_nds')
            # group_cost_lt = cal_group_cost([lt_track_observed, gs_track_observed])
            # group_cost_gs = cal_group_cost([gs_track_observed, lt_track_observed])
            #
            # ipv_collection[t, 0] =
            # ipv_collection[t, 1] =
            "====end of cost-based method===="

            "illustration"
            "可视化IPV估计结果和轨迹状态"
            if illustration_needed:
                ax1.cla()
                ax1.set(ylim=[-2, 2])

                x_range = range(max(0, t - 10), t)
                smoothed_ipv_lt, _ = smooth_ployline(np.array([x_range, ipv_collection[x_range, 0]]).T)
                smoothed_ipv_error_lt, _ = smooth_ployline(np.array([x_range, ipv_error_collection[x_range, 0]]).T)
                smoothed_x = smoothed_ipv_lt[:, 0]
                # plot ipv
                ax1.plot(smoothed_x, smoothed_ipv_lt[:, 1], 'blue')
                # plot error bar
                ax1.fill_between(smoothed_x, smoothed_ipv_lt[:, 1] - smoothed_ipv_error_lt[:, 1],
                                 smoothed_ipv_lt[:, 1] + smoothed_ipv_error_lt[:, 1],
                                 alpha=0.4,
                                 color='blue',
                                 label='estimated lt IPV')

                smoothed_ipv_gs, _ = smooth_ployline(np.array([x_range, ipv_collection[x_range, 1]]).T)
                smoothed_ipv_error_gs, _ = smooth_ployline(np.array([x_range, ipv_error_collection[x_range, 1]]).T)
                # plot ipv
                ax1.plot(smoothed_x, smoothed_ipv_gs[:, 1], 'red')
                # plot error bar
                ax1.fill_between(smoothed_x, smoothed_ipv_gs[:, 1] - smoothed_ipv_error_gs[:, 1],
                                 smoothed_ipv_gs[:, 1] + smoothed_ipv_error_gs[:, 1],
                                 alpha=0.4,
                                 color='red',
                                 label='estimated gs IPV')
                ax1.legend()

                # show trajectory and plans
                ax2.cla()
                ax2.set(xlim=[-22 - 13, 53 - 13], ylim=[-31 - 7.8, 57 - 7.8])
                img = plt.imread('background_pic/Jianhexianxia-v2.png')
                ax2.imshow(img, extent=[-28 - 13, 58 - 13, -42 - 7.8, 64 - 7.8])
                cv1, _ = get_central_vertices('lt_nds', [lt_info[start_time, 0]-13, lt_info[start_time, 1]-7.8])
                cv2, _ = get_central_vertices('gs_nds', [gs_info_multi[inter_id][start_time, 0]-13,
                                                         gs_info_multi[inter_id][start_time, 1]-7.8])
                ax2.plot(cv1[:, 0], cv1[:, 1])
                ax2.plot(cv2[:, 0], cv2[:, 1])

                # actual track
                ax2.scatter(lt_info[start_time:t, 0]-13, lt_info[start_time:t, 1]-7.8,
                            s=50,
                            alpha=0.5,
                            color='blue',
                            label='left-turn')
                candidates_lt = agent_lt.virtual_track_collection
                for track_lt in candidates_lt:
                    ax2.plot(track_lt[:, 0], track_lt[:, 1], color='green', alpha=0.5)
                ax2.scatter(gs_info_multi[inter_id][start_time:t, 0]-13, gs_info_multi[inter_id][start_time:t, 1]-7.8,
                            s=50,
                            alpha=0.5,
                            color='red',
                            label='go-straight')
                candidates_gs = agent_gs.virtual_track_collection
                for track_gs in candidates_gs:
                    ax2.plot(track_gs[:, 0], track_gs[:, 1], color='green', alpha=0.5)
                ax2.legend()

                plt.pause(0.3)

        elif inter_id is None:  # 如果t时刻没有交互对象，则跳过
            if print_needed:
                print('no interaction')

        elif t - start_time < 3:  # 如果交互时长小于3帧，则由于观测不足无法估计
            if print_needed:
                print('no results, more observation needed')


def visualize_nds(case_id):
    """
    播放给定驾驶数据片段中的车辆轨迹
    Parameters
    ----------
    case_id：驾驶数据片段序号

    """
    # abstract interaction info. of a given case
    # 提取给定序号的驾驶数据片段中的所有信息
    case_info = inter_info[case_id]
    # left-turn vehicle
    # 提取左转车信息
    lt_info = case_info[0]
    # go-straight vehicles
    # 提取直行车（们）的信息
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    fig, ax1 = plt.subplots(1, figsize=[8, 8])

    # 遍历左转车存在与场景中的所有时间片刻
    for t in range(np.size(lt_info, 0)):
        t_end = t + 10
        ax1.cla()
        ax1.set(xlim=[-22, 53], ylim=[-31, 57])
        img = plt.imread('background_pic/Jianhexianxia-v2.png')
        ax1.imshow(img, extent=[-28, 58, -42, 64])
        plt.text(-10, 60, 'T=' + str(t), fontsize=30)

        # position of go-straight vehicles
        # 标记直行车的位置和未来轨迹
        for gs_id in range(np.size(gs_info_multi, 0)):
            if np.size(gs_info_multi[gs_id], 0) > t and not gs_info_multi[gs_id][t, 0] == 0:
                # position
                # 当前位置
                draw_rectangle(gs_info_multi[gs_id][t, 0], gs_info_multi[gs_id][t, 1],
                               gs_info_multi[gs_id][t, 4] / math.pi * 180, ax1,
                               para_alpha=1, para_color='#7030A0')
                # ax1.scatter(gs_info_multi[gs_id][t, 0], gs_info_multi[gs_id][t, 1],
                #             s=120,
                #             alpha=0.9,
                #             color='red',
                #             label='go-straight')
                # future track
                # 未来10帧轨迹
                t_end_gs = min(t + 10, np.size(gs_info_multi[gs_id], 0))
                ax1.plot(gs_info_multi[gs_id][t:t_end_gs, 0], gs_info_multi[gs_id][t:t_end_gs, 1],
                         alpha=0.8,
                         color='red')

        # position of left-turn vehicle
        # 标记左转车的位置和未来轨迹

        # position
        # 当前位置
        draw_rectangle(lt_info[t, 0],  lt_info[t, 1],
                       lt_info[t, 4] / math.pi * 180, ax1,
                       para_alpha=1, para_color='#0E76CF')
        # ax1.scatter(lt_info[t, 0], lt_info[t, 1],
        #             s=120,
        #             alpha=0.9,
        #             color='blue',
        #             label='left-turn')
        # future track
        # 未来10帧轨迹
        ax1.plot(lt_info[t:t_end, 0], lt_info[t:t_end, 1],
                 alpha=0.8,
                 color='blue')
        # ax1.legend()
        plt.pause(0.1)
        plt.savefig('../outputs/5_gt_interaction/figures/replay case ' + str(case_id) + '/' + str(t) + '.png', dpi=300)

    # # show full track of all agents
    # ax2.plot(lt_info[:, 0], lt_info[:, 1],
    #          alpha=0.8,
    #          color='blue')
    # for gs_id in range(np.size(gs_info_multi, 0)):
    #     # find solid frames
    #     frames = np.where(gs_info_multi[gs_id][:, 0] < 1e-3)
    #     # the first solid frame id
    #     frame_start = len(frames[0])
    #     # tracks
    #     ax2.plot(gs_info_multi[gs_id][frame_start:, 0], gs_info_multi[gs_id][frame_start:, 1],
    #              alpha=0.8,
    #              color='red')
    # plt.show()


if __name__ == '__main__':
    "calculate ipv in NDS"
    # estimate IPV in natural driving data and write results into excels (along with all agents' motion info)
    # for case_index in range(130):
    #     analyze_nds(case_index)
    # analyze_nds(30)

    # visualize_nds(113)
    time1 = time.perf_counter()
    estimate_ipv_in_nds(0)
    time2 = time.perf_counter()
    print('overall time consumption: ', time2 - time1)
