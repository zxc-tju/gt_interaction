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

# virtual_agent_IPV_range = np.array([-4, -3, -2, -1, 0, 1, 2, 3, 4]) * math.pi / 9

current_nds_data_version = 5

data_path = '../data/3_parallel_game_outputs/'


def visualize_nds(case_id):
    # abstract interaction info. of a given case
    case_info = inter_info[case_id]
    # left-turn vehicle
    lt_info = case_info[0]
    # go-straight vehicles
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    fig = plt.figure(1)
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    ax1 = fig.add_subplot(111)
    # ax2 = fig.add_subplot(122)
    # ax2.set(xlim=[-22, 53], ylim=[-31, 57])
    # img = plt.imread('background_pic/Jianhexianxia.jpg')
    # ax2.imshow(img, extent=[-22, 53, -31, 57])

    for t in range(np.size(lt_info, 0)):
        t_end = t + 10
        ax1.cla()
        ax1.set(xlim=[-22, 53], ylim=[-31, 57])
        img = plt.imread('background_pic/Jianhexianxia.jpg')
        ax1.imshow(img, extent=[-22, 53, -31, 57])
        plt.text(-10, 60, 'T=' + str(t), fontsize=30)

        # position of go-straight vehicles
        for gs_id in range(np.size(gs_info_multi, 0)):
            if np.size(gs_info_multi[gs_id], 0) > t and not gs_info_multi[gs_id][t, 0] == 0:
                # position
                ax1.scatter(gs_info_multi[gs_id][t, 0], gs_info_multi[gs_id][t, 1],
                            s=120,
                            alpha=0.9,
                            color='red',
                            label='go-straight')
                # future track
                t_end_gs = min(t + 10, np.size(gs_info_multi[gs_id], 0))
                ax1.plot(gs_info_multi[gs_id][t:t_end_gs, 0], gs_info_multi[gs_id][t:t_end_gs, 1],
                         alpha=0.8,
                         color='red')

        # position of left-turn vehicle
        ax1.scatter(lt_info[t, 0], lt_info[t, 1],
                    s=120,
                    alpha=0.9,
                    color='blue',
                    label='left-turn')
        # future track
        ax1.plot(lt_info[t:t_end, 0], lt_info[t:t_end, 1],
                 alpha=0.8,
                 color='blue')
        # ax1.legend()
        plt.pause(0.1)

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


def analyze_nds(case_id):
    """
    estimate IPV in natural driving data and write results into excels
    :param case_id:
    :return:
    """
    inter_o, inter_d = find_inter_od(case_id)
    case_info = inter_info[case_id]
    lt_info = case_info[0]

    # find co-present gs agents (not necessarily interacting)
    gs_info_multi = case_info[1:inter_num[0, case_id] + 1]

    # initialize IPV
    start_time = 0
    ipv_collection = np.zeros_like(lt_info[:, 0:2])
    ipv_error_collection = np.ones_like(lt_info[:, 0:2])

    # set figure
    if illustration_needed:
        fig = plt.figure(1)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

    inter_id = 0
    inter_id_save = inter_id
    file_name = data_path + 'NDS_analysis/v' + str(current_nds_data_version) + '/' + str(case_id) + '.xlsx'

    for t in range(np.size(lt_info, 0)):

        "find current interacting agent"
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
        if flag and (t - start_time > 3):

            "====simulation-based method===="
            # generate two agents
            init_position_lt = lt_info[start_time, 0:2]
            init_velocity_lt = lt_info[start_time, 2:4]
            init_heading_lt = lt_info[start_time, 4]
            agent_lt = Agent(init_position_lt, init_velocity_lt, init_heading_lt, 'lt_nds')
            lt_track = lt_info[start_time:t + 1, 0:2]

            init_position_gs = gs_info_multi[inter_id][start_time, 0:2]
            init_velocity_gs = gs_info_multi[inter_id][start_time, 2:4]
            init_heading_gs = gs_info_multi[inter_id][start_time, 4]
            agent_gs = Agent(init_position_gs, init_velocity_gs, init_heading_gs, 'gs_nds')
            gs_track = gs_info_multi[inter_id][start_time:t + 1, 0:2]

            # estimate ipv
            agent_lt.estimate_self_ipv_in_NDS(lt_track, gs_track)
            ipv_collection[t, 0] = agent_lt.ipv
            ipv_error_collection[t, 0] = agent_lt.ipv_error

            agent_gs.estimate_self_ipv_in_NDS(gs_track, lt_track)
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
                ax2.set(xlim=[-22, 53], ylim=[-31, 57])
                img = plt.imread('background_pic/Jianhexianxia.jpg')
                ax2.imshow(img, extent=[-22, 53, -31, 57])
                cv1, _ = get_central_vertices('lt_nds', [lt_info[start_time, 0], lt_info[start_time, 1]])
                cv2, _ = get_central_vertices('gs_nds', [gs_info_multi[inter_id][start_time, 0],
                                                         gs_info_multi[inter_id][start_time, 1]])
                ax2.plot(cv1[:, 0], cv1[:, 1])
                ax2.plot(cv2[:, 0], cv2[:, 1])

                # actual track
                ax2.scatter(lt_info[start_time:t, 0], lt_info[start_time:t, 1],
                            s=50,
                            alpha=0.5,
                            color='blue',
                            label='left-turn')
                candidates_lt = agent_lt.virtual_track_collection
                for track_lt in candidates_lt:
                    ax2.plot(track_lt[:, 0], track_lt[:, 1], color='green', alpha=0.5)
                ax2.scatter(gs_info_multi[inter_id][start_time:t, 0], gs_info_multi[inter_id][start_time:t, 1],
                            s=50,
                            alpha=0.5,
                            color='red',
                            label='go-straight')
                candidates_gs = agent_gs.virtual_track_collection
                for track_gs in candidates_gs:
                    ax2.plot(track_gs[:, 0], track_gs[:, 1], color='green', alpha=0.5)
                ax2.legend()

                plt.pause(0.3)

        elif inter_id is None:
            if print_needed:
                print('no interaction')

        elif t - start_time < 3:
            if print_needed:
                print('no results, more observation needed')


def analyze_ipv_in_nds(case_id, fig=False):
    file_name = data_path + 'NDS_analysis/ipv_estimation/v' + str(current_nds_data_version) \
                + '/' + str(case_id) + '.xlsx'
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


def show_ipv_distribution():
    ipv_cross_lt = []
    ipv_cross_gs = []
    ipv_non_cross_lt = []
    ipv_non_cross_gs = []
    for i in range(np.size(inter_info, 0)):

        _, ipv_cross_temp, ipv_non_cross_temp = analyze_ipv_in_nds(i, False)
        if len(ipv_cross_temp) > 0:
            ipv_cross_lt.append(ipv_cross_temp[:, 0])
            ipv_cross_gs.append(ipv_cross_temp[:, 7])
        if len(ipv_non_cross_temp) > 0:
            for idx in range(len(ipv_non_cross_temp)):
                # print(ipv_non_cross[idx][:, 0])
                ipv_non_cross_lt.append(ipv_non_cross_temp[idx][:, 0])
                ipv_non_cross_gs.append(ipv_non_cross_temp[idx][:, 7])

    "calculate mean ipv value of each type"
    mean_ipv_cross_lt = np.array([np.mean(ipv_cross_lt[0])])
    mean_ipv_cross_gs = np.array([np.mean(ipv_cross_gs[0])])
    mean_ipv_non_cross_lt = np.array([np.mean(ipv_non_cross_lt[0])])
    mean_ipv_non_cross_gs = np.array([np.mean(ipv_non_cross_gs[0])])
    for i in range(len(ipv_cross_lt) - 1):
        if np.size(ipv_cross_lt[i + 1], 0) > 4:
            mean_temp1 = np.array([np.mean(ipv_cross_lt[i + 1])])
            mean_ipv_cross_lt = np.concatenate((mean_ipv_cross_lt, mean_temp1), axis=0)
    for i in range(len(ipv_cross_gs) - 1):
        if np.size(ipv_cross_gs[i + 1], 0) > 4:
            mean_temp2 = np.array([np.mean(ipv_cross_gs[i + 1])])
            mean_ipv_cross_gs = np.concatenate((mean_ipv_cross_gs, mean_temp2), axis=0)
    for i in range(len(ipv_non_cross_lt) - 1):
        if np.size(ipv_non_cross_lt[i + 1], 0) > 4:
            mean_temp3 = np.array([np.mean(ipv_non_cross_lt[i + 1])])
            mean_ipv_non_cross_lt = np.concatenate((mean_ipv_non_cross_lt, mean_temp3), axis=0)
    for i in range(len(ipv_non_cross_gs) - 1):
        if np.size(ipv_non_cross_gs[i + 1], 0) > 4:
            mean_temp4 = np.array([np.mean(ipv_non_cross_gs[i + 1])])
            mean_ipv_non_cross_gs = np.concatenate((mean_ipv_non_cross_gs, mean_temp4), axis=0)

    filename = './outputs/ipv_distribution_v' + str(current_nds_data_version) + '.xlsx'
    with pd.ExcelWriter(filename) as writer:

        data1 = np.vstack((mean_ipv_cross_gs, mean_ipv_cross_lt))
        df_ipv_distribution = pd.DataFrame(data1.T, columns=['cross_gs', 'cross_lt'])
        df_ipv_distribution.to_excel(writer, startcol=0, index=False)

        data2 = np.vstack((mean_ipv_non_cross_gs, mean_ipv_non_cross_lt))
        df_ipv_distribution = pd.DataFrame(data2.T, columns=['non_cross_gs', 'non_cross_lt'])
        df_ipv_distribution.to_excel(writer, startcol=2, index=False)

    plt.figure(1)
    plt.title('Left-turn vehicle rushed')
    plt.hist(mean_ipv_cross_lt,
             alpha=0.5,
             color='blue',
             label='left-turn vehicle')
    plt.hist(mean_ipv_cross_gs,
             alpha=0.5,
             color='red',
             label='go-straight vehicle')
    plt.legend()
    plt.xlabel('IPV')
    plt.ylabel('Counts')

    plt.figure(2)
    plt.title('Left-turn vehicle yielded')
    plt.hist(mean_ipv_non_cross_lt,
             alpha=0.5,
             color='blue',
             label='left-turn vehicle')
    plt.hist(mean_ipv_non_cross_gs,
             alpha=0.5,
             color='red',
             label='go-straight vehicle')
    plt.legend()
    plt.xlabel('IPV')
    plt.ylabel('Counts')
    plt.show()


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


def divide_pet_in_nds(save_collection_analysis=False,
                      save_divided_trj=False,
                      show_fig=False):
    """

    :param save_collection_analysis:
    :param save_divided_trj: divide all the trajectory according to the ipv
    :param show_fig:
    :return:
    """
    comp_lt_collection = []
    coop_lt_collection = []
    comp_gs_collection = []
    coop_gs_collection = []
    all_collection = []
    non_cross_apet_collection = []
    num_coop_lt_trj = 0
    num_comp_lt_trj = 0
    # date = datetime.now().strftime("%Y_%m_%d-%I:%M:%S_%p")
    date = datetime.now().strftime("%Y%m%d")
    filename_divided_trj = data_path + 'NDS_analysis/ipv_estimation/v' + str(current_nds_data_version) \
                           + '/divide_trj_by_ipv' + date + '.xlsx'

    fig = plt.figure(1)
    ax1 = fig.add_subplot(111)
    ax1.set(xlim=[-23, 53], ylim=[-31, 57])
    img = plt.imread('background_pic/Jianhexianxia.jpg')
    ax1.imshow(img, extent=[-23, 53, -31, 57])

    if save_divided_trj:
        workbook = xlsxwriter.Workbook(filename_divided_trj)
        # 新增工作簿。
        worksheet = workbook.add_worksheet('lt_coop')
        #  关闭工作簿。在文件夹中打开文件，查看写入的结果。
        workbook.close()  # 一定要关闭workbook后才会产生文件！

    for case_index in range(case_number):

        cross_id, data_cross, data_non_cross = analyze_ipv_in_nds(case_index)
        # save data into an excel with the format of:
        # 0-ipv_lt | ipv_lt_error | lt_px | lt_py  | lt_vx  | lt_vy  | lt_heading  |...
        # 7-ipv_gs | ipv_gs_error | gs_px | gs_py  | gs_vx  | gs_vy  | gs_heading  |
        o, _ = find_inter_od(case_index)
        start_frame = int(o[cross_id])

        "1. analyse cross cases"
        if not cross_id == -1:  # and case_index not in {114, 129}:

            # go-straight vehicles
            gs_info_multi = inter_info[case_index][1:inter_num[0, case_index] + 1]
            gs_trj = gs_info_multi[cross_id][start_frame:, 0:2]
            # left-turn vehicle
            lt_trj = inter_info[case_index][0][start_frame:, 0:2]

            pet_temp, _ = cal_pet(lt_trj, gs_trj, 'pet')
            apet, ttcp_lt, ttcp_gs = cal_pet(lt_trj, gs_trj, 'apet')

            data_cross = data_cross[4:, :]
            lt_ipv = np.mean(data_cross[:, 0])
            gs_ipv = np.mean(data_cross[:, 7])

            vel_lt = np.linalg.norm(data_cross[:, 4:6], axis=1)
            vel_mean_lt = np.mean(vel_lt)
            vel_gs = np.linalg.norm(data_cross[:, 11:13], axis=1)
            vel_mean_gs = np.mean(vel_gs)
            vel_gs_max = max(vel_gs)
            vel_ave = (vel_mean_lt + vel_mean_gs) * 0.5

            acc_gs = (vel_gs[1:] - vel_gs[:-1]) / 0.12
            acc_mean_gs = np.mean(acc_gs)
            acc_min_gs = min(acc_gs)

            all_collection.append([case_index, lt_ipv, gs_ipv, pet_temp, vel_ave, acc_mean_gs, acc_min_gs])

            # divide and show trajectories according to ipv
            if lt_ipv < 0:
                comp_lt_collection.append([case_index, lt_ipv, gs_ipv, pet_temp, vel_ave,
                                           acc_mean_gs, acc_min_gs, ttcp_lt[0], ttcp_gs[0]])
                ax1.plot(inter_info[case_index][0][:, 0], inter_info[case_index][0][:, 1], color="red", alpha=0.5)
                # alpha=-np.mean(ipv_data_cross[:, 0] * (1 - ipv_data_cross[:, 1])) / 1.57

                if save_divided_trj:
                    lt_trj_comp = pd.DataFrame(inter_info[case_index][0][:, 0:2],
                                               columns=['case-' + str(case_index) + '-x', 'y'])
                    with pd.ExcelWriter(filename_divided_trj,
                                        mode='a',
                                        if_sheet_exists="overlay",
                                        engine="openpyxl") as writer:
                        lt_trj_comp.to_excel(writer, startcol=2 * num_comp_lt_trj, index=False, sheet_name='lt_comp')

                num_comp_lt_trj += 1
            else:
                coop_lt_collection.append([case_index, lt_ipv, gs_ipv, pet_temp, vel_ave,
                                           acc_mean_gs, acc_min_gs, ttcp_lt[0], ttcp_gs[0]])
                ax1.plot(inter_info[case_index][0][:, 0], inter_info[case_index][0][:, 1], color="green", alpha=0.5)

                if save_divided_trj:
                    lt_trj_coop = pd.DataFrame(inter_info[case_index][0][:, 0:2],
                                               columns=['case-' + str(case_index) + '-x', 'y'])
                    with pd.ExcelWriter(filename_divided_trj,
                                        mode='a',
                                        if_sheet_exists="overlay",
                                        engine="openpyxl") as writer:
                        lt_trj_coop.to_excel(writer, startcol=2 * num_coop_lt_trj, index=False, sheet_name='lt_coop')

                num_coop_lt_trj += 1

            # delete invalid (0,0) positions
            gs_trj_temp = gs_info_multi[cross_id][:, 0:2]
            invalid_len = len((np.where(gs_trj_temp[:, 0] == 0))[0])

            if gs_ipv < 0:
                comp_gs_collection.append([case_index, lt_ipv, gs_ipv, pet_temp, vel_ave, vel_mean_gs, vel_gs_max])
                ax1.plot(gs_info_multi[cross_id][invalid_len:, 0],
                         gs_info_multi[cross_id][invalid_len:, 1], color="red", alpha=0.5)

            else:
                coop_gs_collection.append([case_index, lt_ipv, gs_ipv, pet_temp, vel_ave, vel_mean_gs, vel_gs_max])
                ax1.plot(gs_info_multi[cross_id][invalid_len:, 0],
                         gs_info_multi[cross_id][invalid_len:, 1], color="green", alpha=0.5)

        "2. analyse non-cross cases"
        for non_cross_id in range(len(data_non_cross)):
            if not non_cross_id == cross_id:
                start_frame = int(o[non_cross_id])
                # go-straight vehicles
                gs_info_multi = inter_info[case_index][1:inter_num[0, case_index] + 1]
                gs_trj = gs_info_multi[non_cross_id][start_frame:, 0:2]

                # left-turn vehicle
                lt_trj = inter_info[case_index][0][start_frame:, 0:2]

                # pet_temp, _ = cal_pet(lt_trj, gs_trj, 'pet')
                if np.size(gs_trj, 0)>8:
                    apet, ttcp_lt, ttcp_gs = cal_pet(lt_trj, gs_trj, 'apet')
                    non_cross_apet_collection.append([ttcp_lt[0], ttcp_gs[0]])

    # reconstruct data into array
    pet_collection = np.array(all_collection)
    comp_lt_collection = np.array(comp_lt_collection)
    coop_lt_collection = np.array(coop_lt_collection)
    comp_gs_collection = np.array(comp_gs_collection)
    coop_gs_collection = np.array(coop_gs_collection)
    non_cross_apet_collection = np.array(non_cross_apet_collection)

    if show_fig:
        plt.figure(2)
        plt.title('PET distribution (divided by ipv of LT vehicles)')
        plt.hist(comp_lt_collection[:, 3], bins=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                 alpha=0.5,
                 color='red',
                 label='competitive')
        plt.hist(coop_lt_collection[:, 3], bins=[1, 2, 3, 4, 5, 6, 7, 8, 9],
                 alpha=0.5,
                 color='green',
                 label='cooperative')
        plt.legend()
        plt.xlabel('PET')
        plt.ylabel('Counts')

        plt.figure(3)
        ax1 = plt.subplot(121)
        plt.title('mean GS acc distribution (divided by ipv of LT vehicles)')
        ax1.hist(comp_lt_collection[:, 5],
                 alpha=0.5,
                 color='red',
                 label='competitive')
        ax1.hist(coop_lt_collection[:, 5],
                 alpha=0.5,
                 color='green',
                 label='cooperative')
        ax1.legend()
        plt.xlabel('mean GS acc')
        plt.ylabel('Counts')

        ax2 = plt.subplot(122)
        plt.title('min GS acc distribution (divided by ipv of LT vehicles)')
        ax2.hist(comp_lt_collection[:, 6],
                 alpha=0.5,
                 color='red',
                 label='competitive')
        ax2.hist(coop_lt_collection[:, 6],
                 alpha=0.5,
                 color='green',
                 label='cooperative')
        ax2.legend()
        plt.xlabel('min GS acc')
        plt.ylabel('Counts')

        plt.figure(4)
        ax1 = plt.subplot(121)
        plt.title('mean GS speed (divided by ipv of GS vehicles)')
        ax1.hist(comp_gs_collection[:, 5],
                 alpha=0.5,
                 color='red',
                 label='competitive')
        ax1.hist(coop_gs_collection[:, 5],
                 alpha=0.5,
                 color='green',
                 label='cooperative')
        ax1.legend()
        plt.xlabel('mean GS speed')
        plt.ylabel('Counts')

        ax2 = plt.subplot(122)
        plt.title('max GS speed distribution (divided by ipv of GS vehicles)')
        ax2.hist(comp_gs_collection[:, 6],
                 alpha=0.5,
                 color='red',
                 label='competitive')
        ax2.hist(coop_gs_collection[:, 6],
                 alpha=0.5,
                 color='green',
                 label='cooperative')
        ax2.legend()
        plt.xlabel('max GS speed')
        plt.ylabel('Counts')

        plt.show()

    if save_collection_analysis:
        # save pet data
        filename = data_path + 'NDS_analysis/ipv_estimation/v' + str(current_nds_data_version) \
                   + '/collection_analysis_v' + str(current_nds_data_version) + '.xlsx'
        with pd.ExcelWriter(filename) as writer:
            df_all = pd.DataFrame(pet_collection,
                                  columns=['case_index', 'lt_ipv', 'gs_ipv',
                                           'pet_temp', 'vel', 'acc_mean_gs', 'acc_min_gs'])
            # data divided by LT's IPV
            df_comp_lt = pd.DataFrame(comp_lt_collection,
                                      columns=['case_index', 'lt_ipv', 'gs_ipv', 'pet_temp', 'vel',
                                               'acc_mean_gs', 'acc_min_gs', 'init_ttcp_lt', 'init_ttcp_gs'])
            df_coop_lt = pd.DataFrame(coop_lt_collection,
                                      columns=['case_index', 'lt_ipv', 'gs_ipv', 'pet_temp', 'vel',
                                               'acc_mean_gs', 'acc_min_gs', 'init_ttcp_lt', 'init_ttcp_gs'])
            # data divided by GS's IPV
            df_comp_gs = pd.DataFrame(comp_gs_collection,
                                      columns=['case_index', 'lt_ipv', 'gs_ipv', 'pet_temp',
                                               'vel_ave', 'vel_mean_gs', 'vel_gs_max'])
            df_coop_gs = pd.DataFrame(coop_gs_collection,
                                      columns=['case_index', 'lt_ipv', 'gs_ipv', 'pet_temp',
                                               'vel_ave', 'vel_mean_gs', 'vel_gs_max'])

            df_non_cross_init_apet = pd.DataFrame(non_cross_apet_collection, columns=['ttcp_lt', 'ttcp_gs'])

            df_all.to_excel(writer, startcol=0, index=False, sheet_name="all")
            df_comp_lt.to_excel(writer, startcol=0, index=False, sheet_name="competitive_lt")
            df_coop_lt.to_excel(writer, startcol=0, index=False, sheet_name="cooperative_lt")
            df_comp_gs.to_excel(writer, startcol=0, index=False, sheet_name="competitive_gs")
            df_coop_gs.to_excel(writer, startcol=0, index=False, sheet_name="cooperative_gs")
            df_non_cross_init_apet.to_excel(writer, startcol=0, index=False, sheet_name="non_cross_init_apet")


def show_crossing_event(case_index, isfig=True, issavedata=False):
    cross_id, data_cross, _ = analyze_ipv_in_nds(case_index)
    o, _ = find_inter_od(case_index)
    start_frame = int(o[cross_id])

    if not cross_id == -1:

        # go-straight vehicles
        gs_info_multi = inter_info[case_index][1:inter_num[0, case_index] + 1]
        gs_trj = gs_info_multi[cross_id][start_frame:, 0:2]
        # left-turn vehicle
        lt_trj = inter_info[case_index][0][start_frame:, 0:2]

        # calculate PET of the whole event
        pet, conflict_point = cal_pet(lt_trj, gs_trj, "pet")
        if isfig:
            fig = plt.figure(1)
            ax1 = fig.add_subplot(111)
            ax1.set(xlim=[-23, 53], ylim=[-31, 57])
            img = plt.imread('background_pic/Jianhexianxia.jpg')
            ax1.imshow(img, extent=[-23, 53, -31, 57])

            # ax1.scatter(lt_trj[t_step_lt, 0], lt_trj[t_step_lt, 1])
            # ax1.scatter(gs_trj[t_step_gs, 0], gs_trj[t_step_gs, 1])
            ax1.scatter(conflict_point[0], conflict_point[1], color="black", alpha=0.5)

            plt.text(55, 30, 'PET:' + str(pet))
            lt_mean_ipv = np.mean(data_cross[4:, 0])
            gs_mean_ipv = np.mean(data_cross[4:, 7])
            ax1.text(0, 60, 'LT:' + str(lt_mean_ipv), fontsize=10)
            ax1.text(0, 65, 'GS:' + str(gs_mean_ipv), fontsize=10)
            #

            # if np.mean(data_cross[4:, 0] * (1 - data_cross[4:, 1])) < 0:
            if np.mean(data_cross[4:, 0]) < 0:
                ax1.plot(lt_trj[:, 0], lt_trj[:, 1], color="red", alpha=0.5)
                # alpha=-np.mean(ipv_data_cross[:, 0] * (1 - ipv_data_cross[:, 1])) / 1.57
                ax1.plot(gs_trj[:, 0], gs_trj[:, 1], color="red", alpha=0.5)
                # alpha=-np.mean(ipv_data_cross[:, 0] * (1 - ipv_data_cross[:, 1])) / 1.57
            else:
                ax1.plot(lt_trj[:, 0], lt_trj[:, 1], color="green", alpha=0.5)
                ax1.plot(gs_trj[:, 0], gs_trj[:, 1], color="green", alpha=0.5)
            # if issavedata:
            # plt.savefig('./outputs/NDS_analysis/crossing_event_v' + str(current_nds_data_version)
            #             + '/' + str(case_index) + '.png')

        # calculate anticipated PET of the process
        apet, ttc_lt, ttc_gs = cal_pet(lt_trj, gs_trj, "apet")
        x_range = range(start_frame, start_frame + len(apet))

        if issavedata:
            df_xrange = pd.DataFrame(x_range, columns=['time'])
            df_apet = pd.DataFrame(apet, columns=['APET'])
            df_ttc_lt = pd.DataFrame(ttc_lt[0: len(apet)], columns=['TTCP_lt'])
            df_ttc_gs = pd.DataFrame(ttc_gs[0: len(apet)], columns=['TTCP_gs'])
            df_pet = pd.DataFrame(np.array([pet]), columns=['PET'])
            filename = data_path + 'NDS_analysis/v' + str(current_nds_data_version) \
                       + '/APET_case_' + str(case_index) + '.xlsx'
            with pd.ExcelWriter(filename) as writer:
                df_xrange.to_excel(writer, startcol=0, index=False)
                df_apet.to_excel(writer, startcol=1, index=False)
                df_ttc_lt.to_excel(writer, startcol=2, index=False)
                df_ttc_gs.to_excel(writer, startcol=3, index=False)
                df_pet.to_excel(writer, startcol=4, index=False)

        if isfig:
            plt.figure(2)
            plt.plot(x_range, apet, color='black')
            plt.plot(x_range, ttc_lt[0: len(apet)], color='blue')
            plt.plot(x_range, ttc_gs[0: len(apet)], color='purple')
            # plt.ylim([-10, 30])


if __name__ == '__main__':
    "calculate ipv in NDS"
    # estimate IPV in natural driving data and write results into excels (along with all agents' motion info)
    # for case_index in range(99, 100):
    #     analyze_nds(case_index)
    # analyze_nds(30)

    "show trajectories in NDS"
    # visualize_nds(129)

    "find crossing event and the ipv of yield front-coming vehicle (if there is)"
    # cross_id, ipv_data_cross, ipv_data_non_cross = analyze_ipv_in_nds(30, True)

    "show ipv distribution in whole dataset"
    # show_ipv_distribution()

    "find the origin and ending of the each interaction event in a single case"
    # o, d = find_inter_od(30)

    # draw_rectangle(5, 5, 45)

    "show properties divided by the ipv of two agents"
    divide_pet_in_nds(save_collection_analysis=True, save_divided_trj=True, show_fig=False)

    "show crossing trajectories and pet process in a case"
    # show_crossing_event(30, isfig=True, issavedata=True)
