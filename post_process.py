"""
analysis of simulation results
"""

import pickle
import xlsxwriter
import math
import gc
import pandas as pd
from matplotlib import pyplot as plt
from tools.utility import get_central_vertices, smooth_ployline, draw_rectangle
from NDS_analysis import cal_pet
import numpy as np

ipv_update_method = 1
show_gif = 1
save_fig = 1
save_data = 0
save_fig_for_paper = 1


def get_results(rd, case_id):
    # import data
    version_num = '28'
    tag = 'VGIM-dyna-gs-4'
    filedir = '../data/3_parallel_game_outputs/simulation/version' + str(version_num)
    filename = filedir + '/data/agents_infocase' + '_round' + str(rd) + '-' + tag + '.pckl'
    # filename = filedir + '/data/agents_info' + '_round_' + str(rd) + '_case_' + str(case_id) + '.pckl'
    # filename = filedir + '/data/' + 'NE-Coop_task_1_case_0.pckl'
    f = open(filename, 'rb')
    agent_lt, agent_gs, _, _, _ = pickle.load(f)
    f.close()

    "====data abstraction===="
    # lt track (observed and planned)
    lt_ob_trj = agent_lt.observed_trajectory[:, 0:2]
    lt_heading = agent_lt.observed_trajectory[:, 4] / math.pi * 180
    lt_trj_coll = agent_lt.trj_solution_collection[0][:, 0:2]
    # skip one trj every two trjs
    for i in range(int(len(lt_ob_trj) / 2 - 1)):
        coll_temp = agent_lt.trj_solution_collection[(i + 1) * 2][:, 0:2]
        lt_trj_coll = np.concatenate([lt_trj_coll, coll_temp], axis=1)

    # gs track (observed and planned)
    gs_ob_trj = agent_gs.observed_trajectory[:, 0:2]
    gs_heading = agent_gs.observed_trajectory[:, 4] / math.pi * 180
    gs_trj_coll = agent_gs.trj_solution_collection[0][:, 0:2]
    # skip one trj every two trjs
    for i in range(int(len(lt_ob_trj) / 2 - 1)):
        coll_temp = agent_gs.trj_solution_collection[(i + 1) * 2][:, 0:2]
        gs_trj_coll = np.concatenate([gs_trj_coll, coll_temp], axis=1)

    # link from gs to lt
    link = np.concatenate([[lt_ob_trj[0, :]], [gs_ob_trj[0, :]]])
    # skip one trj every two trjs
    for i in range(int(len(lt_ob_trj) / 2)):
        link = np.concatenate([link, np.concatenate([[lt_ob_trj[(i + 1) * 2, :]],
                                                     [gs_ob_trj[(i + 1) * 2, :]]])], axis=1)
    # process data for figures
    cv_lt, progress_lt = get_central_vertices('lt', None)
    cv_gs, progress_gs = get_central_vertices('gs', None)

    "====calculate PET===="
    pet, _, _ = cal_pet(lt_ob_trj, gs_ob_trj, 'apet')

    "====save data to excel===="
    if save_data:
        df_lt_ob_trj = pd.DataFrame(lt_ob_trj)
        df_lt_trj_coll = pd.DataFrame(lt_trj_coll)
        df_gs_ob_trj = pd.DataFrame(gs_ob_trj)
        df_gs_trj_coll = pd.DataFrame(gs_trj_coll)
        df_link = pd.DataFrame(link)
        df_pet = pd.DataFrame(pet)
        df_estimated_gs_ipv = pd.DataFrame(agent_lt.estimated_inter_agent.ipv_collection, columns=['ipv'])
        df_estimated_gs_ipv_error = pd.DataFrame(agent_lt.estimated_inter_agent.ipv_error_collection, columns=['error'])
        df_estimated_lt_ipv = pd.DataFrame(agent_gs.estimated_inter_agent.ipv_collection, columns=['ipv'])
        df_estimated_lt_ipv_error = pd.DataFrame(agent_gs.estimated_inter_agent.ipv_error_collection, columns=['error'])

        filename_data = filedir + '/excel/output' + '_round_' + str(rd) + '_case_' + tag + '.xlsx'
        workbook = xlsxwriter.Workbook(filename_data)
        # 新增工作簿。
        worksheet = workbook.add_worksheet('lt_ob_trj')
        #  关闭工作簿。在文件夹中打开文件，查看写入的结果。
        workbook.close()  # 一定要关闭workbook后才会产生文件！

        with pd.ExcelWriter(filename_data,
                            mode='a',
                            if_sheet_exists="overlay",
                            engine="openpyxl") as writer:
            df_lt_ob_trj.to_excel(writer, index=False, sheet_name='lt_ob_trj')
            df_gs_ob_trj.to_excel(writer, index=False, sheet_name='gs_ob_trj')
            df_lt_trj_coll.to_excel(writer, index=False, sheet_name='lt_trj_coll')
            df_gs_trj_coll.to_excel(writer, index=False, sheet_name='gs_trj_coll')
            df_link.to_excel(writer, index=False, sheet_name='link')
            df_pet.to_excel(writer, index=False, sheet_name='PET')
            df_estimated_gs_ipv.to_excel(writer, index=False, startcol=0, sheet_name='ipv_gs')
            df_estimated_gs_ipv_error.to_excel(writer, index=False, startcol=1, sheet_name='ipv_gs')
            df_estimated_lt_ipv.to_excel(writer, index=False, startcol=0, sheet_name='ipv_lt')
            df_estimated_lt_ipv_error.to_excel(writer, index=False, startcol=1, sheet_name='ipv_lt')

    if save_fig or show_gif:

        # set figure
        fig = plt.figure(figsize=(12, 12))
        ax1 = plt.subplot(311)
        ax2 = fig.add_subplot(312)
        ax3 = fig.add_subplot(313)
        img = plt.imread('background_pic/T_intersection.jpg')

        "====final observed_trajectory===="
        num_frame = len(lt_ob_trj)
        # num_frame = 16
        for t in range(num_frame):
            ax1.cla()
            ax1.imshow(img, extent=[-9.1, 24.9, -13, 8])
            ax1.set(xlim=[-9.1, 35], ylim=[-13, 8])
            if not save_fig_for_paper:
                # central vertices
                ax1.plot(cv_lt[:, 0], cv_lt[:, 1], 'r-')
                ax1.plot(cv_gs[:, 0], cv_gs[:, 1], 'b-')
            # # ---- show position: version 1 ---- #
            # # left-turn
            # ax1.scatter(lt_ob_trj[:t + 1, 0],
            #             lt_ob_trj[:t + 1, 1],
            #             s=120,
            #             alpha=0.4,
            #             color='red',
            #             label='left-turn')
            # # go-straight
            # ax1.scatter(gs_ob_trj[:t + 1, 0],
            #             gs_ob_trj[:t + 1, 1],
            #             s=120,
            #             alpha=0.4,
            #             color='blue',
            #             label='go-straight')

            # ---- show position: version 2 ----#
            for s in range(t+1):
                draw_rectangle(lt_ob_trj[s, 0], lt_ob_trj[s, 1], lt_heading[s], ax1,
                               para_alpha=(s+1)/num_frame, para_color='#0E76CF')
                draw_rectangle(gs_ob_trj[s, 0], gs_ob_trj[s, 1], gs_heading[s], ax1,
                               para_alpha=(s+1)/num_frame, para_color='#7030A0')

                # non-interacting following car
                draw_rectangle(30-s*0.5-0.5, -2, 0, ax1, para_alpha=(s+1)/num_frame, para_color='gray')

            if not save_fig_for_paper:
                if t < len(lt_ob_trj) - 1:
                    # real-time virtual plans of ## ego ## at time step t
                    lt_track = agent_lt.trj_solution_collection[t]
                    ax1.plot(lt_track[:, 0], lt_track[:, 1], '--', linewidth=3)
                    gs_track = agent_gs.trj_solution_collection[t]
                    ax1.plot(gs_track[:, 0], gs_track[:, 1], '--', linewidth=3)
                    if ipv_update_method == 1:
                        # real-time virtual plans of ## interacting agent ## at time step t
                        candidates_lt = agent_lt.estimated_inter_agent.virtual_track_collection[t]
                        candidates_gs = agent_gs.estimated_inter_agent.virtual_track_collection[t]
                        for track_lt in candidates_lt:
                            ax1.plot(track_lt[:, 0], track_lt[:, 1], color='green', alpha=0.5)
                        for track_gs in candidates_gs:
                            ax1.plot(track_gs[:, 0], track_gs[:, 1], color='green', alpha=0.5)
                # position link
                ax1.plot([lt_ob_trj[t, 0], gs_ob_trj[t, 0]],
                         [lt_ob_trj[t, 1], gs_ob_trj[t, 1]],
                         color='gray',
                         alpha=0.2)
            if show_gif:
                plt.pause(0.1)
        if not save_fig_for_paper:
            # full position link
            for t in range(len(lt_ob_trj)):
                ax1.plot([lt_ob_trj[t, 0], gs_ob_trj[t, 0]],
                         [lt_ob_trj[t, 1], gs_ob_trj[t, 1]],
                         color='gray',
                         alpha=0.1)
        ax1.set_title('gs_' + str(agent_gs.ipv) + '_lt_' + str(agent_lt.ipv), fontsize=12)

        "====ipv estimation===="
        x_range = np.array(range(len(agent_lt.estimated_inter_agent.ipv_collection)))

        # ====draw left turn
        y_lt = np.array(agent_lt.estimated_inter_agent.ipv_collection)
        point_smoothed_lt, _ = smooth_ployline(np.array([x_range, y_lt]).T)
        x_smoothed_lt = point_smoothed_lt[:, 0]
        y_lt_smoothed = point_smoothed_lt[:, 1]
        ax2.plot(x_smoothed_lt, y_lt_smoothed,
                 alpha=1,
                 color='blue',
                 label='estimated gs IPV')
        if ipv_update_method == 1:
            # error bar
            y_error_lt = np.array(agent_lt.estimated_inter_agent.ipv_error_collection)
            error_smoothed_lt, _ = smooth_ployline(np.array([x_range, y_error_lt]).T)
            y_error_lt_smoothed = error_smoothed_lt[:, 1]
            ax2.fill_between(x_smoothed_lt, y_lt_smoothed - y_error_lt_smoothed, y_lt_smoothed + y_error_lt_smoothed,
                             alpha=0.3,
                             color='blue',
                             label='estimated gs IPV')
        # ground truth
        ax2.plot(x_range, agent_gs.ipv * np.ones_like(x_range),
                 linewidth=5,
                 label='actual gs IPV')

        # ====draw go straight
        y_gs = np.array(agent_gs.estimated_inter_agent.ipv_collection)
        # smoothen data
        point_smoothed_gs, _ = smooth_ployline(np.array([x_range, y_gs]).T)
        x_smoothed_gs = point_smoothed_gs[:, 0]
        y_gs_smoothed = point_smoothed_gs[:, 1]
        ax2.plot(x_smoothed_gs, y_gs_smoothed,
                 alpha=1,
                 color='red',
                 label='estimated gs IPV')
        if ipv_update_method == 1:
            # error bar
            y_error_gs = np.array(agent_gs.estimated_inter_agent.ipv_error_collection)
            error_smoothed_gs, _ = smooth_ployline(np.array([x_range, y_error_gs]).T)
            y_error_gs_smoothed = error_smoothed_gs[:, 1]
            ax2.fill_between(x_smoothed_gs, y_gs_smoothed - y_error_gs_smoothed, y_gs_smoothed + y_error_gs_smoothed,
                             alpha=0.3,
                             color='red',
                             label='estimated lt IPV')
        # ground truth
        ax2.plot(x_range, agent_lt.ipv * np.ones_like(x_range),
                 linewidth=5,
                 label='actual lt IPV')

        "====PET===="
        ax3.plot(pet)

        if show_gif:
            plt.show()

        # save figure
        if save_fig:
            plt.savefig(filedir + '/figures/' + tag
                        + '_round_' + str(rd)
                        + '_case_' + str(case_id) + '.svg', format='svg')
            # plt.clf()
            # plt.close()
            # gc.collect()


if __name__ == '__main__':
    # ipv_list = [-3, -2, -1, 0, 1, 2, 3]
    # ipv_list = [-2, 0, 2]
    # ipv_list = [-2]
    # for gs in [2]:
    #     for lt in ipv_list:
    #         get_results(gs, lt)
    rd = 3
    caseid = 1
    get_results(rd, caseid)
