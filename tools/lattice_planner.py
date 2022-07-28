from tools.Lattice import TrajPoint, Obstacle, CalcRefLine, SampleBasis, LocalPlanner, Dist
import math
import numpy as np
import matplotlib.pyplot as plt

M_PI = math.pi

# 车辆属性, global const. and local var. check!
VEH_L = 1
VEH_W = 0.5
MAX_V = 200
MIN_V = -10000

'''
MAX_A = 10
MIN_A = -20
MAX_LAT_A = 100 #参考apollo，横向约束应该是给到向心加速度，而不是角速度
'''


def lattice_planning(path_data, obs_data, init_state, show_res=False):
    rx = path_data[:, 0]
    ry = path_data[:, 1]
    cts_points = np.array([rx, ry])
    path_points = CalcRefLine(cts_points)
    obstacles = [Obstacle([obs_data['px'], obs_data['py'], obs_data['v'], 3, 1.5, M_PI / 6])]
    obstacles = [Obstacle([rx[150], ry[150], obs_data['v'], 3, 1.5, M_PI / 6])]
    # obstacles = [Obstacle([rx[150], ry[150], 0, 0.5, 0.5, M_PI / 6]),
    #              Obstacle([-260, -500, 0, 1, 1, M_PI / 2]),
    #              Obstacle([rx[500] + 1, ry[500], 0, 1, 1, M_PI / 3])]

    tp_list = [init_state['px'], init_state['py'], init_state['v'], 0, 3., 0]
    # tp_list = [rx[0], ry[0], 0, 0, 3., 0]
    traj_point = TrajPoint(tp_list)
    traj_point.MatchPath(path_points)

    for obstacle in obstacles:
        obstacle.MatchPath(path_points)

    theta_thr = M_PI / 6  # delta theta threhold, deviation from matched path
    ttcs = [3, 4, 5]  # static ascending time-to-collision, sec

    samp_basis = SampleBasis(traj_point, theta_thr, ttcs)
    local_planner = LocalPlanner(traj_point, path_points, obstacles, samp_basis)
    print(local_planner.status, local_planner.to_stop)
    traj_points_opt = local_planner.LocalPlanning(traj_point, path_points, obstacles, samp_basis)
    print(local_planner.v_end, samp_basis.v_end)

    if not traj_points_opt:
        print("扩大范围")
        theta_thr_ = M_PI / 3
        ttcs_ = [2, 3, 4, 5, 6, 7, 8]
        samp_basis = SampleBasis(traj_point, theta_thr_, ttcs_)
        local_planner = LocalPlanner(traj_point, path_points, obstacles, samp_basis)
        traj_points_opt = local_planner.LocalPlanning(traj_point, path_points, obstacles, samp_basis)
    if not traj_points_opt:
        traj_points = [[0, 0, 0, 0, 0, 0]]
        print("无解")
    else:
        traj_points = []
        for tp_opt in traj_points_opt:
            traj_points.append([tp_opt.x, tp_opt.y, tp_opt.v, tp_opt.a, tp_opt.theta, tp_opt.kappa])
    tx = [x[0] for x in traj_points]
    ty = [y[1] for y in traj_points]

    if show_res:
        plt.figure(1)
        plt.plot(rx, ry, 'b')
        plt.plot(traj_point.x, traj_point.y, 'or')
        plt.plot(tx, ty, 'r')
        for obstacle in obstacles:
            plt.gca().add_patch(plt.Rectangle((obstacle.corner[0], obstacle.corner[1]), obstacle.length,
                                              obstacle.width, color='r', angle=obstacle.heading * 180 / M_PI))
            plt.axis('scaled')
        plt.show()

    return [[x[0], x[1], x[2]*np.cos(x[4]), x[2]*np.sin(x[4]), x[4]] for x in traj_points]


if __name__ == '__main__':

    path_point = np.loadtxt("roadMap_lzjSouth1.txt")
    path_point = path_point[:, 1:3]
    obstacle_data = {'px': -260,
                     'py': -500,
                     'v': 0}
    initial_state = {'px': -251,
                     'py': -503,
                     'v': 5}
    res = lattice_planning(path_point, obstacle_data, initial_state, show_res=True)

