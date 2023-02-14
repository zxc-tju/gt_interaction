import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
import math
from shapely.geometry import LineString
import matplotlib.patches as patches
import matplotlib.transforms as mt
import scipy.signal
from tools.Lattice import PathPoint


def smooth_ployline(cv_init, point_num=1000):
    cv = cv_init
    list_x = cv[:, 0]
    list_y = cv[:, 1]
    if type(cv) is not np.ndarray:
        cv = np.array(cv)
    delta_cv = cv[1:, ] - cv[:-1, ]
    s_cv = np.linalg.norm(delta_cv, axis=1)

    s_cv = np.array([0] + list(s_cv))
    s_cv = np.cumsum(s_cv)

    bspl_x = splrep(s_cv, list_x, s=0.1)
    bspl_y = splrep(s_cv, list_y, s=0.1)
    # values for the x axis
    s_smooth = np.linspace(0, max(s_cv), point_num)
    # get y values from interpolated curve
    x_smooth = splev(s_smooth, bspl_x)
    y_smooth = splev(s_smooth, bspl_y)
    new_cv = np.array([x_smooth, y_smooth]).T

    delta_new_cv = new_cv[1:, ] - new_cv[:-1, ]
    s_accumulated = np.cumsum(np.linalg.norm(delta_new_cv, axis=1))
    s_accumulated = np.concatenate(([0], s_accumulated), axis=0)
    return new_cv, s_accumulated


def get_central_vertices(cv_type, origin_point=None):
    cv_init = None
    if cv_type == 'lt':  # left turn
        cv_init = np.array([[0, -10], [9, -7.5], [12, -5.2], [13.5, 0], [14, 10], [14, 20], [14, 30]])
    elif cv_type == 'gs':  # go straight
        cv_init = np.array([[100, -2], [10, -2], [0, -2], [-150, -2]])
    elif cv_type == 'lt_nds':  # left turn in NDS
        cv_init = np.array([origin_point, [34.9-13, 16.6-7.8], [45.2-13, 18.8-7.8], [51.6-13, 20.3-7.8], [81.4, 25]])
    elif cv_type == 'gs_nds':  # go straight in NDS
        cv_init = np.array([origin_point, [8.5, 39.2], [9.9, 62.8], [11.3, 74.5]])
    elif cv_type == 'ml':  # main line
        cv_init = np.array([[-50, -2], [-30, -2], [-10, -2], [20, -2], [40, -2]])
    elif cv_type == 'ir':  # in-ramp
        cv_init = np.array([[-50, -10], [-39, -9.6], [-30, -8.5], [-20.4, -6.7],
                            [-10.3, -3.9], [-4, -2.1], [-3, -2.1], [-2, -2.1], [-1, -2.1], [-0, -2.1],
                            [7, -2], [8, -2], [9, -2], [10, -2], [11, -2], [12, -2],
                            [13, -2], [14, -2], [15, -2], [16, -2], [17, -2], [18, -2], [30, -2], [40, -2]])
    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_ployline(cv_init)
    return cv_smoothed, s_accumulated


def bicycle_model(u, init_state, TRACK_LEN, dt):
    if not np.size(u, 0) == TRACK_LEN - 1:
        u = np.array([u[0:TRACK_LEN - 1], u[TRACK_LEN - 1:]]).T
    r_len = 0.8
    f_len = 1
    x, y, vx, vy, h = init_state
    psi = h
    track = [[x, y, vx, vy, h]]
    v_temp = np.sqrt(vx ** 2 + vy ** 2)

    for i in range(len(u)):
        a = u[i, 0]
        delta = u[i, 1]
        beta = math.atan((r_len / (r_len + f_len)) * math.tan(delta))
        x = x + v_temp * np.cos(psi + beta) * dt
        y = y + v_temp * np.sin(psi + beta) * dt
        psi = psi + (v_temp / f_len) * np.sin(beta) * dt
        v_temp = v_temp + a * dt

        vx = v_temp * np.cos(psi)
        vy = v_temp * np.sin(psi)

        track.append([x, y, vx, vy, psi])
    return np.array(track)


def mass_point_model(u, init_state,dt):

    # u = np.array([u[0:TRACK_LEN - 1], u[TRACK_LEN - 1:]]).T
    x, y, vx, vy, h = init_state
    track = [[x, y, vx, vy, h]]
    for i in range(int(np.size(u)/2)):
        vx = vx + u[i] * dt
        vy = vy + u[i+int(np.size(u)/2)] * dt
        x = x + vx * dt
        y = y + vy * dt
        heading = math.atan(vy/vx)
        # heading = 0
        track.append([x, y, vx, vy, heading])
    return np.array(track)


def draw_rectangle(x, y, deg, ax, para_alpha=0.5, para_color='blue'):
    car_len = 1.2
    car_wid = 2
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # r1 = patches.Rectangle((x - car_wid / 2, y - car_len / 2), car_wid, car_len, color="blue", alpha=transp)
    r2 = patches.Rectangle((x - car_wid / 2, y - car_len / 2), car_wid, car_len, color=para_color, alpha=para_alpha)

    t2 = mt.Affine2D().rotate_deg_around(x, y, deg) + ax.transData
    r2.set_transform(t2)

    # ax.add_patch(r1)
    ax.add_patch(r2)


def get_intersection_point(polyline1, polyline2):
    s1 = LineString(polyline1)
    s2 = LineString(polyline2)

    inter_point = s1.intersection(s2)
    return inter_point


def CalcRefLine(cts_points):
    ''' deal with reference path points 2d-array
    to calculate rs/rtheta/rkappa/rdkappa according to cartesian points'''
    rx = cts_points[0]  # the x value
    ry = cts_points[1]  # the y value
    rs = np.zeros_like(rx)
    rtheta = np.zeros_like(rx)
    rkappa = np.zeros_like(rx)  # 曲率
    rdkappa = np.zeros_like(rx)  # 曲率变化率
    for i, x_i in enumerate(rx):
        # y_i = ry[i]
        if i != 0:
            dx = rx[i] - rx[i - 1]
            dy = ry[i] - ry[i - 1]
            rs[i] = rs[i - 1] + math.sqrt(dx ** 2 + dy ** 2)
        if i < len(ry) - 1:
            dx = rx[i + 1] - rx[i]
            dy = ry[i + 1] - ry[i]
            ds = math.sqrt(dx ** 2 + dy ** 2)
            rtheta[i] = math.copysign(math.acos(dx / ds), dy)
    rtheta[-1] = rtheta[-2]
    rkappa[:-1] = np.diff(rtheta) / np.diff(rs)
    rdkappa[:-1] = np.diff(rkappa) / np.diff(rs)
    rkappa[-1] = rkappa[-2]
    rdkappa[-1] = rdkappa[-3]
    rdkappa[-2] = rdkappa[-3]
    rkappa = scipy.signal.savgol_filter(rkappa, 333, 5)
    rdkappa = scipy.signal.savgol_filter(rdkappa, 555, 5)
    # plt.figure(1)
    # plt.subplot(211)
    # plt.plot(rkappa)
    # plt.subplot(212)
    # plt.plot(rdkappa)
    # plt.show()
    path_points = []
    for i in range(len(rx)):
        path_points.append(PathPoint([rx[i], ry[i], rs[i], rtheta[i], rkappa[i], rdkappa[i]]))
    return path_points