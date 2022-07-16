import numpy as np
from matplotlib import pyplot as plt
from scipy.interpolate import splrep, splev
import math
from shapely.geometry import LineString
import matplotlib.patches as patches
import matplotlib.transforms as mt


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


def savitzky_golay(y, window_size, order, deriv=0, rate=1):
    r"""Smooth (and optionally differentiate) data with a Savitzky-Golay filter.
    The Savitzky-Golay filter removes high frequency noise from data.
    It has the advantage of preserving the original shape and
    features of the signal better than other types of filtering
    approaches, such as moving averages techniques.
    Parameters
    ----------
    y : array_like, shape (N,)
        the values of the time history of the signal.
    window_size : int
        the length of the window. Must be an odd integer number.
    order : int
        the order of the polynomial used in the filtering.
        Must be less then `window_size` - 1.
    deriv: int
        the order of the derivative to compute (default = 0 means only smoothing)
    Returns
    -------
    ys : ndarray, shape (N)
        the smoothed signal (or it's n-th derivative).
    Notes
    -----
    The Savitzky-Golay is a type of low-pass filter, particularly
    suited for smoothing noisy data. The main idea behind this
    approach is to make for each point a least-square fit with a
    polynomial of high order over a odd-sized window centered at
    the point.
    Examples
    --------
    t = np.linspace(-4, 4, 500)
    y = np.exp( -t**2 ) + np.random.normal(0, 0.05, t.shape)
    ysg = savitzky_golay(y, window_size=31, order=4)
    import matplotlib.pyplot as plt
    plt.plot(t, y, label='Noisy signal')
    plt.plot(t, np.exp(-t**2), 'k', lw=1.5, label='Original signal')
    plt.plot(t, ysg, 'r', label='Filtered signal')
    plt.legend()
    plt.show()
    References
    ----------
    .. [1] A. Savitzky, M. J. E. Golay, Smoothing and Differentiation of
       Data by Simplified Least Squares Procedures. Analytical
       Chemistry, 1964, 36 (8), pp 1627-1639.
    .. [2] Numerical Recipes 3rd Edition: The Art of Scientific Computing
       W.H. Press, S.A. Teukolsky, W.T. Vetterling, B.P. Flannery
       Cambridge University Press ISBN-13: 9780521880688
    """
    import numpy as np
    from math import factorial

    try:
        window_size = np.abs(np.int(window_size))
        order = np.abs(np.int(order))
    except ValueError as msg:
        raise ValueError("window_size and order have to be of type int")
    if window_size % 2 != 1 or window_size < 1:
        raise TypeError("window_size size must be a positive odd number")
    if window_size < order + 2:
        raise TypeError("window_size is too small for the polynomials order")
    order_range = range(order + 1)
    half_window = (window_size - 1) // 2
    # precompute coefficients
    b = np.mat([[k ** i for i in order_range] for k in range(-half_window, half_window + 1)])
    m = np.linalg.pinv(b).A[deriv] * rate ** deriv * factorial(deriv)
    # pad the signal at the extremes with
    # values taken from the signal itself
    firstvals = y[0] - np.abs(y[1:half_window + 1][::-1] - y[0])
    lastvals = y[-1] + np.abs(y[-half_window - 1:-1][::-1] - y[-1])
    y = np.concatenate((firstvals, y, lastvals))
    return np.convolve(m[::-1], y, mode='valid')


def get_central_vertices(cv_type, origin_point=None):
    cv_init = None
    if cv_type == 'lt':  # left turn
        cv_init = np.array([[0, -10], [9, -7.5], [12, -5.2], [13.5, 0], [14, 10], [14, 20], [14, 30]])
    elif cv_type == 'gs':  # go straight
        cv_init = np.array([[20, -2], [10, -2], [0, -2], [-150, -2]])
    elif cv_type == 'lt_nds':  # left turn in NDS
        cv_init = np.array([origin_point, [34.9, 16.6], [45.2, 18.8], [51.6, 20.3]])
    elif cv_type == 'gs_nds':  # go straight in NDS
        cv_init = np.array([origin_point, [23, 45.77], [23.55, 49.654], [24.58, 56.65]])
    assert cv_init is not None
    cv_smoothed, s_accumulated = smooth_ployline(cv_init)
    return cv_smoothed, s_accumulated


def kinematic_model(u, init_state, TRACK_LEN, dt):
    if not np.size(u, 0) == TRACK_LEN - 1:
        u = np.array([u[0:TRACK_LEN - 1], u[TRACK_LEN - 1:]]).T
    r_len = 0.8
    f_len = 1
    x, y, vx, vy, h = init_state
    # track = [init_state]
    psi = h
    track = [[x, y, vx, vy, h]]
    v_temp = np.sqrt(vx ** 2 + vy ** 2)

    for i in range(len(u)):
        a = u[i][0]
        delta = u[i][1]
        beta = math.atan((r_len / (r_len + f_len)) * math.tan(delta))
        x = x + v_temp * np.cos(psi + beta) * dt
        y = y + v_temp * np.sin(psi + beta) * dt
        psi = psi + (v_temp / f_len) * np.sin(beta) * dt
        v_temp = v_temp + a * dt

        vx = v_temp * np.cos(psi)
        vy = v_temp * np.sin(psi)

        track.append([x, y, vx, vy, psi])
    return np.array(track)


def get_intersection_point(polyline1, polyline2):
    s1 = LineString(polyline1)
    s2 = LineString(polyline2)

    inter_point = s1.intersection(s2)
    return inter_point


def draw_rectangle(x, y, deg, ax, para_alpha=0.5, para_color='blue'):
    car_len = 1
    car_wid = 2
    # fig = plt.figure()
    # ax = fig.add_subplot(111)

    # r1 = patches.Rectangle((x - car_wid / 2, y - car_len / 2), car_wid, car_len, color="blue", alpha=transp)
    r2 = patches.Rectangle((x - car_wid / 2, y - car_len / 2), car_wid, car_len, color=para_color, alpha=para_alpha)

    t2 = mt.Affine2D().rotate_deg_around(x, y, deg) + ax.transData
    r2.set_transform(t2)

    # ax.add_patch(r1)
    ax.add_patch(r2)

    # plt.grid(True)
    # plt.axis('equal')
    #
    # plt.show()