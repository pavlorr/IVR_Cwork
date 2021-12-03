import math
import numpy as np

LINK_1_LENGTH = 4.0
LINK_1_PIXEL_LENGTH = 105
LINK_2_LENGTH = 0.0
LINK_3_LENGTH = 3.2
LINK_3_PIXEL_LENGTH = 80
LINK_4_LENGTH = 2.8
LINK_4_PIXEL_LENGTH = 76


def forward_kinematics(z, x, y) -> np.array:
    r_array = np.array([0, 0, 0])
    r_array[0] = LINK_4_PIXEL_LENGTH * np.cos(z) * np.sin(y) + \
                 np.sin(x) * np.sin(z) * (LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH * np.cos(y))

    r_array[1] = LINK_4_PIXEL_LENGTH * np.sin(y) * np.sin(z) - \
                 np.cos(z) * np.sin(x) * (LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH * np.cos(y))

    r_array[2] = LINK_1_PIXEL_LENGTH + np.cos(x) * (LINK_3_PIXEL_LENGTH + LINK_4_PIXEL_LENGTH * np.cos(y))
    return r_array
