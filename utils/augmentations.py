import numpy as np


def normalize_data(x):
    """ Normalize the batch data, use coordinates of the block centered at origin,
        Input:
            NxC array
        Output:
            NxC array
    """
    N, C = x.shape
    centroid = np.mean(x, axis=0)
    pc = x - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def shuffle_points(x):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Input:
            NxC array
        Output:
            NxC array
    """
    idx = np.arange(x.shape[1])
    np.random.shuffle(idx)
    return x[idx, :]


def rotate_point_cloud(x):
    """ Randomly rotate the point clouds to augment the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    return np.dot(x.reshape((-1, 3)), rotation_matrix)


def rotate_point_clouds_z(x):
    """ Randomly rotate the point clouds to augment the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original batch of point clouds
        Return:
          Nx3 array, rotated batch of point clouds
    """
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, sinval, 0],
                                [-sinval, cosval, 0],
                                [0, 0, 1]])
    return np.dot(x.reshape((-1, 3)), rotation_matrix)


def rotate_point_clouds_with_normal(x):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            x: N,6, first three channels are XYZ, last 3 all normal
        Output:
            N,6, rotated XYZ, normal point cloud
    '''
    rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    new_pt = np.empty_like(x)
    new_pt[:, 0:3] = np.dot(x[:, 0:3].reshape((-1, 3)), rotation_matrix)
    new_pt[:, 3:6] = np.dot(x[:, 3:6].reshape((-1, 3)), rotation_matrix)
    return new_pt


def rotate_perturbation_point_clouds_with_normal(x, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx6 array, original point cloud and point normals
        Return:
          Nx3 array, rotated point cloud
    """
    rotated_data = np.empty_like(x)
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    rotated_data[:, 0:3] = np.dot(x[:, :3].reshape((-1, 3)), R)
    rotated_data[:, 3:] = np.dot(x[:, 3:].reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(x, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    # rotation_angle = np.random.uniform() * 2 * np.pi
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    return np.dot(x, rotation_matrix)


def rotate_point_cloud_by_angle_with_normal(x, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          Nx6 array, original point clouds with normal scalar, angle of rotation
        Return:
          Nx6 array, rotated point clouds iwth normal
    """
    rotated_data = np.empty_like(x)
    cosval = np.cos(rotation_angle)
    sinval = np.sin(rotation_angle)
    rotation_matrix = np.array([[cosval, 0, sinval],
                                [0, 1, 0],
                                [-sinval, 0, cosval]])
    rotated_data[:, 0:3] = np.dot(x[:, :3], rotation_matrix)
    rotated_data[:, 3:] = np.dot(x[:, 3:], rotation_matrix)
    return rotated_data


def rotate_perturbation_point_cloud(x, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, rotated point clouds
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    return np.dot(x, R)


def jitter_point_cloud(x, sigma=0.01, clip=0.05):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, jittered point clouds
    """
    N, C = x.shape
    assert (clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(N, C), -1 * clip, clip)
    jittered_data += x
    return jittered_data


def shift_point_cloud(x, shift_range=0.1):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original point clouds
        Return:
          Nx3 array, shifted point clouds
    """
    N, C = x.shape
    shifted_x = x + np.random.uniform(-shift_range, shift_range, (1, 3))
    return shifted_x


def random_scale_point_cloud(x, scale_low=0.8, scale_high=1.25):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original point clouds
        Return:
            Nx3 array, scaled point clouds
    """
    N, C = x.shape
    scales = np.random.uniform(scale_low, scale_high, 1)
    scaled_x = x * scales
    return scaled_x


def random_point_dropout(x, max_dropout_ratio=0.875):
    ''' batch_pc: Nx3 '''
    dropout_ratio = np.random.random() * max_dropout_ratio  # 0~0.875
    drop_idx = np.where(np.random.random((x.shape[0])) <= dropout_ratio)[0]
    masked_x = x.copy()
    if len(drop_idx) > 0:
        masked_x[drop_idx, :] = 0  # set to 0
    return masked_x
