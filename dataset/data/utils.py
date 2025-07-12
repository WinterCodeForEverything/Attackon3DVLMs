import numpy as np
import open3d as o3d


def pc_normalize( point_cloud ):
    """ pc: NxC, return NxC """
    xyz = point_cloud[:, :3]
    other_feature = point_cloud[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    point_cloud = np.concatenate((xyz, other_feature), axis=1)
    return point_cloud


def compute_normals(point_cloud):
    """
    Compute normals for a point cloud using open3d.
    Input:
        point_cloud: (N, 6) numpy array, first 3 dims are xyz, last 3 are color
    Output:
        normals: (N, 3) numpy array
    """
    pc_xyz = point_cloud[:, :3]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(pc_xyz)
    
    # Estimate normals
    pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
    
    # Use Open3D's built-in method to orient normals consistently
    pcd.orient_normals_consistent_tangent_plane(k=50)
    
    normals = np.asarray(pcd.normals)
    return normals

def redirect_normals(point_cloud):
    """
    Redirect normals to ensure they point outward from the centroid of the point cloud.

    Args:
        point_cloud (np.ndarray): (N, 6) or (N, 9) array. First 3 dims are xyz, last 3 are normals.

    Returns:
        np.ndarray: point_cloud with normals redirected to point outward.
    """
    coords = point_cloud[:, :3]
    normals = point_cloud[:, -3:]
    centroid = np.mean(coords, axis=0)
    centered_coords = coords - centroid
    dot_products = np.sum(centered_coords * normals, axis=1)
    flip_mask = dot_products < 0
    normals[flip_mask] *= -1
    point_cloud[:, -3:] = normals
    return point_cloud


# def redirect_normals(coords, normals):
#     """
#     Redirect normals to ensure they point outward from the centroid of the point cloud.

#     Args:
#         coords (np.ndarray): (N, 3) array of point coordinates.
#         normals (np.ndarray): (N, 3) array of normal vectors.

#     Returns:
#         np.ndarray: (N, 3) array of normals, redirected to point outward.
#     """    
#     centroid = np.mean(coords, axis=0)
#     centered_coords = coords - centroid
#     # For each normal, if it points inward, flip it
#     dot_products = np.sum(centered_coords * normals, axis=1)
#     flip_mask = dot_products < 0
#     normals[flip_mask] *= -1
#     return normals
