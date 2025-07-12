import os
import numpy as np
import json
from torch.utils.data import Dataset
from .utils import *


class ObjaverseDataset(Dataset):
    """Dataset utilities for objaverse."""
    def __init__(self,
                 data_path=None,
                 anno_path=None,
                 normalized_pc=True,
                 has_color=True,
                 has_normal=False,
                 redirect_normals=True,
                 use_color=True,
                 use_normal=True,
                 pts_processor=None,
                 text_processor=None,
                #  tokenizer=None,
                 pointnum=8192,
                #  split='train',
                #  conversation_types=None, # * default is simple_des, used for stage1 pre-train

                #  data_args=None
                ):

        """
        """
        super(ObjaverseDataset, self).__init__()

        """Initialize dataset with object point clouds and text"""
        self.data_path = data_path
        self.anno_path = anno_path
        
        self.normalized_pc = normalized_pc
        self.has_color = has_color
        self.has_normal = has_normal
        self.redirect_normals = redirect_normals
        self.use_color = use_color
        self.use_normal = use_normal

        self.pts_processor = pts_processor
        self.text_processor = text_processor

        self.pointnum = pointnum

        # Load the data list from JSON
        print(f"Loading anno file from {anno_path}.")
        with open(anno_path, "r") as json_file:
            self.list_data_dict = json.load(json_file)

    # def _compute_normals(self, point_cloud):
    #     """
    #     Compute normals for a point cloud using open3d.
    #     Input:
    #         point_cloud: (N, 6) numpy array, first 3 dims are xyz, last 3 are color
    #     Output:
    #         normals: (N, 3) numpy array
    #     """
    #     pc_xyz = point_cloud[:, :3]
    #     pcd = o3d.geometry.PointCloud()
    #     pcd.points = o3d.utility.Vector3dVector(pc_xyz)
        
    #     # Estimate normals
    #     pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamKNN(knn=30))
        
    #     # Use Open3D's built-in method to orient normals consistently
    #     pcd.orient_normals_consistent_tangent_plane(k=50)
        
    #     normals = np.asarray(pcd.normals)
    #     return normals
    

    # def _redirect_normals(self, coords, normals):
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

    def _points_preprocess(self, point_cloud):
        """Preprocess the point cloud data."""
        
        if self.normalized_pc:
            point_cloud = pc_normalize(point_cloud)
            
        # point color
        if  self.use_color and not self.has_color:
            white_color = np.array([1.0, 1.0, 1.0], dtype=np.float32)   # white rgb
            white_colors = np.tile(white_color, (point_cloud.shape[0], 1))
            if point_cloud.shape[-1] == 3:
                point_cloud = np.concatenate((point_cloud, white_colors), axis=1)
            elif point_cloud.shape[-1] > 3:
                point_cloud = np.concatenate((point_cloud[:, :3], white_colors, point_cloud[:, 3:]), axis=1)
        
        has_normal = self.has_normal
        if self.use_normal and not self.has_normal:
            # compute normals if not provided
            normals = compute_normals(point_cloud)
            point_cloud = np.concatenate((point_cloud, normals), axis=1)
            has_normal = True
            
        if has_normal and self.redirect_normals:
            # print("Redirecting normals to point outward.")
            point_cloud = redirect_normals(point_cloud)
        # point_cloud = np.concatenate((point_cloud, normals), axis=1)
            
        # point transform
        if self.pts_processor is not None:
            point_cloud[:, :6] = self.pts_processor(point_cloud[:, :6])
        
        return point_cloud
    

    def _load_point_cloud(self, object_id, type='objaverse'):
        if type == 'objaverse':
            return self._load_objaverse_point_cloud(object_id) 

    
    def _load_objaverse_point_cloud(self, object_id):
        filename = f"{object_id}_{self.pointnum}.npy"
        point_cloud = np.load(os.path.join(self.data_path, filename))

        return point_cloud
    
    def __getitem__(self, index):
        sources = self.list_data_dict[index]
        # if isinstance(index, int):
        #     sources = [sources]
        # assert len(sources) == 1, "sources should be a list"
        
        object_id = self.list_data_dict[index]['object_id']

        # Point cloud representation
        point_cloud = self._load_point_cloud(object_id) # * N, C
        
        point_cloud = self._points_preprocess(point_cloud)
        

        # point transform
        if self.pts_processor is not None:
            point_cloud[:, :6] = self.pts_processor(point_cloud[:, :6])
        
        label = sources['conversations'][1]['value']
        
        if self.text_processor is not None: 
            label = self.text_processor(label)

        return {
            "object_id": object_id,
            "point_cloud": point_cloud,
            "label": label
        }

    def __len__(self):
        """Return number of utterances."""
        return len(self.list_data_dict)
 