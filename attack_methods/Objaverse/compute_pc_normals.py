import os
import json
import numpy as np
import open3d as o3d
from tqdm import tqdm


def compute_pc_normals(point_cloud):
    """
    Compute normals for a point cloud and ensure consistent orientation.
    Args:
        point_cloud (np.ndarray): Input point cloud of shape (N, 3).
    Returns:
        np.ndarray: Normals of shape (N, 3).
    """
    # Create Open3D point cloud object
    o3d_pc = o3d.geometry.PointCloud()
    o3d_pc.points = o3d.utility.Vector3dVector(point_cloud)

    # Estimate normals
    o3d_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

    # # Use Open3D's built-in method to orient normals consistently
    o3d_pc.orient_normals_consistent_tangent_plane(k=50)


    # Return normals as a numpy array
    normals = np.asarray(o3d_pc.normals)
    return normals

# def compute_pc_normals( point_cloud):
#     """
#     Compute normals for a point cloud.
#     Args:
#         point_cloud (np.ndarray): Input point cloud of shape (N, 3).
#     Returns:
#         np.ndarray: Normals of shape (N, 3).
#     """
#     # Create Open3D point cloud object
#     o3d_pc = o3d.geometry.PointCloud()
#     o3d_pc.points = o3d.utility.Vector3dVector(point_cloud)

#     # Estimate normals
#     o3d_pc.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))

#     # Return normals as a numpy array
#     normals = np.asarray(o3d_pc.normals)
#     return normals

def save_point_cloud(input_path, object_id, output_path, vis_path = None):
    filename = f"{object_id}_8192.npy"
    output_file = os.path.join(output_path, filename)
    if os.path.exists(output_file):
        print(f"File {output_file} already exists, skipping...")
        return
    point_cloud = np.load(os.path.join(input_path, filename))

    normals = compute_pc_normals(point_cloud[:, :3])
    point_cloud = np.concatenate((point_cloud, normals), axis=1)
    #output_filename = f"{object_id}_8192.npy"
    np.save( output_file, point_cloud )
    #print(f"Saved point cloud with normals to {os.path.join(output_path, filename)}")
    if vis_path is not None:
        np.savetxt(os.path.join(vis_path, filename.replace('.npy', '.txt')), point_cloud)
    
if __name__ == "__main__":
    input_pc_path = '/mnt/ssd/liuchao/PointLLM/Objaverse_npy_cp'
    ori_cap = "/home/liuchao/PointLLM2/objaverse_result/objaverse_eval/chosen_gpt_eval_results.json"
    #'/mnt/ssd/liuchao/PointLLM/PointLLM_brief_description_adv_200.json'
    output_pc_path = '/mnt/ssd/liuchao/PointLLM/Objaverse_npy_cp_normals'
    visal_path = '/mnt/ssd/liuchao/PointLLM/Objaverse_npy_cp_with_normals_visual'
    
    if not os.path.exists(output_pc_path):
        os.makedirs(output_pc_path)
        
    if not os.path.exists(visal_path):
        os.makedirs(visal_path)
    
    with open(ori_cap, "r") as json_file:
        list_ori_cap_dict = json.load(json_file)
    object_ids = [item['object_id'] for item in list_ori_cap_dict]

    for object_id in tqdm(object_ids):
        save_point_cloud(input_pc_path, object_id, output_pc_path, visal_path)
        
   
