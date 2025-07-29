import os
import numpy as np
import json
from tqdm import tqdm
import open3d as o3d

def normalize_pc( point_cloud ):
    """ pc: NxC, return NxC """
    xyz = point_cloud[:, :3]
    other_feature = point_cloud[:, 3:]

    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    m = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    xyz = xyz / m

    point_cloud = np.concatenate((xyz, other_feature), axis=1)
    return point_cloud

def npy_to_xyz(input_pc_path, output_pc_path, eval_results_path=None, visualized_sample_num=None, pc_normalize=True):
    """
    Converts .npy point cloud files in input_pc_path to .xyz format in output_pc_path.
    If eval_results_path is provided, only visualizes the top visualized_sample_num samples
    sorted by 'curv_std_dist' from pc_evaluation_results_recall_in_training.json.
    """
    npy_files = [f for f in os.listdir(input_pc_path) if f.endswith('.npy')]
    npy_files.sort()  # Ensure consistent order
    # print(f"Found {len(npy_files)} .npy files in {input_pc_path}")

    if eval_results_path is not None:

        gpt_eval_results = json.load(open(os.path.join(eval_results_path, 'Objaverse_classification_prompt_0_evaluated_gpt-4o-mini_in_training.json'), 'r'))
        gpt_eval_results = gpt_eval_results['eval_results']
        # points_ids = [str(result['object_id']) for result in gpt_eval_results]
        gpt_eval_results_dict = {str(result['object_id']): result for result in gpt_eval_results}
        
        
        # Load evaluation results
        pc_eval_results = json.load(open(os.path.join(eval_results_path, 'pc_evaluation_results_recall_in_training.json'), 'r'))
        pc_eval_results = pc_eval_results['results']
        CSDs = [result['curv_std_dist'] for result in pc_eval_results]
        # Sort by CSD
        pc_eval_results = [x for _, x in sorted(zip(CSDs, pc_eval_results), key=lambda pair: pair[0])]
        visualized_samples = pc_eval_results[:visualized_sample_num] if visualized_sample_num is not None else pc_eval_results

        visualized_samples_save_path = os.path.join(eval_results_path, 'visualized_sample')
        os.makedirs(visualized_samples_save_path, exist_ok=True)
        visualized_gpt_eval_results = []
        for result in tqdm(visualized_samples, total=len(visualized_samples)):
            obj_id = result['obj_id']
            npy_file = os.path.join(input_pc_path, f"{obj_id}_8192.npy")
            if not os.path.exists(npy_file):
                continue
            points = np.load(npy_file)
            if pc_normalize:
                points = normalize_pc(points)
            xyz_file = os.path.join(visualized_samples_save_path, f"{obj_id}.xyz")
            np.savetxt(xyz_file, points, fmt='%.6f')
            visualized_gpt_eval_results.append(gpt_eval_results_dict[obj_id])
            
        with open(os.path.join(eval_results_path, 'visualized_gpt_eval_results.json'), 'w') as f:
            json.dump(visualized_gpt_eval_results, f, indent=4)
    else:
        os.makedirs(output_pc_path, exist_ok=True)
        npy_files = npy_files[:visualized_sample_num] if visualized_sample_num is not None else npy_files
        for i, fname in tqdm(enumerate(npy_files), total=len(npy_files)):
            npy_file = os.path.join(input_pc_path, fname)
            points = np.load(npy_file)
            if pc_normalize:
                points = normalize_pc(points)
            xyz_file = os.path.join(output_pc_path, os.path.splitext(fname)[0] + '.xyz')
            np.savetxt(xyz_file, points, fmt='%.6f')
            
def npy_to_ply(input_pc_path, output_pc_path, eval_results_path=None, visualized_sample_num=None, pc_normalize=True):
    """
    Converts .npy point cloud files in input_pc_path to .ply format in output_pc_path using Open3D.
    The point cloud is expected to be of shape (N, 9): xyz + rgb + normals.
    """

    def write_ply(filename, points):
        """
        Write points (N,9) to a .ply file using Open3D.
        points: Nx9 array (x, y, z, r, g, b, nx, ny, nz)
        """
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])
        # Normalize RGB to [0,1] for Open3D
        # colors = np.clip(points[:, 3:6], 0, 255) / 255.0
        # colors = points[:, 3:6]
        if 1:
            ori_pc_file = filename.split('/')[-1]
            ori_pc_id = os.path.splitext(ori_pc_file)[0]
            ori_pc_id = ori_pc_id.split('_')[0]
            ori_pc_file_path = os.path.join("/mnt/ssd/liuchao/Attackon3DVLMs/Objaverse_ori_npy", f"{ori_pc_id}_8192.npy")
            ori_pc = np.load(ori_pc_file_path)
            colors = ori_pc[:, 3:6]
        pcd.colors = o3d.utility.Vector3dVector(colors)
        pcd.normals = o3d.utility.Vector3dVector(points[:, 6:9])
        o3d.io.write_point_cloud(filename, pcd)
        # print(f"Saved {filename} with {len(points)} points.")

    npy_files = [f for f in os.listdir(input_pc_path) if f.endswith('.npy')]
    npy_files.sort()
    if eval_results_path is not None:
        gpt_eval_results = json.load(open(os.path.join(eval_results_path, 'Objaverse_classification_prompt_0_evaluated_gpt-4o-mini_in_training.json'), 'r'))
        gpt_eval_results = gpt_eval_results['eval_results']
        gpt_eval_results_dict = {str(result['object_id']): result for result in gpt_eval_results}
        pc_eval_results = json.load(open(os.path.join(eval_results_path, 'pc_evaluation_results_recall_in_training.json'), 'r'))
        pc_eval_results = pc_eval_results['results']
        CSDs = [result['curv_std_dist'] for result in pc_eval_results]
        pc_eval_results = [x for _, x in sorted(zip(CSDs, pc_eval_results), key=lambda pair: pair[0])]
        visualized_samples = pc_eval_results[:visualized_sample_num] if visualized_sample_num is not None else pc_eval_results

        visualized_samples_save_path = os.path.join(eval_results_path, 'visualized_sample_ply')
        os.makedirs(visualized_samples_save_path, exist_ok=True)
        visualized_gpt_eval_results = []
        for result in tqdm(visualized_samples, total=len(visualized_samples)):
            obj_id = result['obj_id']
            npy_file = os.path.join(input_pc_path, f"{obj_id}_8192.npy")
            if not os.path.exists(npy_file):
                continue
            points = np.load(npy_file)
            if pc_normalize:
                points = normalize_pc(points)
            ply_file = os.path.join(visualized_samples_save_path, f"{obj_id}.ply")
            write_ply(ply_file, points)
            visualized_gpt_eval_results.append(gpt_eval_results_dict[obj_id])
        with open(os.path.join(eval_results_path, 'visualized_gpt_eval_results_ply.json'), 'w') as f:
            json.dump(visualized_gpt_eval_results, f, indent=4)
    else:
        os.makedirs(output_pc_path, exist_ok=True)
        npy_files = npy_files[:visualized_sample_num] if visualized_sample_num is not None else npy_files
        for i, fname in tqdm(enumerate(npy_files), total=len(npy_files)):
            npy_file = os.path.join(input_pc_path, fname)
            points = np.load(npy_file)
            if pc_normalize:
                points = normalize_pc(points)
            ply_file = os.path.join(output_pc_path, os.path.splitext(fname)[0] + '.ply')
            write_ply(ply_file, points)
            
# def npy_to_xyz(input_pc_path, output_pc_path, pc_normalize=True):
#     """
#     Converts all .npy point cloud files in input_pc_path to .xyz (txt) format in output_pc_path.

#     Args:
#         input_pc_path (str): Directory containing .npy point cloud files.
#         output_pc_path (str): Directory to save .xyz files.
#     """
#     os.makedirs(output_pc_path, exist_ok=True)
#     for fname in tqdm(os.listdir(input_pc_path)):
#         if fname.endswith('.npy'):
#             npy_file = os.path.join(input_pc_path, fname)
#             xyz_file = os.path.join(output_pc_path, os.path.splitext(fname)[0] + '.xyz')
#             points = np.load(npy_file)  # shape: (N, 3) or (N, >=3)
#             # Only take first 3 columns if more exist
#             # points_xyz = points[:, :3]
#             if pc_normalize:
#                 points = normalize_pc(points)
#             np.savetxt(xyz_file, points, fmt='%.6f')
            
if __name__ == "__main__":
    import argparse


    parser = argparse.ArgumentParser(description='Convert .dat point clouds to other format.')
    parser.add_argument('--input_pc_file', type=str, required=True, help='Input file with .dat point clouds.')
    parser.add_argument('--eval_results_path', type=str, help='Path to the evaluation results.')
    parser.add_argument('--visualized_sample_num', type=int, default=None, help='Number of samples to visualize.')
    parser.add_argument('--output_pc_path',  help='Output directory for .xyz files.')
    parser.add_argument('--save_format', type=str, default='xyz', choices=['xyz', 'ply'], help='Format to save point clouds (xyz or ply).')

    args = parser.parse_args()
    if args.save_format == 'xyz':
        npy_to_xyz(args.input_pc_file, args.output_pc_path, args.eval_results_path, args.visualized_sample_num)
    elif args.save_format == 'ply':
        npy_to_ply(args.input_pc_file, args.output_pc_path, args.eval_results_path, args.visualized_sample_num)
    else:
        raise ValueError("Unsupported save format. Use 'xyz' or 'ply'.")
    # npy_to_xyz(args.input_pc_file, args.output_pc_path, args.eval_results_path, args.visualized_sample_num)