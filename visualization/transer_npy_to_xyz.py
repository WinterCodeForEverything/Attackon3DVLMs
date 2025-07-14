import os
import numpy as np
from tqdm import tqdm

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



def npy_to_xyz(input_pc_path, output_pc_path, pc_normalize=True):
    """
    Converts all .npy point cloud files in input_pc_path to .xyz (txt) format in output_pc_path.

    Args:
        input_pc_path (str): Directory containing .npy point cloud files.
        output_pc_path (str): Directory to save .xyz files.
    """
    os.makedirs(output_pc_path, exist_ok=True)
    for fname in tqdm(os.listdir(input_pc_path)):
        if fname.endswith('.npy'):
            npy_file = os.path.join(input_pc_path, fname)
            xyz_file = os.path.join(output_pc_path, os.path.splitext(fname)[0] + '.xyz')
            points = np.load(npy_file)  # shape: (N, 3) or (N, >=3)
            # Only take first 3 columns if more exist
            # points_xyz = points[:, :3]
            if pc_normalize:
                points = normalize_pc(points)
            np.savetxt(xyz_file, points, fmt='%.6f')
            
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert .npy point clouds to .xyz format.')
    parser.add_argument('--input_pc_path', type=str, required=True, help='Input directory with .npy files.')
    parser.add_argument('--output_pc_path', type=str, required=True, help='Output directory for .xyz files.')

    args = parser.parse_args()
    npy_to_xyz(args.input_pc_path, args.output_pc_path)