import os
import numpy as np
import pickle
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



def dat_to_xyz(input_pc_file, output_pc_path, pc_normalize=True):
    
    with open(input_pc_file, 'rb') as f:
        list_of_points, _ = pickle.load(f)
        
    os.makedirs(output_pc_path, exist_ok=True)
    for i, points in tqdm(enumerate(list_of_points), total=len(list_of_points)):
        if pc_normalize:
            points = normalize_pc(points)
        # Ensure output directory exists
        xyz_file = os.path.join(output_pc_path, f'{i}.xyz' )
        np.savetxt(xyz_file, points, fmt='%.6f')
        
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Convert .dat point clouds to .xyz format.')
    parser.add_argument('--input_pc_file', type=str, required=True, help='Input file with .dat point clouds.')
    parser.add_argument('--output_pc_path', type=str, required=True, help='Output directory for .xyz files.')

    args = parser.parse_args()
    dat_to_xyz(args.input_pc_file, args.output_pc_path)
        
