import os
import numpy as np
import pickle
from tqdm import tqdm
import json

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



def dat_to_xyz(input_pc_file, output_pc_path, eval_results_path= None, visualized_sample_num= 100, pc_normalize=False):
    
    with open(input_pc_file, 'rb') as f:
        list_of_points, _ = pickle.load(f)
        
        
    if eval_results_path is not None:
        if 'modelnet' in eval_results_path:
            gpt_eval_results = json.load(open(os.path.join(eval_results_path, 'ModelNet_classification_prompt_0_evaluated_gpt-4o-mini_in_training.json'), 'r'))
        elif 'objaverse' in eval_results_path:
            gpt_eval_results = json.load(open(os.path.join(eval_results_path, 'Objaverse_classification_prompt_0_evaluated_gpt-4o-mini_in_training.json'), 'r'))
        gpt_eval_results = gpt_eval_results['eval_results']
        points_ids = [str(result['object_id']) for result in gpt_eval_results]
        gpt_eval_results_dict = {str(result['object_id']): result for result in gpt_eval_results}
        
        dict_of_points = {str(id): points for id, points in zip(points_ids, list_of_points)}

        pc_eval_results = json.load(open(os.path.join(eval_results_path, 'pc_evaluation_results_recall_in_training.json'), 'r'))
        pc_eval_results = pc_eval_results['results']
            
        CSDs = [ result['curv_std_dist'] for result in pc_eval_results ]
        # Sort pc_eval_results based on CSDs in increasing order
        pc_eval_results = [x for _, x in sorted(zip(CSDs, pc_eval_results), key=lambda pair: pair[0])]
        
            
        visualized_samples = pc_eval_results[:visualized_sample_num]

        visualized_gpt_eval_results = []
        visualized_samples_save_path = os.path.join(eval_results_path, 'visualized_sample')
        os.makedirs(visualized_samples_save_path, exist_ok=True)
        for i, result in tqdm(enumerate(visualized_samples), total=len(visualized_samples)):
            obj_id = result['obj_id']
            points = dict_of_points[obj_id]
            if pc_normalize:
                points = normalize_pc(points)
            xyz_file = os.path.join(visualized_samples_save_path, f'{obj_id}.xyz')
            np.savetxt(xyz_file, points, fmt='%.6f')
            visualized_gpt_eval_results.append(gpt_eval_results_dict[obj_id])
            # print(f"Saved {xyz_file} with CSD: {result['curv_std_dist']:.6f}")
        # Save visualized GPT evaluation results
        with open(os.path.join(eval_results_path, 'visualized_gpt_eval_results.json'), 'w') as f:
            json.dump(visualized_gpt_eval_results, f, indent=4)
    else:  
        os.makedirs(output_pc_path, exist_ok=True)
        list_of_points = list_of_points[:visualized_sample_num]
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
    parser.add_argument('--eval_results_path', type=str, help='Path to the evaluation results.')
    parser.add_argument('--visualized_sample_num', type=int, default=100, help='Number of samples to visualize.')
    parser.add_argument('--output_pc_path', type=str, help='Output directory for .xyz files.')

    args = parser.parse_args()
    dat_to_xyz(args.input_pc_file, args.output_pc_path, args.eval_results_path, args.visualized_sample_num, pc_normalize=True)