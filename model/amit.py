import os
import pickle
from tqdm import tqdm

def gen_dataSet(ft_path, lb_path, gph_path, seg_sz=20, dim=2048, save_path=None):
    """
    Generate a dataset by sampling context features, labels, and graph hops for movie shots.

    Args:
        ft_path (str): Path to the pickle file containing features.
        lb_path (str): Path to the directory containing label files.
        gph_path (str): Path to the directory containing graph hop files.
        seg_sz (int, optional): Segment size for context sampling. Defaults to 20.
        dim (int, optional): Dimensionality of features. Defaults to 2048.
        save_path (str, optional): Directory to save the processed dataset. If None, defaults to the current directory.

    Returns:
        int: Returns 1 on successful completion.
    """
    # Ensure save_path is defined
    if save_path is None:
        save_path = "./processed_data"
    os.makedirs(save_path, exist_ok=True)
    
    # Read features and movie names
    feats = read_pkl(ft_path)
    if feats is None:
        print("Feature file could not be read. Exiting...")
        return -1
    
    mnames = gen_labelName(lb_path)
    if not mnames:
        print("No movie names found. Exiting...")
        return -1
    
    for name in tqdm(mnames, desc="Processing Movies"):
        if name not in feats:
            print(f"Warning: {name} not found in features. Skipping...")
            continue
        
        feat_m = feats[name]
        label_m = read_label(os.path.join(lb_path, f"{name}.txt"))
        n_shot = len(label_m)
        
        for c_id in range(n_shot):
            # Sample context and labels
            ctx, c_label = sampleCtx(feat_m, label_m, c_id, seg_sz, dim=dim)
            if not ctx or c_label is None:
                print(f"Skipping invalid data for {name}, shot {c_id}...")
                continue
            
            # Save path for the current shot
            shot_path = os.path.join(save_path, f"{name}_shot{c_id}.pkl")
            
            # Read graph hop information
            graph_file = os.path.join(gph_path, f"{name}_shot{c_id}.pkl")
            if not os.path.exists(graph_file):
                print(f"Warning: Graph file {graph_file} not found. Skipping...")
                continue
            
            hop_link = read_pkl(graph_file).get('hop', None)
            if hop_link is None:
                print(f"Warning: 'hop' key missing in {graph_file}. Skipping...")
                continue
            
            # Create sample dictionary and save
            sample = {'data': ctx, 'label': c_label, 'hop': hop_link}
            write_pkl(shot_path, sample)
    
    print("Dataset generation completed successfully!")
    return 1

# Example Usage
if __name__ == "__main__":
    ft_path = "/data/OpenDataLab___MovieNet/raw/ImageNet_shot.pkl"
    lb_path = "/data/manan/label318"
    gph_path = "/data/manan/gph_file"
    save_path = "/data/OpenDataLab___MovieNet/raw/dataset"
    
    gen_dataSet(ft_path, lb_path, gph_path, seg_sz=20, dim=2048, save_path=save_path)

