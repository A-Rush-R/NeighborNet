import os
import json
import pickle as pkl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

# ==========================
# 📌 Utility Functions
# ==========================
def read_pkl(filepath):
    """Reads a pickle file and returns its content."""
    with open(filepath, 'rb') as file:
        return pkl.load(file)

def write_pkl(filepath, data):
    """Writes data to a pickle file."""
    with open(filepath, 'wb') as file:
        pkl.dump(data, file)

def read_label(filepath):
    """Reads a label file and returns a dictionary of shot labels."""
    label_dict = {}
    with open(filepath, 'r') as file:
        for line in file:
            shot_id, label = line.strip().split()
            label_dict[int(shot_id)] = int(label)
    return label_dict

def get_movie_names(label_dir):
    """Returns a list of movie names from label files."""
    return [file.split('.')[0] for file in os.listdir(label_dir)]

# ==========================
# 📌 Graph Construction
# ==========================
def build_graph(links):
    """Builds a graph adjacency matrix from given shot links."""
    return np.array(links)  # Placeholder; Replace with actual graph logic

def gen_reason_link(links):
    """Generates reasoning-based links for higher-order shot relationships."""
    return links  # Placeholder; Modify as needed

def index_matrix(links, top_k):
    """Generates index matrices for neighbor selection."""
    return np.argsort(links, axis=1)[:, :top_k]



# ==========================
# 📌 MovieNet Dataset Class
# ==========================
class MovieNetDataset(Dataset):
    def __init__(self, sample_list, mode='train', topk=1):
        """Initializes the dataset with sample file paths."""
        self.sample_list = sample_list
        self.mode = mode
        self.topk = topk

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):
        data = read_pkl(self.sample_list[idx])
        sample, label, alink = data['data'], data['label'], data['hop']

        # Handle invalid labels
        label = 1 if label not in [0, 1] else label
        sample = torch.tensor(sample, dtype=torch.float)
        
        # Construct Graphs
        hop = [build_graph(alink)[None], build_graph(gen_reason_link(alink))[None]]
        inxs = [index_matrix(alink, self.topk)[None]]
        
        # Convert to tensors
        hop, inxs = map(lambda x: torch.tensor(np.concatenate(x, axis=0), dtype=torch.float), [hop, inxs])
        label = torch.tensor(label, dtype=torch.float)

        return (sample, hop, inxs, label) if self.mode == 'train' else (self.sample_list[idx], sample, hop, inxs, label)

# ==========================
# 📌 Data Preprocessing
# ==========================
def sample_context(data, labels, center_id, seg_size=20, dim=2048):
    """Extracts a context window centered around a shot."""
    max_id = len(data.keys())
    ctx_id = np.clip(np.arange(center_id - seg_size // 2 + 1, center_id + seg_size // 2 + 1), 0, max_id - 1)
    ctx = np.zeros((seg_size, dim))
    ctx_labels = [labels.get(shot_id, 0) for shot_id in ctx_id]
    
    for i, shot_id in enumerate(ctx_id):
        ctx[i] = data[f'{shot_id:04d}'][None]
    
    return ctx, np.array(ctx_labels)[seg_size // 2 - 1]

def generate_dataset(feature_path, label_path, graph_path, save_path, seg_size=20, dim=2048):
    """Processes and saves the dataset."""
    os.makedirs(save_path, exist_ok=True)
    features = read_pkl(feature_path)
    movie_names = get_movie_names(label_path)

    for name in tqdm(movie_names, desc="Processing Movies"):
        if name not in features:
            continue

        feat_m, label_m = features[name], read_label(os.path.join(label_path, f"{name}.txt"))
        for shot_id in range(len(label_m)):
            ctx, c_label = sample_context(feat_m, label_m, shot_id, seg_size, dim)
            graph_file = os.path.join(graph_path, f"{name}_shot{shot_id}.pkl")
            if not os.path.exists(graph_file):
                continue
            hop_data = read_pkl(graph_file)

# If the file contains a NumPy array, use it directly
            if isinstance(hop_data, np.ndarray):
                hop_link = hop_data
            else:
                hop_link = hop_data.get('hop', None)  # If it's a dictionary, get 'hop'

            if hop_link is None:
                continue
            
            write_pkl(os.path.join(save_path, f"{name}_shot{shot_id}.pkl"), {'data': ctx, 'label': c_label, 'hop': hop_link})
    
    print("Dataset generation completed!")

# ==========================
# 📌 Data Loading Functions
# ==========================
def load_data(data_path, split_path, batch_size, mode='train', topk=1):
    with open(split_path, 'r') as f:
        split_data = json.load(f)
    
    sample_list = [os.path.join(data_path, f) for f in os.listdir(data_path) if f.split('_')[0] in split_data[mode]]
    dataset = MovieNetDataset(sample_list, mode=mode, topk=topk)
    return DataLoader(dataset, batch_size=batch_size, shuffle=(mode == 'train'), drop_last=(mode == 'train'), num_workers=4)

# ==========================
# 📌 Example Usage
# ==========================
if __name__ == "__main__":
    # Paths
    feature_path = "/data/OpenDataLab___MovieNet/raw/ImageNet_shot.pkl"
    label_path = "/data/manan/label318"
    graph_path = "/data/manan/gph_folder/gph_file_k_1"
    save_path = "/data/OpenDataLab___MovieNet/raw/gendataset/gendatasetk1"
    
    # Generate dataset
    generate_dataset(feature_path, label_path, graph_path, save_path)
    
    # Load dataset
    batch_size = 256
    split_path = "/data/OpenDataLab___MovieNet/raw/movie1K.split.v1.json"
    train_loader = load_data(save_path, split_path, batch_size, mode='train')
    
    # Iterate over DataLoader
    for data in tqdm(train_loader):
        samples, hop, inxs, labels = data  # Extract batch datals
        