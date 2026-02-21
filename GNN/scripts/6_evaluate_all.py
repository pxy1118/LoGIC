import argparse
import os
import sys
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# Add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from models.cnn_backbone import CNN3DBackbone
from models.pore_gnn import PoreGNN
from data.dataset import VoxelDataset
from data.graph_builder import build_knn_edges
from torch_geometric.data import Data, Batch

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate trained models on the entire dataset')
    parser.add_argument('--dataset_dir', type=str, default=os.path.join(ROOT_DIR, 'dataset'), help='Path to dataset directory')
    parser.add_argument('--cnn_checkpoint', type=str, default=os.path.join(ROOT_DIR, 'checkpoints', 'cnn_backbone_best.pth'))
    parser.add_argument('--gnn_checkpoint', type=str, default=os.path.join(ROOT_DIR, 'checkpoints', 'pore_gnn_graph_best.pth'))
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for evaluation')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default=os.path.join(ROOT_DIR, 'evaluation'), help='Directory to save results')
    parser.add_argument('--k', type=int, default=5, help='KNN parameter for graph building')
    return parser.parse_args()

def load_models(args, num_outputs, device):
    """Load both CNN and GNN models"""
    print(f"Loading CNN from {args.cnn_checkpoint}")
    cnn_ckpt = torch.load(args.cnn_checkpoint, map_location=device, weights_only=False)
    cnn_config = cnn_ckpt.get('config', {}).get('cnn', {})
    
    cnn = CNN3DBackbone(
        in_channels=1,
        filters=tuple(cnn_config.get('filters', [8, 16, 32])),
        fc_dim=cnn_config.get('fc_dim', 128),
        adaptive_pool_size=cnn_config.get('adaptive_pool_size', 6),
        num_outputs=num_outputs
    ).to(device)
    cnn.load_state_dict(cnn_ckpt['model_state_dict'])
    cnn.eval()

    gnn = None
    if os.path.exists(args.gnn_checkpoint):
        print(f"Loading GNN from {args.gnn_checkpoint}")
        gnn_ckpt = torch.load(args.gnn_checkpoint, map_location=device, weights_only=False)
        gnn_config = gnn_ckpt.get('config', {}).get('gnn', {})
        
        gnn = PoreGNN(
            mode='graph',
            hidden_channels=gnn_config.get('hidden_dim', 32),
            num_layers=gnn_config.get('num_layers', 3),
            conv_type=gnn_ckpt.get('conv_type', 'ChebConv'),
            cheb_k=gnn_config.get('cheb_k', 2),
            pooling=gnn_config.get('pooling', 'sum'),
            num_outputs=num_outputs
        ).to(device)
        gnn.load_state_dict(gnn_ckpt['model_state_dict'])
        gnn.eval()
    else:
        print(f"Warning: GNN checkpoint not found at {args.gnn_checkpoint}")

    return cnn, gnn

def denormalize(data, mean, std):
    """Inverse transform normalized data (z-score)"""
    return data * std + mean

def plot_parity(y_true, y_pred, title, unit, filename):
    """Draw a parity plot (True vs Predicted)"""
    plt.figure(figsize=(7, 6))
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    
    plt.scatter(y_true, y_pred, alpha=0.5, s=15, edgecolors='k', linewidth=0.3)
    
    # Diagonal line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    plt.plot([min_val - margin, max_val + margin], [min_val - margin, max_val + margin], 'r--', alpha=0.8)
    
    plt.xlabel(f'True {unit}')
    plt.ylabel(f'Predicted {unit}')
    plt.title(f'{title}\nMAE={mae:.3f}, R²={r2:.3f}, RMSE={rmse:.3f}')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Load Dataset
    print(f"Loading full dataset from {args.dataset_dir}")
    # Normalize labels=True so we feed normalized targets to the model directly,
    # but we will store dataset.label_mean/std to denormalize for final metrics.
    dataset = VoxelDataset(
        dataset_dir=args.dataset_dir,
        label_type='permeability', # To get both K and dP
        indices=None, # Use all data
        transform=None, # No augmentation for evaluation
        normalize_label=True
    )
    
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    
    stats = dataset.get_statistics()
    num_outputs = stats['num_outputs']
    label_mean = np.array(stats['label_mean'])
    label_std = np.array(stats['label_std'])
    print(f"Dataset stats: {len(dataset)} samples")
    print(f"Normalization - Mean: {label_mean}, Std: {label_std}")

    # 2. Load Models
    cnn, gnn = load_models(args, num_outputs, args.device)

    # 3. Evaluation Loop
    results = {
        'sample_id': [],
        'true_logK': [],
        'pred_cnn_logK': [],
        'pred_gnn_logK': [],
        'true_dP': [],
        'pred_cnn_dP': [],
        'pred_gnn_dP': []
    }

    print("Starting evaluation...")
    with torch.no_grad():
        for batch in tqdm(dataloader):
            voxels = batch['voxel'].to(args.device)  # (B, 1, 120, 120, 120)
            labels = batch['label'].cpu().numpy()    # (B, 2) normalized
            sample_ids = batch['sample_id'].cpu().numpy()
            
            # --- CNN Prediction ---
            # Forward pass to get final output AND features
            features = cnn.forward_features(voxels)
            cnn_pred_norm = features['output'].cpu().numpy() # (B, 2)
            
            # --- GNN Prediction ---
            gnn_pred_norm = np.zeros_like(cnn_pred_norm)
            if gnn is not None:
                layer4 = features['layer4'].cpu().numpy() # (B, 32, 6, 6, 6)
                batch_graphs = []
                
                # Build graphs for each sample in batch
                for i in range(len(layer4)):
                    feat = layer4[i] # (32, 6, 6, 6)
                    # Reshape to (216, 32)
                    node_features = feat.transpose(1, 2, 3, 0).reshape(-1, feat.shape[0])
                    edge_index, edge_weight = build_knn_edges(node_features, k=args.k)
                    
                    batch_graphs.append(Data(
                        x=torch.tensor(node_features, dtype=torch.float32),
                        edge_index=torch.tensor(edge_index, dtype=torch.long),
                        edge_attr=torch.tensor(edge_weight, dtype=torch.float32).view(-1, 1),
                        batch=torch.full((node_features.shape[0],), i, dtype=torch.long)
                    ))
                
                # Batch graphs and predict
                gnn_batch = Batch.from_data_list(batch_graphs).to(args.device)
                gnn_out = gnn(gnn_batch)
                gnn_pred_norm = gnn_out.cpu().numpy()

            # --- De-normalization & Storage ---
            
            # True Labels
            true_denorm = denormalize(labels, label_mean, label_std)
            
            # CNN Preds
            cnn_pred_denorm = denormalize(cnn_pred_norm, label_mean, label_std)
            
            # GNN Preds (if available)
            gnn_pred_denorm = denormalize(gnn_pred_norm, label_mean, label_std) if gnn else cnn_pred_denorm

            # Store per-sample results
            for i in range(len(sample_ids)):
                results['sample_id'].append(sample_ids[i])
                
                results['true_logK'].append(true_denorm[i, 0])
                results['true_dP'].append(true_denorm[i, 1])
                
                results['pred_cnn_logK'].append(cnn_pred_denorm[i, 0])
                results['pred_cnn_dP'].append(cnn_pred_denorm[i, 1])
                
                results['pred_gnn_logK'].append(gnn_pred_denorm[i, 0])
                results['pred_gnn_dP'].append(gnn_pred_denorm[i, 1])

    # 4. Save & Report
    df_res = pd.DataFrame(results)
    
    # Calculate Metrics
    print("\n" + "="*50)
    print("FINAL EVALUATION REPORT")
    print("="*50)
    
    metrics = []

    # Helper for printing
    def print_metric(name, y_true, y_pred):
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        print(f"{name:<15} | MAE: {mae:.4f} | R²: {r2:.4f}")
        return mae, r2

    print("--- Permeability (log K) ---")
    print_metric("CNN Backbone", df_res['true_logK'], df_res['pred_cnn_logK'])
    if gnn:
        print_metric("Pore-GNN", df_res['true_logK'], df_res['pred_gnn_logK'])

    print("\n--- Pressure Drop (dP) ---")
    print_metric("CNN Backbone", df_res['true_dP'], df_res['pred_cnn_dP'])
    if gnn:
        print_metric("Pore-GNN", df_res['true_dP'], df_res['pred_gnn_dP'])
    
    # Save CSV
    csv_path = os.path.join(args.output_dir, 'evaluation_results.csv')
    df_res.to_csv(csv_path, index=False)
    print(f"\nSaved detailed results to {csv_path}")

    # Plotting
    print("Generating plots...")
    
    # Plot logK
    plot_parity(df_res['true_logK'], df_res['pred_cnn_logK'], "CNN Backbone: Permeability", "log(K)", 
                os.path.join(args.output_dir, 'parity_cnn_logK.png'))
    if gnn:
        plot_parity(df_res['true_logK'], df_res['pred_gnn_logK'], "Pore-GNN: Permeability", "log(K)", 
                    os.path.join(args.output_dir, 'parity_gnn_logK.png'))

    # Plot dP
    plot_parity(df_res['true_dP'], df_res['pred_cnn_dP'], "CNN Backbone: Pressure Drop", "Pa", 
                os.path.join(args.output_dir, 'parity_cnn_dP.png'))
    if gnn:
        plot_parity(df_res['true_dP'], df_res['pred_gnn_dP'], "Pore-GNN: Pressure Drop", "Pa", 
                    os.path.join(args.output_dir, 'parity_gnn_dP.png'))
    
    print("Done.")

if __name__ == '__main__':
    main()
