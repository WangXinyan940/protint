from .model.submodules import load_protein_mpnn, load_esm_c_model
from .dataset.gen_embed import parse_pdb_file
from .dataset.dataloader import create_dataloader, create_pair_dataloader
from .workflow.train import train
from .workflow.predict import predict_single, predict_pkl, predict_directory, predict_pairs_directory, load_model, save_results
from .model.model import AntigenAntibodyModel
import json
import torch
import argparse
from pathlib import Path
import pickle


def parse_pdb_dataset(args: argparse.Namespace):
    """Generate embeddings for all proteins in a dataset and create train/val CSV files.

    The dataset directory should contain:
    - antigen/: PDB files for antigens
    - antibody/: PDB files for antibodies
    - targets.csv: Pairing information with labels

    Output:
    - Individual pkl files for each protein in the output directory root
    - train.csv: Training pairs with pkl filename references
    - val.csv: Validation pairs with pkl filename references
    """
    import pandas as pd

    esm_c = load_esm_c_model(args.esm)
    protein_mpnn = load_protein_mpnn(args.mpnn)

    dataset_dir = args.input
    output_dir = args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    # Read targets.csv
    targets_path = dataset_dir / "targets.csv"
    if not targets_path.exists():
        raise FileNotFoundError(f"targets.csv not found in {dataset_dir}")

    targets_df = pd.read_csv(targets_path)

    # Get unique antigen and antibody names
    unique_antigens = set(targets_df['antigen'].unique())
    unique_antibodies = set(targets_df['antibody'].unique())

    # Process antigens
    antigen_dir = dataset_dir / "antigen"
    print(f"\nProcessing {len(unique_antigens)} antigens from {antigen_dir}...")
    for antigen_name in unique_antigens:
        pdb_file = antigen_dir / f"{antigen_name}.pdb"
        if not pdb_file.exists():
            raise FileNotFoundError(f"Antigen PDB not found: {pdb_file}")

        embed = parse_pdb_file(str(pdb_file), esm_c, protein_mpnn, is_antibody=False)
        output_path = output_dir / f"{antigen_name}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(embed, f)
        print(f"  Saved: {output_path}")

    # Process antibodies
    antibody_dir = dataset_dir / "antibody"
    print(f"\nProcessing {len(unique_antibodies)} antibodies from {antibody_dir}...")
    for antibody_name in unique_antibodies:
        pdb_file = antibody_dir / f"{antibody_name}.pdb"
        if not pdb_file.exists():
            raise FileNotFoundError(f"Antibody PDB not found: {pdb_file}")

        embed = parse_pdb_file(str(pdb_file), esm_c, protein_mpnn, is_antibody=True)
        output_path = output_dir / f"{antibody_name}.pkl"
        with open(output_path, "wb") as f:
            pickle.dump(embed, f)
        print(f"  Saved: {output_path}")

    # Create pairs.csv with pkl filenames
    pairs_df = targets_df.copy()
    pairs_df['antibody_pkl'] = pairs_df['antibody'] + '.pkl'
    pairs_df['antigen_pkl'] = pairs_df['antigen'] + '.pkl'

    # Shuffle and split into train/val sets
    pairs_df = pairs_df.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    n_val = int(len(pairs_df) * args.val_ratio)
    val_df = pairs_df.iloc[:n_val].reset_index(drop=True)
    train_df = pairs_df.iloc[n_val:].reset_index(drop=True)

    # Save train.csv and val.csv in the root of output directory
    train_df[['antibody_pkl', 'antigen_pkl', 'classification_label']].to_csv(
        output_dir / "train.csv", index=False
    )
    val_df[['antibody_pkl', 'antigen_pkl', 'classification_label']].to_csv(
        output_dir / "val.csv", index=False
    )

    print(f"\nSaved train.csv to: {output_dir / 'train.csv'}")
    print(f"  Training pairs: {len(train_df)}")
    print(f"Saved val.csv to: {output_dir / 'val.csv'}")
    print(f"  Validation pairs: {len(val_df)}")


def train_model(args: argparse.Namespace):
    """Train the antigen-antibody prediction model.

    Expects data directory to contain:
    - pkl files for each protein
    - train.csv and val.csv with columns: antibody_pkl, antigen_pkl, classification_label
    """
    print("=" * 60)
    print("Training Antigen-Antibody Prediction Model")
    print("=" * 60)

    # Create data loaders using PairDataset
    print(f"\nLoading training data from: {args.data_dir}")
    train_loader = create_pair_dataloader(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        csv_file="train.csv",
    )
    print(f"  Training pairs: {len(train_loader.dataset)}")

    val_loader = None
    if args.val_data_dir:
        print(f"Loading validation data from: {args.val_data_dir}")
        val_loader = create_pair_dataloader(
            data_dir=args.val_data_dir,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            csv_file="val.csv",
        )
        print(f"  Validation pairs: {len(val_loader.dataset)}")

    # Create model
    print("\nCreating model...")
    model = AntigenAntibodyModel(
        node_input_dim=args.node_input_dim,
        edge_input_dim=args.edge_input_dim,
        hidden_dim=args.hidden_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        dropout=args.dropout,
    )
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\nStarting training...")
    lit_model = train(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        classification_weight=args.classification_weight,
        accelerator=args.accelerator,
        devices=args.devices,
        checkpoint_dir=args.checkpoint_dir,
    )

    print(f"\nTraining completed!")
    print(f"Best checkpoint saved to: {args.checkpoint_dir}")


def predict(args: argparse.Namespace):
    """Run prediction with trained model.

    Supports three modes:
    1. Directory with train.csv/val.csv or pairs.csv: Predicts all pairs defined in CSV
    2. Single pkl file or directory of paired pkl files: Legacy mode

    Output format is inferred from the output file extension:
    - .pkl: pickle format (binary)
    - .csv: CSV format (human-readable, excludes vector columns)
    - .json: JSON format (human-readable, includes all data)
    """
    print("=" * 60)
    print("Running Prediction")
    print("=" * 60)

    # Load model
    print(f"\nLoading model from: {args.checkpoint}")
    model = load_model(args.checkpoint, device=args.device)

    # Determine prediction mode
    input_path = args.input
    results = None

    if input_path.is_dir():
        # Check for CSV files in priority order: train.csv, val.csv, pairs.csv
        csv_file = None
        for candidate in ["train.csv", "val.csv", "pairs.csv"]:
            if (input_path / candidate).exists():
                csv_file = candidate
                break

        if csv_file:
            # Dataset mode: use CSV file
            print(f"Predicting on dataset directory: {args.input} (using {csv_file})")
            results = predict_pairs_directory(
                model=model,
                data_dir=args.input,
                output_path=args.output,
                device=args.device,
                csv_file=csv_file,
            )
            print(f"Processed {len(results)} pairs")
        else:
            # Legacy directory mode (no CSV, assume all pkl files are paired)
            print(f"Predicting on directory: {args.input}")
            results = predict_directory(
                model=model,
                data_dir=args.input,
                output_path=args.output,
                device=args.device,
            )
            print(f"Processed {len(results)} samples")
    else:
        # Single file mode
        print(f"Predicting file: {args.input}")
        result = predict_pkl(model, str(args.input), args.device)
        print(f"\nResults:")
        print(f"  Classification probability: {result['classification_prob']:.4f}")

        if args.output:
            save_results([result], str(args.output))


def entry():
    # initialize argument parser
    parser = argparse.ArgumentParser(description="ProtInt CLI for protein structure embedding using ESM-C and ProteinMPNN models.")
    # generate submodules.
    subparsers = parser.add_subparsers(dest="command")
    # gen-embed submodule
    gen_embed_parser = subparsers.add_parser("embed", help="Generate embeddings for proteins in a dataset directory.")
    gen_embed_parser.add_argument("-d", "--input", type=Path, required=True, dest="input", help="Path to the dataset directory (must contain antigen/, antibody/, and targets.csv).")
    gen_embed_parser.add_argument("-o", "--output", type=Path, required=True, help="Path to the output folder for saving embeddings, train.csv, and val.csv.")
    gen_embed_parser.add_argument("--esm", type=str, required=True, help="Path to the pre-trained ESM-C model file.")
    gen_embed_parser.add_argument("--mpnn", type=str, required=True, help="Path to the pre-trained ProteinMPNN model file.")
    gen_embed_parser.add_argument("--val-ratio", type=float, default=0.2, help="Validation set ratio (default: 0.2)")
    gen_embed_parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility (default: 42)")
    gen_embed_parser.set_defaults(func=parse_pdb_dataset)

    # train 子模块
    train_parser = subparsers.add_parser("train", help="Train the antigen-antibody prediction model.")
    train_parser.add_argument("--data-dir", type=Path, required=True, help="Directory containing training pkl files, train.csv, and val.csv")
    train_parser.add_argument("--val-data-dir", type=Path, help="Directory containing validation pkl files and val.csv (optional, uses train.csv/val.csv from data-dir if not provided)")
    train_parser.add_argument("--node-input-dim", type=int, default=1098, help="Node input dimension (ESM-C 960 + ProteinMPNN 128 + IMGT region 7 + chain type 3)")
    train_parser.add_argument("--edge-input-dim", type=int, default=128, help="Edge input dimension")
    train_parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    train_parser.add_argument("--num-heads", type=int, default=8, help="Number of attention heads")
    train_parser.add_argument("--num-layers", type=int, default=3, help="Number of Graph Transformer layers")
    train_parser.add_argument("--dropout", type=float, default=0.1, help="Dropout rate")
    train_parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    train_parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    train_parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    train_parser.add_argument("--weight-decay", type=float, default=1e-2, help="Weight decay")
    train_parser.add_argument("--classification-weight", type=float, default=1.0, help="Weight for classification loss")
    train_parser.add_argument("--num-workers", type=int, default=0, help="Number of data loading workers")
    train_parser.add_argument("--accelerator", type=str, default="auto", choices=["cpu", "gpu", "auto"], help="Accelerator type")
    train_parser.add_argument("--devices", type=int, default=1, help="Number of devices")
    train_parser.add_argument("--checkpoint-dir", type=Path, default="checkpoints", help="Directory to save checkpoints")
    train_parser.set_defaults(func=train_model)

    # predict submodule
    predict_parser = subparsers.add_parser("predict", help="Run prediction with trained model.")
    predict_parser.add_argument("-c", "--checkpoint", type=str, required=True, help="Path to model checkpoint")
    predict_parser.add_argument("-i", "--input", type=Path, required=True, help="Input pkl file, directory, or dataset directory with pairs.csv")
    predict_parser.add_argument("-o", "--output", type=Path, help="Output file for results (.pkl, .csv, or .json)")
    predict_parser.add_argument("--device", type=str, default=None, choices=["cpu", "cuda"], help="Device for inference")
    predict_parser.set_defaults(func=predict)

    args = parser.parse_args()

    if hasattr(args, "func"):
        args.func(args)
    else:
        parser.print_help()

        