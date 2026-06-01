#!/usr/bin/env python
"""
ASL Transformer - Unified CLI Entry Point
Handles: train, infer, api, ui subcommands
"""

import argparse
import sys
from pathlib import Path

# Add root to path
sys.path.insert(0, str(Path(__file__).parent))

def run_train(args):
    """Train CTC model"""
    from train_ctc import TrainingConfig, CTCTrainer, DummySignLanguageDataset, collate_fn
    from torch.utils.data import DataLoader
    
    # Create config with overrides from CLI
    config = TrainingConfig()
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    
    # Create trainer (handles model creation internally)
    trainer = CTCTrainer(config)
    
    # Create dummy dataset (replace with real data later)
    train_dataset = DummySignLanguageDataset(num_samples=100)
    val_dataset = DummySignLanguageDataset(num_samples=20)
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, collate_fn=collate_fn)
    
    # Train
    trainer.train(train_loader, val_loader)

def run_infer(args):
    """Run inference (webcam or video file)"""
    from realtime.inference_ctc import FingerspellingInference
    import torch
    from utils.config import DEVICE, BEST_MODEL
    
    device = args.device if args.device else DEVICE
    model_path = args.model if hasattr(args, 'model') and args.model else str(BEST_MODEL)
    
    inference = FingerspellingInference(
        model_path=model_path,
        device=device,
        beam_width=args.beam_width
    )
    
    if args.mode == 'webcam':
        inference.run_webcam()
    elif args.mode == 'video':
        if not args.video:
            print("ERROR: --video required for video mode")
            sys.exit(1)
        inference.run_video(args.video)
    else:
        print(f"ERROR: Unknown mode {args.mode}")
        sys.exit(1)

def run_api(args):
    """Run FastAPI server"""
    import uvicorn
    from app.main import app
    
    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)

def run_ui(args):
    """Run Streamlit UI"""
    import subprocess
    
    subprocess.run([
        'streamlit', 'run',
        str(Path(__file__).parent / 'ui' / 'streamlit_app.py'),
        '--server.port', str(args.port)
    ])

def main():
    parser = argparse.ArgumentParser(
        description='ASL Transformer - Unified CLI'
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # TRAIN subcommand
    train_parser = subparsers.add_parser('train', help='Train CTC model')
    train_parser.add_argument('--epochs', type=int, default=100, help='Number of epochs')
    train_parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train_parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    train_parser.set_defaults(func=run_train)
    
    # INFER subcommand
    infer_parser = subparsers.add_parser('infer', help='Run inference')
    infer_parser.add_argument('--mode', type=str, choices=['webcam', 'video'], 
                             default='webcam', help='Inference mode')
    infer_parser.add_argument('--video', type=str, help='Path to video file (for video mode)')
    infer_parser.add_argument('--model', type=str, help='Path to model checkpoint')
    infer_parser.add_argument('--beam-width', type=int, default=5, help='Beam search width')
    infer_parser.add_argument('--device', type=str, default='cpu', help='Device (cpu/cuda)')
    infer_parser.set_defaults(func=run_infer)
    
    # API subcommand
    api_parser = subparsers.add_parser('api', help='Run FastAPI server')
    api_parser.add_argument('--host', type=str, default='127.0.0.1', help='Host')
    api_parser.add_argument('--port', type=int, default=8000, help='Port')
    api_parser.add_argument('--reload', action='store_true', help='Auto-reload on changes')
    api_parser.set_defaults(func=run_api)
    
    # UI subcommand
    ui_parser = subparsers.add_parser('ui', help='Run Streamlit UI')
    ui_parser.add_argument('--port', type=int, default=8501, help='Port')
    ui_parser.set_defaults(func=run_ui)
    
    # Parse args
    args = parser.parse_args()
    
    # Execute command
    if hasattr(args, 'func'):
        args.func(args)
    else:
        parser.print_help()
        sys.exit(1)

if __name__ == '__main__':
    main()