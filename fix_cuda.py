#!/usr/bin/env python3
"""
Fix CUDA support by installing correct PyTorch version.

This script will:
1. Uninstall CPU-only PyTorch
2. Install PyTorch with CUDA 12.1 support
3. Verify installation
"""

import subprocess
import sys

def run_command(cmd, description):
    """Run a command and show progress."""
    print(f"\n{'='*60}")
    print(f"{description}")
    print(f"{'='*60}")
    print(f"Command: {cmd}\n")
    
    result = subprocess.run(cmd, shell=True, capture_output=False, text=True)
    
    if result.returncode != 0:
        print(f"\n[ERROR] {description} failed!")
        return False
    
    print(f"\n[SUCCESS] {description} completed!")
    return True

def main():
    print("\n" + "="*60)
    print("CUDA FIX SCRIPT - Installing PyTorch with CUDA Support")
    print("="*60)
    print("\nYour GPU: NVIDIA GeForce GTX 1650")
    print("CUDA Version: 12.9")
    print("Required: PyTorch with CUDA 12.1 support")
    print("\n" + "="*60)
    
    response = input("\nThis will reinstall PyTorch. Continue? (y/N): ")
    if response.lower() != 'y':
        print("Aborted.")
        return
    
    # Step 1: Uninstall current PyTorch
    if not run_command(
        "pip uninstall -y torch torchvision torchaudio",
        "Uninstalling CPU-only PyTorch"
    ):
        print("\n[WARNING] Uninstall had issues, continuing anyway...")
    
    # Step 2: Install PyTorch with CUDA 12.1
    if not run_command(
        "pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121",
        "Installing PyTorch with CUDA 12.1 support"
    ):
        print("\n[ERROR] Installation failed!")
        print("\nTry manually:")
        print("pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121")
        return
    
    # Step 3: Verify installation
    print("\n" + "="*60)
    print("VERIFYING CUDA SUPPORT")
    print("="*60)
    
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        print(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"CUDA version: {torch.version.cuda}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
            
            print("\n" + "="*60)
            print("[SUCCESS] CUDA is now working!")
            print("="*60)
            
            # Quick GPU test
            print("\nRunning quick GPU test...")
            x = torch.randn(1000, 1000).cuda()
            y = torch.mm(x, x)
            print("[SUCCESS] GPU computation test passed!")
            
        else:
            print("\n[ERROR] CUDA still not available!")
            print("\nPossible issues:")
            print("1. NVIDIA drivers not installed")
            print("2. CUDA toolkit not installed")
            print("3. Need to restart computer")
            
    except Exception as e:
        print(f"\n[ERROR] Verification failed: {e}")
        return
    
    print("\n" + "="*60)
    print("NEXT STEPS")
    print("="*60)
    print("\n1. Verify: python main.py info")
    print("2. Validate: python validate_system.py")
    print("3. Measure: python main.py measure img1.jpg img2.jpg img3.jpg")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()

