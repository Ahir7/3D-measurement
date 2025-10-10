#!/usr/bin/env python3
"""
Project Cleanup Script

This script removes old implementations and organizes the project structure:
1. Removes old DUSt3R/MASt3R implementation
2. Removes old server directory
3. Removes test files
4. Organizes images into examples/ directory
5. Cleans up unnecessary files
"""

import shutil
import os
from pathlib import Path

def cleanup_project():
    """Clean up the project directory."""
    
    project_root = Path(".")
    
    print("=" * 60)
    print("PROJECT CLEANUP")
    print("=" * 60)
    print()
    
    # List of directories to remove
    dirs_to_remove = [
        "dust3r",           # Old DUSt3R implementation
        "mast3r",           # Old MASt3R implementation
        "server",           # Old Flask/FastAPI server
        "tests",            # Old test files
        "scripts",          # Old setup scripts
        "mobile_app",       # Mobile app (if not needed)
        "results",          # Old results (keep output/)
        "config",           # Old config (we use configs/)
        "data",             # Old data directory
        "models",           # Old models directory (we use downloaded models)
    ]
    
    # List of files to remove
    files_to_remove = [
        "test_dust3r.py",
        "test_main.py",
        "test_scale_recovery.py",
        "check_progress.py",
        "python",           # Empty file
    ]
    
    removed_dirs = []
    removed_files = []
    
    # Remove directories
    print("Removing old implementation directories...")
    print("-" * 60)
    for dir_name in dirs_to_remove:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            try:
                shutil.rmtree(dir_path)
                removed_dirs.append(dir_name)
                print(f"  [REMOVED] {dir_name}/")
            except Exception as e:
                print(f"  [ERROR] Could not remove {dir_name}/: {e}")
        else:
            print(f"  [SKIP] {dir_name}/ (not found)")
    
    print()
    
    # Remove files
    print("Removing unnecessary files...")
    print("-" * 60)
    for file_name in files_to_remove:
        file_path = project_root / file_name
        if file_path.exists() and file_path.is_file():
            try:
                file_path.unlink()
                removed_files.append(file_name)
                print(f"  [REMOVED] {file_name}")
            except Exception as e:
                print(f"  [ERROR] Could not remove {file_name}: {e}")
        else:
            print(f"  [SKIP] {file_name} (not found)")
    
    print()
    
    # Create examples directory and move images
    print("Organizing images into examples/ directory...")
    print("-" * 60)
    
    examples_dir = project_root / "examples"
    examples_dir.mkdir(exist_ok=True)
    
    # Create subdirectories
    (examples_dir / "original").mkdir(exist_ok=True)
    (examples_dir / "resized").mkdir(exist_ok=True)
    
    # Move original images (1.jpg, 2.jpg, etc.)
    moved_original = 0
    for jpg_file in project_root.glob("*.jpg"):
        try:
            shutil.move(str(jpg_file), str(examples_dir / "original" / jpg_file.name))
            moved_original += 1
        except Exception as e:
            print(f"  [ERROR] Could not move {jpg_file.name}: {e}")
    
    print(f"  [MOVED] {moved_original} images to examples/original/")
    
    # Move resized directory
    resized_dir = project_root / "resized"
    if resized_dir.exists():
        try:
            # Move contents
            for jpg_file in resized_dir.glob("*.jpg"):
                shutil.move(str(jpg_file), str(examples_dir / "resized" / jpg_file.name))
            
            # Remove empty resized directory
            resized_dir.rmdir()
            print(f"  [MOVED] Resized images to examples/resized/")
        except Exception as e:
            print(f"  [ERROR] Could not move resized images: {e}")
    
    print()
    
    # Create README in examples
    examples_readme = examples_dir / "README.md"
    with open(examples_readme, 'w') as f:
        f.write("""# Example Images

This directory contains example images for testing the 3D measurement system.

## Directory Structure

- `original/` - Original high-resolution images (3072x4096)
- `resized/` - Resized images (768x1024) optimized for 4GB GPU

## Usage

### Using Original Images
```bash
# Resize them first
python resize_images.py

# Then measure
python main.py measure examples/resized/*.jpg
```

### Using Pre-resized Images
```bash
python main.py measure examples/resized/*.jpg
```

## Image Requirements

For best results:
- **Minimum**: 15 images
- **Optimal**: 20-25 images
- **Maximum**: 30 images (for 4GB GPU)
- 60-80% overlap between consecutive images
- Good lighting and focus
- Cover the object from multiple angles

See `IMAGE_CAPTURE_GUIDE.md` for detailed guidelines.
""")
    print(f"  [CREATED] examples/README.md")
    
    print()
    print("=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Directories removed: {len(removed_dirs)}")
    for d in removed_dirs:
        print(f"  - {d}/")
    print()
    print(f"Files removed: {len(removed_files)}")
    for f in removed_files:
        print(f"  - {f}")
    print()
    print(f"Images organized:")
    print(f"  - Original images moved to examples/original/")
    print(f"  - Resized images moved to examples/resized/")
    print()
    print("[SUCCESS] Project cleanup complete!")
    print()
    print("Next steps:")
    print("  1. Review the changes")
    print("  2. Update any scripts that reference old paths")
    print("  3. Test the system: python main.py measure examples/resized/*.jpg")
    print()

if __name__ == "__main__":
    try:
        cleanup_project()
    except KeyboardInterrupt:
        print("\n[CANCELLED] Cleanup cancelled by user")
    except Exception as e:
        print(f"\n[ERROR] Cleanup failed: {e}")
        import traceback
        traceback.print_exc()

