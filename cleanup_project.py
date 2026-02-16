#!/usr/bin/env python3
"""Project cleanup utility (non-destructive defaults).

Removes transient caches/artifacts and optionally moves numbered root JPGs into examples/.
"""

from pathlib import Path
import shutil


def cleanup_project() -> None:
    project_root = Path(".")

    print("=" * 60)
    print("PROJECT CLEANUP")
    print("=" * 60)

    dirs_to_remove = [
        "__pycache__",
        ".pytest_cache",
        ".mypy_cache",
        "build",
        "dist",
    ]
    files_to_remove = [".DS_Store"]

    removed_dirs = []
    removed_files = []

    print("\nRemoving transient directories...")
    print("-" * 60)
    for dir_name in dirs_to_remove:
        dir_path = project_root / dir_name
        if dir_path.exists() and dir_path.is_dir():
            shutil.rmtree(dir_path)
            removed_dirs.append(dir_name)
            print(f"  [REMOVED] {dir_name}/")
        else:
            print(f"  [SKIP] {dir_name}/ (not found)")

    print("\nRemoving transient files...")
    print("-" * 60)
    for file_name in files_to_remove:
        file_path = project_root / file_name
        if file_path.exists() and file_path.is_file():
            file_path.unlink()
            removed_files.append(file_name)
            print(f"  [REMOVED] {file_name}")
        else:
            print(f"  [SKIP] {file_name} (not found)")

    print("\nOrganizing numbered root images...")
    print("-" * 60)
    examples_original = project_root / "examples" / "original"
    examples_original.mkdir(parents=True, exist_ok=True)

    moved_images = 0
    for jpg_file in project_root.glob("*.jpg"):
        if not jpg_file.stem.isdigit():
            continue
        shutil.move(str(jpg_file), str(examples_original / jpg_file.name))
        moved_images += 1
    print(f"  [MOVED] {moved_images} images to examples/original/")

    print("\n" + "=" * 60)
    print("CLEANUP SUMMARY")
    print("=" * 60)
    print(f"Directories removed: {len(removed_dirs)}")
    print(f"Files removed: {len(removed_files)}")
    print(f"Images moved: {moved_images}")
    print("[SUCCESS] Project cleanup complete")


if __name__ == "__main__":
    cleanup_project()
