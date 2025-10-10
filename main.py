#!/usr/bin/env python3
"""
Main entry point for 3D Measurement System.

Usage:
    python main.py --help                    # Show help
    python main.py serve                     # Start API server
    python main.py measure image1.jpg image2.jpg image3.jpg  # Measure from command line
    python main.py benchmark                 # Run performance benchmark
"""

import sys
import argparse
import logging
from pathlib import Path

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def serve_command(args):
    """Start the FastAPI server."""
    from src.api.rest_api import start_server
    
    logger.info(f"Starting API server on {args.host}:{args.port}")
    start_server(
        host=args.host,
        port=args.port,
        reload=args.reload
    )


def measure_command(args):
    """Run measurement from command line."""
    import torch
    import cv2
    import numpy as np
    import json
    from src.core.measurement_system_gpu import MeasurementSystemGPU
    from src.core.config import SystemConfig
    
    # Check GPU
    if not torch.cuda.is_available():
        logger.error("GPU not available. This system requires CUDA.")
        return 1
    
    # Load images
    images = []
    image_paths = []
    
    for img_path in args.images:
        path = Path(img_path)
        if not path.exists():
            logger.error(f"Image not found: {img_path}")
            return 1
        
        img = cv2.imread(str(path))
        if img is None:
            logger.error(f"Failed to load image: {img_path}")
            return 1
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        image_paths.append(path)
    
    logger.info(f"Loaded {len(images)} images")
    
    # Load optional data
    imu_data = None
    if args.imu_data:
        with open(args.imu_data, 'r') as f:
            imu_data = json.load(f)
        logger.info("Loaded IMU data")
    
    metadata = None
    if args.metadata:
        with open(args.metadata, 'r') as f:
            metadata = json.load(f)
        logger.info("Loaded metadata")
    
    # Initialize system
    logger.info("Initializing measurement system...")
    config = SystemConfig()
    config.output_dir = Path(args.output)
    system = MeasurementSystemGPU(config)
    
    # Run measurement
    logger.info("Running measurement...")
    result = system.measure(
        images=images,
        image_paths=image_paths,
        imu_data=imu_data,
        metadata=metadata
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("MEASUREMENT RESULTS")
    print("=" * 60)
    
    if result.error_bounds:
        # Print with error bounds
        print(f"Width:  {result.measurements['width']:.2f} ± {result.error_bounds['width_error']:.2f} cm")
        print(f"Height: {result.measurements['height']:.2f} ± {result.error_bounds['height_error']:.2f} cm")
        print(f"Depth:  {result.measurements['depth']:.2f} ± {result.error_bounds['depth_error']:.2f} cm")
        print(f"Volume: {result.measurements['volume_cm3']:.2f} ± {result.error_bounds.get('volume_error', 0):.2f} cm³")
        print(f"\nEstimated Error: ±{result.error_bounds['relative_error_percent']:.1f}%")
        print(f"Quality: {result.error_bounds.get('quality', 'N/A')}")
    else:
        # Print without error bounds
        print(f"Width:  {result.measurements['width']:.2f} cm")
        print(f"Height: {result.measurements['height']:.2f} cm")
        print(f"Depth:  {result.measurements['depth']:.2f} cm")
        print(f"Volume: {result.measurements['volume_cm3']:.2f} cm³")
    
    print(f"\nConfidence: {result.confidence:.1%}")
    print(f"Processing Time: {result.total_time:.2f}s")
    print(f"GPU Time: {result.gpu_time:.2f}s")
    print("=" * 60)
    
    # Save results
    output_file = config.output_dir / "results.json"
    with open(output_file, 'w') as f:
        json.dump(result.to_dict(), f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    if result.pointcloud_path:
        print(f"Point cloud saved to: {result.pointcloud_path}")
    
    return 0


def benchmark_command(args):
    """Run performance benchmark."""
    import torch
    from src.core.measurement_system_gpu import MeasurementSystemGPU
    from src.core.config import SystemConfig
    
    if not torch.cuda.is_available():
        logger.error("GPU not available. This system requires CUDA.")
        return 1
    
    logger.info("Initializing measurement system...")
    config = SystemConfig()
    system = MeasurementSystemGPU(config)
    
    logger.info(f"Running benchmark: {args.num_images} images, {args.num_runs} runs")
    results = system.benchmark(
        num_images=args.num_images,
        image_size=(args.image_size, args.image_size),
        num_runs=args.num_runs
    )
    
    # Print results
    print("\n" + "=" * 60)
    print("BENCHMARK RESULTS")
    print("=" * 60)
    print(f"Images: {results['num_images']}")
    print(f"Image Size: {results['image_size']}")
    print(f"Runs: {results['num_runs']}")
    print(f"\nMean Time: {results['mean_time']:.3f}s ± {results['std_time']:.3f}s")
    print(f"Min Time:  {results['min_time']:.3f}s")
    print(f"Max Time:  {results['max_time']:.3f}s")
    print(f"Throughput: {results['throughput']:.2f} images/second")
    print("=" * 60)
    
    return 0


def info_command(args):
    """Show system information."""
    import torch
    from src.core.config import get_gpu_info
    
    print("\n" + "=" * 60)
    print("3D MEASUREMENT SYSTEM INFORMATION")
    print("=" * 60)
    
    # Python and PyTorch
    print(f"\nPython: {sys.version}")
    print(f"PyTorch: {torch.__version__}")
    
    # GPU info
    if torch.cuda.is_available():
        gpu_info = get_gpu_info()
        print(f"\nGPU: {gpu_info['name']}")
        print(f"CUDA: {gpu_info['cuda_version']}")
        print(f"Compute Capability: {gpu_info['compute_capability']}")
        print(f"Total Memory: {gpu_info['total_memory_gb']:.2f} GB")
        print(f"Allocated: {gpu_info['allocated_memory_gb']:.2f} GB")
        print(f"Reserved: {gpu_info['reserved_memory_gb']:.2f} GB")
    else:
        print("\nGPU: Not available")
        print("WARNING: This system requires a CUDA-capable GPU")
    
    print("=" * 60 + "\n")
    
    return 0


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="3D Measurement System - GPU-accelerated dimensional analysis"
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Serve command
    serve_parser = subparsers.add_parser('serve', help='Start API server')
    serve_parser.add_argument('--host', default='0.0.0.0', help='Server host')
    serve_parser.add_argument('--port', type=int, default=8000, help='Server port')
    serve_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    
    # Measure command
    measure_parser = subparsers.add_parser('measure', help='Measure from images')
    measure_parser.add_argument('images', nargs='+', help='Input image files')
    measure_parser.add_argument('--imu-data', help='IMU data JSON file')
    measure_parser.add_argument('--metadata', help='Metadata JSON file')
    measure_parser.add_argument('--output', '-o', default='output', help='Output directory')
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser('benchmark', help='Run performance benchmark')
    benchmark_parser.add_argument('--num-images', type=int, default=5, help='Number of images')
    benchmark_parser.add_argument('--image-size', type=int, default=1024, help='Image size')
    benchmark_parser.add_argument('--num-runs', type=int, default=3, help='Number of runs')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system information')
    
    args = parser.parse_args()
    
    if args.command == 'serve':
        return serve_command(args)
    elif args.command == 'measure':
        return measure_command(args)
    elif args.command == 'benchmark':
        return benchmark_command(args)
    elif args.command == 'info':
        return info_command(args)
    else:
        parser.print_help()
        return 0


if __name__ == "__main__":
    sys.exit(main())

