#!/usr/bin/env python3
"""
API Testing Script for 3D Measurement System

This script tests all API endpoints and helps verify the server is working correctly.

Usage:
    python test_api.py                    # Test all endpoints
    python test_api.py --health           # Test health endpoint only
    python test_api.py --measure IMG1 IMG2 IMG3  # Test measurement endpoint
"""

import sys
import argparse
import requests
import json
from pathlib import Path
from typing import List

# Configuration
API_URL = "http://localhost:8000"
TIMEOUT = 180  # 3 minutes for measurement endpoint


def print_section(title: str):
    """Print formatted section header."""
    print("\n" + "=" * 60)
    print(f" {title}")
    print("=" * 60)


def print_success(message: str):
    """Print success message."""
    print(f"[OK] {message}")


def print_error(message: str):
    """Print error message."""
    print(f"[ERROR] {message}")


def print_info(message: str):
    """Print info message."""
    print(f"[INFO] {message}")


def test_root():
    """Test root endpoint."""
    print_section("Testing Root Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            print_success("Root endpoint is accessible")
            print_info(f"Service: {data.get('service')}")
            print_info(f"Version: {data.get('version')}")
            print_info(f"Status: {data.get('status')}")
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except requests.exceptions.ConnectionError:
        print_error("Cannot connect to server. Is it running?")
        print_info(f"Try: python main.py serve --host 0.0.0.0 --port 8000")
        return False
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def test_health():
    """Test health endpoint."""
    print_section("Testing Health Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print_success("Health endpoint is accessible")
            print_info(f"Status: {data.get('status')}")
            print_info(f"System Ready: {data.get('system_ready')}")
            print_info(f"GPU Available: {data.get('gpu_available')}")
            
            gpu_info = data.get('gpu_info', {})
            if gpu_info.get('available'):
                print_info(f"GPU Device: {gpu_info.get('device_name')}")
                print_info(f"CUDA Version: {gpu_info.get('cuda_version')}")
                print_info(f"GPU Memory: {gpu_info.get('total_memory_gb'):.2f} GB")
                print_info(f"Free Memory: {gpu_info.get('free_memory_gb'):.2f} GB")
            
            if data.get('system_ready'):
                print_success("System is READY for measurements")
                return True
            else:
                print_error("System is NOT ready")
                return False
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def test_gpu_stats():
    """Test GPU stats endpoint."""
    print_section("Testing GPU Stats Endpoint")
    
    try:
        response = requests.get(f"{API_URL}/gpu-stats", timeout=5)
        
        if response.status_code == 200:
            data = response.json()
            
            print_success("GPU stats endpoint is accessible")
            
            stats = data.get('gpu_stats', {})
            if stats.get('available'):
                print_info(f"Device: {stats.get('device_name')}")
                print_info(f"Total Memory: {stats.get('total_memory_gb'):.2f} GB")
                print_info(f"Allocated: {stats.get('allocated_memory_gb'):.2f} GB")
                print_info(f"Reserved: {stats.get('reserved_memory_gb'):.2f} GB")
                print_info(f"Free: {stats.get('free_memory_gb'):.2f} GB")
            else:
                print_error("GPU not available")
            
            return True
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            return False
            
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def test_measure(image_paths: List[str]):
    """Test measurement endpoint."""
    print_section("Testing Measurement Endpoint")
    
    # Validate image files
    files_data = []
    for img_path in image_paths:
        path = Path(img_path)
        if not path.exists():
            print_error(f"Image not found: {img_path}")
            return False
        files_data.append(('files', (path.name, open(path, 'rb'), 'image/jpeg')))
    
    print_info(f"Uploading {len(files_data)} images...")
    
    try:
        response = requests.post(
            f"{API_URL}/measure",
            files=files_data,
            timeout=TIMEOUT
        )
        
        # Close files
        for _, (_, f, _) in files_data:
            f.close()
        
        if response.status_code == 200:
            data = response.json()
            
            print_success("Measurement completed successfully")
            print()
            
            # Display measurements
            measurements = data.get('measurements', {})
            print("Measurements:")
            print(f"  Width:  {measurements.get('width', 0):.2f} cm")
            print(f"  Height: {measurements.get('height', 0):.2f} cm")
            print(f"  Depth:  {measurements.get('depth', 0):.2f} cm")
            print(f"  Volume: {measurements.get('volume_cm3', 0):.2f} cm³")
            print()
            
            # Display confidence
            confidence = data.get('confidence', 0)
            print(f"Confidence: {confidence:.1f}%")
            print()
            
            # Display timing
            times = data.get('processing_times', {})
            print("Processing Times:")
            print(f"  Total: {times.get('total_time', 0):.2f}s")
            print(f"  GPU: {times.get('gpu_time', 0):.2f}s")
            print()
            
            # Display reconstruction stats
            recon = data.get('reconstruction_stats', {})
            print("Reconstruction:")
            print(f"  Images: {recon.get('num_images', 0)}")
            print(f"  3D Points: {recon.get('num_3d_points', 0)}")
            print()
            
            # Display scale recovery
            scale = data.get('scale_recovery', {})
            print("Scale Recovery:")
            print(f"  Method: {scale.get('method', 'unknown')}")
            print(f"  Scale Factor: {scale.get('scale_factor', 0):.4f}")
            print()
            
            # Save to file
            output_file = Path("test_api_results.json")
            with open(output_file, 'w') as f:
                json.dump(data, f, indent=2)
            print_info(f"Full results saved to: {output_file}")
            
            return True
            
        elif response.status_code == 400:
            error = response.json()
            print_error(f"Bad request: {error.get('detail')}")
            return False
            
        elif response.status_code == 503:
            print_error("Server not ready. Check server logs.")
            return False
            
        elif response.status_code == 507:
            print_error("GPU out of memory. Try fewer/smaller images.")
            return False
            
        else:
            print_error(f"Unexpected status code: {response.status_code}")
            try:
                error = response.json()
                print_error(f"Detail: {error.get('detail')}")
            except:
                pass
            return False
            
    except requests.exceptions.Timeout:
        print_error(f"Request timeout after {TIMEOUT}s")
        print_info("Processing may still be running on server")
        return False
        
    except Exception as e:
        print_error(f"Request failed: {e}")
        return False


def test_all():
    """Run all tests."""
    print_section("API Test Suite")
    print(f"Server: {API_URL}")
    
    results = {
        'root': test_root(),
        'health': test_health(),
        'gpu_stats': test_gpu_stats(),
    }
    
    # Summary
    print_section("Test Summary")
    
    total = len(results)
    passed = sum(1 for v in results.values() if v)
    failed = total - passed
    
    print(f"Total Tests: {total}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print_success("All tests passed!")
        print()
        print("You can now test measurement with:")
        print("  python test_api.py --measure image1.jpg image2.jpg image3.jpg")
        return True
    else:
        print_error("Some tests failed. Check server status.")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Test 3D Measurement API endpoints"
    )
    
    parser.add_argument(
        '--url',
        default='http://localhost:8000',
        help='API server URL (default: http://localhost:8000)'
    )
    
    parser.add_argument(
        '--health',
        action='store_true',
        help='Test health endpoint only'
    )
    
    parser.add_argument(
        '--measure',
        nargs='+',
        metavar='IMAGE',
        help='Test measurement endpoint with images'
    )
    
    parser.add_argument(
        '--gpu-stats',
        action='store_true',
        help='Test GPU stats endpoint only'
    )
    
    args = parser.parse_args()
    
    # Update API URL
    global API_URL
    API_URL = args.url.rstrip('/')
    
    # Run specific tests
    if args.health:
        success = test_health()
        return 0 if success else 1
    
    if args.gpu_stats:
        success = test_gpu_stats()
        return 0 if success else 1
    
    if args.measure:
        success = test_measure(args.measure)
        return 0 if success else 1
    
    # Run all tests
    success = test_all()
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())

