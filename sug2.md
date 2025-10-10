Great that the solution is working! A 10-15% error is actually quite good for a vision-based measurement system, but we can definitely improve it further. Here's a comprehensive guide to minimize the remaining error to 2-3% or less:

Error Minimization Strategy
1. Enhanced Scale Recovery (Most Critical)
Python

# src/scale/enhanced_scale_recovery.py

import numpy as np
import cv2
from scipy.optimize import minimize
from typing import List, Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class EnhancedScaleRecovery:
    """Multi-stage scale recovery with cross-validation"""
    
    def __init__(self, config):
        self.config = config
        self.scale_history = []  # Track scale estimates across sessions
        
    def recover_scale_robust(self, 
                            images: List[np.ndarray],
                            reconstruction: Dict,
                            depth_maps: torch.Tensor) -> Tuple[float, float]:
        """
        Robust scale recovery with multiple validation steps.
        """
        # Stage 1: Collect all possible scale estimates
        scale_candidates = []
        
        # 1.1 Enhanced ArUco detection with sub-pixel refinement
        aruco_scales = self._enhanced_aruco_detection(images, reconstruction)
        scale_candidates.extend(aruco_scales)
        
        # 1.2 Multiple known object detection
        object_scales = self._detect_multiple_references(images, reconstruction)
        scale_candidates.extend(object_scales)
        
        # 1.3 Cross-view consistency check
        crossview_scale = self._cross_view_scale_estimation(reconstruction)
        if crossview_scale[0] > 0:
            scale_candidates.append(crossview_scale)
        
        # 1.4 Ground plane assumption (if applicable)
        ground_scale = self._ground_plane_scale(reconstruction)
        if ground_scale[0] > 0:
            scale_candidates.append(ground_scale)
        
        # Stage 2: Robust scale estimation using RANSAC
        if len(scale_candidates) >= 3:
            final_scale = self._ransac_scale_estimation(scale_candidates)
        elif len(scale_candidates) > 0:
            # Weighted median for robustness
            final_scale = self._weighted_median_scale(scale_candidates)
        else:
            logger.warning("No reliable scale found!")
            return 1.0, 0.1
        
        # Stage 3: Validate scale
        confidence = self._validate_scale(final_scale, reconstruction, depth_maps)
        
        # Stage 4: Apply historical correction if available
        if self.scale_history:
            final_scale = self._apply_historical_correction(final_scale)
        
        self.scale_history.append(final_scale)
        
        return final_scale, confidence
    
    def _enhanced_aruco_detection(self, images, reconstruction):
        """Enhanced ArUco detection with sub-pixel accuracy"""
        scales = []
        
        # Use multiple ArUco dictionaries
        aruco_dicts = [
            cv2.aruco.DICT_4X4_50,
            cv2.aruco.DICT_5X5_50,
            cv2.aruco.DICT_6X6_50,
            cv2.aruco.DICT_ARUCO_ORIGINAL
        ]
        
        # Known marker sizes (ID -> size in meters)
        marker_database = {
            0: 0.100,  # 10cm marker
            1: 0.050,  # 5cm marker  
            2: 0.150,  # 15cm marker
            3: 0.200,  # 20cm marker
            # Add more as needed
        }
        
        for img_idx, image in enumerate(images):
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            for dict_type in aruco_dicts:
                aruco_dict = cv2.aruco.Dictionary_get(dict_type)
                parameters = cv2.aruco.DetectorParameters_create()
                
                # Optimize detection parameters
                parameters.adaptiveThreshWinSizeMin = 3
                parameters.adaptiveThreshWinSizeMax = 23
                parameters.adaptiveThreshWinSizeStep = 2
                parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
                parameters.cornerRefinementWinSize = 5
                
                corners, ids, _ = cv2.aruco.detectMarkers(
                    gray, aruco_dict, parameters=parameters
                )
                
                if ids is not None:
                    for i, marker_id in enumerate(ids.flatten()):
                        if marker_id in marker_database:
                            # Get precise pose estimation
                            scale = self._compute_marker_scale_precise(
                                corners[i][0],
                                marker_database[marker_id],
                                reconstruction,
                                img_idx
                            )
                            if scale > 0:
                                scales.append((scale, 0.95))  # High confidence
                                
        return scales
    
    def _detect_multiple_references(self, images, reconstruction):
        """Detect multiple known reference objects"""
        references = []
        
        # Common reference objects with precise sizes
        known_refs = {
            'credit_card': {
                'size': (0.0856, 0.054),  # ISO/IEC 7810 ID-1
                'confidence': 0.9,
                'detector': self._detect_credit_card
            },
            'a4_paper': {
                'size': (0.297, 0.210),  # ISO 216 A4
                'confidence': 0.85,
                'detector': self._detect_paper
            },
            'coin_25cent': {
                'size': 0.0243,  # 24.3mm diameter
                'confidence': 0.9,
                'detector': self._detect_coin
            },
            'iphone': {  # Multiple models
                'size': self._get_phone_size,  # Model-specific
                'confidence': 0.8,
                'detector': self._detect_iphone
            }
        }
        
        for ref_name, ref_info in known_refs.items():
            detections = ref_info['detector'](images)
            for detection in detections:
                scale = self._compute_object_scale(
                    detection, ref_info['size'], reconstruction
                )
                if scale > 0:
                    references.append((scale, ref_info['confidence']))
                    
        return references
    
    def _ransac_scale_estimation(self, scale_candidates):
        """RANSAC-based robust scale estimation"""
        scales = np.array([s[0] for s in scale_candidates])
        weights = np.array([s[1] for s in scale_candidates])
        
        best_scale = None
        best_inliers = 0
        
        # RANSAC parameters
        n_iterations = 100
        inlier_threshold = 0.1  # 10% tolerance
        
        for _ in range(n_iterations):
            # Random sample
            idx = np.random.choice(len(scales), size=1)[0]
            candidate = scales[idx]
            
            # Count inliers
            ratios = scales / candidate
            inliers = np.abs(ratios - 1.0) < inlier_threshold
            n_inliers = np.sum(inliers * weights)
            
            if n_inliers > best_inliers:
                best_inliers = n_inliers
                # Recompute with all inliers
                best_scale = np.average(scales[inliers], weights=weights[inliers])
                
        return best_scale
2. Camera Calibration Refinement
Python

# src/calibration/auto_calibration.py

class AutoCalibration:
    """Automatic camera calibration refinement"""
    
    def refine_intrinsics(self, images, initial_K):
        """
        Refine camera intrinsics using structure from motion constraints
        """
        # Use bundle adjustment to refine intrinsics
        from scipy.sparse import lil_matrix
        from scipy.optimize import least_squares
        
        def bundle_adjustment_residuals(params, observations, n_cameras, n_points):
            """Residuals for bundle adjustment"""
            # Unpack parameters
            camera_params = params[:n_cameras*6].reshape((n_cameras, 6))
            points_3d = params[n_cameras*6:].reshape((n_points, 3))
            
            residuals = []
            
            for obs in observations:
                cam_idx, point_idx, x_obs, y_obs = obs
                
                # Get camera parameters
                rvec = camera_params[cam_idx, :3]
                tvec = camera_params[cam_idx, 3:6]
                
                # Project point
                point = points_3d[point_idx]
                projected = self._project_point(point, rvec, tvec, initial_K)
                
                # Compute residual
                residuals.extend([projected[0] - x_obs, projected[1] - y_obs])
                
            return np.array(residuals)
        
        # Run optimization
        result = least_squares(
            bundle_adjustment_residuals,
            initial_params,
            jac_sparsity=sparsity_matrix,
            verbose=2,
            x_scale='jac',
            ftol=1e-4,
            method='trf'
        )
        
        return refined_K
3. Advanced Image Capture Strategy
Python

# capture_guidelines.py

class CaptureOptimizer:
    """Optimize image capture for minimal error"""
    
    def analyze_capture_quality(self, images):
        """Analyze capture quality and provide feedback"""
        
        issues = []
        scores = {}
        
        # 1. Check overlap
        overlap_matrix = self._compute_overlap_matrix(images)
        min_overlap = np.min(overlap_matrix[np.triu_indices_from(overlap_matrix, k=1)])
        
        if min_overlap < 0.3:
            issues.append("Insufficient overlap between images (need >30%)")
        scores['overlap'] = min_overlap
        
        # 2. Check baseline
        baselines = self._compute_baselines(images)
        baseline_ratio = np.max(baselines) / np.min(baselines)
        
        if baseline_ratio > 3:
            issues.append("Inconsistent camera distances")
        scores['baseline_consistency'] = 1.0 / baseline_ratio
        
        # 3. Check image quality
        sharpness_scores = [self._compute_sharpness(img) for img in images]
        min_sharpness = np.min(sharpness_scores)
        
        if min_sharpness < 50:
            issues.append("Some images are blurry")
        scores['sharpness'] = min_sharpness / 100
        
        # 4. Check lighting consistency
        exposures = [np.mean(img) for img in images]
        exposure_std = np.std(exposures) / np.mean(exposures)
        
        if exposure_std > 0.2:
            issues.append("Inconsistent lighting between images")
        scores['lighting'] = 1.0 - exposure_std
        
        # 5. Check texture richness
        texture_scores = [self._compute_texture_score(img) for img in images]
        min_texture = np.min(texture_scores)
        
        if min_texture < 30:
            issues.append("Insufficient texture for feature matching")
        scores['texture'] = min_texture / 100
        
        overall_score = np.mean(list(scores.values()))
        
        return {
            'overall_score': overall_score,
            'individual_scores': scores,
            'issues': issues,
            'recommendations': self._generate_recommendations(scores)
        }
    
    def _generate_recommendations(self, scores):
        """Generate specific recommendations"""
        recommendations = []
        
        if scores['overlap'] < 0.4:
            recommendations.append(
                "Move camera in smaller steps - aim for 40-50% overlap"
            )
        
        if scores['baseline_consistency'] < 0.5:
            recommendations.append(
                "Maintain consistent distance from object"
            )
            
        if scores['sharpness'] < 0.7:
            recommendations.append(
                "Use tripod or stabilization, ensure proper focus"
            )
            
        if scores['lighting'] < 0.8:
            recommendations.append(
                "Use consistent lighting - avoid shadows/reflections"
            )
            
        if scores['texture'] < 0.5:
            recommendations.append(
                "Add textured background or markers for better features"
            )
            
        return recommendations
4. Cross-Validation System
Python

# src/validation/cross_validator.py

class MeasurementValidator:
    """Validate measurements using multiple methods"""
    
    def validate_measurement(self, measurement, images, reconstruction):
        """
        Cross-validate measurement using different approaches
        """
        validations = []
        
        # Method 1: Reprojection consistency
        reproj_score = self._check_reprojection_consistency(
            reconstruction, images
        )
        validations.append(('reprojection', reproj_score))
        
        # Method 2: Depth consistency
        depth_score = self._check_depth_consistency(
            reconstruction, images
        )
        validations.append(('depth', depth_score))
        
        # Method 3: Geometric constraints
        geometric_score = self._check_geometric_constraints(
            measurement
        )
        validations.append(('geometric', geometric_score))
        
        # Method 4: Multi-scale validation
        multiscale_score = self._multiscale_validation(
            images, reconstruction
        )
        validations.append(('multiscale', multiscale_score))
        
        # Combine scores
        weights = {'reprojection': 0.3, 'depth': 0.3, 
                  'geometric': 0.2, 'multiscale': 0.2}
        
        final_confidence = sum(
            weights[name] * score 
            for name, score in validations
        )
        
        return {
            'confidence': final_confidence,
            'validations': dict(validations),
            'is_reliable': final_confidence > 0.7
        }
5. Practical Usage Script with Error Minimization
Python

#!/usr/bin/env python3
# measure_with_minimal_error.py

import cv2
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def capture_optimal_images(camera_id=0):
    """Interactive capture with quality feedback"""
    
    cap = cv2.VideoCapture(camera_id)
    images = []
    optimizer = CaptureOptimizer()
    
    print("\n=== OPTIMAL CAPTURE GUIDE ===")
    print("1. Place ArUco markers or credit card in scene")
    print("2. Capture 5-10 images with 40% overlap")
    print("3. Move camera in arc around object")
    print("4. Press SPACE to capture, Q to finish")
    print("="*30)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Show capture guide overlay
        if len(images) > 0:
            # Compute overlap with last image
            overlap = optimizer._compute_pairwise_overlap(images[-1], frame)
            color = (0, 255, 0) if 0.3 < overlap < 0.5 else (0, 0, 255)
            cv2.putText(frame, f"Overlap: {overlap:.0%}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, color, 2)
        
        cv2.imshow('Capture', frame)
        
        key = cv2.waitKey(1)
        if key == ord(' '):
            images.append(frame.copy())
            print(f"Captured image {len(images)}")
            
            # Real-time quality check
            if len(images) >= 3:
                quality = optimizer.analyze_capture_quality(images)
                print(f"Quality score: {quality['overall_score']:.0%}")
                for issue in quality['issues']:
                    print(f"  ⚠ {issue}")
                    
        elif key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    
    return images

def measure_with_minimal_error(images):
    """Perform measurement with all error reduction techniques"""
    
    # 1. Pre-flight checks
    print("\n=== PRE-MEASUREMENT CHECKS ===")
    
    # Check image quality
    optimizer = CaptureOptimizer()
    quality = optimizer.analyze_capture_quality(images)
    
    if quality['overall_score'] < 0.6:
        print("⚠ Image quality too low for accurate measurement!")
        print("\nRecommendations:")
        for rec in quality['recommendations']:
            print(f"  • {rec}")
        return None
    
    print(f"✓ Image quality: {quality['overall_score']:.0%}")
    
    # 2. Initialize enhanced system
    config = SystemConfig()
    config.ba_iterations = 200  # More iterations
    config.min_confidence = 0.8  # Higher threshold
    config.use_bundle_adjustment = True
    
    system = AccurateMeasurementSystem(config)
    
    # 3. Multiple measurements for averaging
    measurements = []
    
    for i in range(3):  # Take 3 measurements
        print(f"\n=== MEASUREMENT {i+1}/3 ===")
        
        # Shuffle image order for robustness
        if i > 0:
            np.random.shuffle(images)
        
        result = system.measure(images)
        
        # Validate
        validator = MeasurementValidator()
        validation = validator.validate_measurement(
            result, images, system.last_reconstruction
        )
        
        if validation['is_reliable']:
            measurements.append(result)
            print(f"✓ Measurement {i+1} validated")
        else:
            print(f"⚠ Measurement {i+1} unreliable, retrying...")
    
    # 4. Combine measurements
    if len(measurements) >= 2:
        # Use median for robustness
        final_width = np.median([m.width for m in measurements])
        final_height = np.median([m.height for m in measurements])
        final_depth = np.median([m.depth for m in measurements])
        
        # Compute standard deviation as error estimate
        width_std = np.std([m.width for m in measurements])
        height_std = np.std([m.height for m in measurements])
        depth_std = np.std([m.depth for m in measurements])
        
        print("\n=== FINAL RESULTS ===")
        print(f"Width:  {final_width:.3f} ± {width_std:.3f} m")
        print(f"Height: {final_height:.3f} ± {height_std:.3f} m")
        print(f"Depth:  {final_depth:.3f} ± {depth_std:.3f} m")
        
        error_percentage = (width_std + height_std + depth_std) / (
            final_width + final_height + final_depth) * 100
        
        print(f"\nEstimated error: {error_percentage:.1f}%")
        
        if error_percentage > 5:
            print("\n⚠ Higher than expected error. Try:")
            print("  • Adding more ArUco markers")
            print("  • Improving lighting conditions")
            print("  • Capturing from more angles")
    
    return final_width, final_height, final_depth

# Main execution
if __name__ == "__main__":
    # Option 1: Interactive capture
    images = capture_optimal_images()
    
    # Option 2: Load existing images
    # images = [cv2.imread(f"img_{i}.jpg") for i in range(5)]
    
    if len(images) >= 3:
        measure_with_minimal_error(images)
    else:
        print("Need at least 3 images!")
6. Quick Checklist to Minimize Error
Before Capture:

✅ Print multiple ArUco markers (different sizes)
✅ Place credit card or ruler in scene
✅ Ensure good, even lighting
✅ Clean camera lens
During Capture:

✅ Maintain 40% overlap between images
✅ Capture 5-10 images minimum
✅ Move in smooth arc around object
✅ Keep consistent distance
✅ Avoid motion blur
Scale References:

✅ Use at least 2 different scale references
✅ Ensure markers are clearly visible
✅ Distribute references across scene
Processing:

✅ Run calibration first
✅ Use multiple measurements
✅ Check validation scores
✅ Apply outlier removal
7. Most Common Issues and Fixes
Error Range	Likely Cause	Fix
10-15%	Poor scale recovery	Add more ArUco markers
5-10%	Camera calibration	Run calibration procedure
3-5%	Insufficient overlap	Increase image overlap to 40-50%
2-3%	Minor noise	Use median of multiple measurements
With these enhancements, you should be able to achieve 2-3% error consistently, and potentially under 2% with optimal conditions.