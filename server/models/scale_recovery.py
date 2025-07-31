import numpy as np

class MultiMethodScaleRecovery:
    def __init__(self):
        self.confidence_weights = {
            'imu': 0.35,
            'metadata': 0.25,
            'geometric': 0.20,
            'ml_depth': 0.15,
            'motion_parallax': 0.05
        }
    async def estimate_scale(self, images, imu_data, metadata, timestamps, reconstruction):
        scale_estimates = {}
        confidences = {}
        if imu_data:
            imu_result = await self.estimate_scale_from_imu(
                imu_data, reconstruction['camera_poses'], timestamps
            )
            if imu_result['success']:
                scale_estimates['imu'] = imu_result['scale']
                confidences['imu'] = imu_result['confidence']
        if metadata:
            metadata_result = await self.estimate_scale_from_metadata(
                metadata, reconstruction.get('depth_maps', [])
            )
            if metadata_result['success']:
                scale_estimates['metadata'] = metadata_result['scale']
                confidences['metadata'] = metadata_result['confidence']
        geometric_result = await self.estimate_scale_from_geometry(
            reconstruction['camera_poses'], reconstruction['point_cloud']
        )
        if geometric_result['success']:
            scale_estimates['geometric'] = geometric_result['scale']
            confidences['geometric'] = geometric_result['confidence']
        final_scale, final_confidence = self.fuse_scale_estimates(
            scale_estimates, confidences
        )
        return {
            'scale_factor': final_scale,
            'confidence': final_confidence,
            'methods_used': list(scale_estimates.keys()),
            'individual_estimates': scale_estimates,
            'individual_confidences': confidences
        }
    async def estimate_scale_from_imu(self, imu_data, camera_poses, timestamps):
        try:
            if not imu_data or len(imu_data) < 2:
                return {'success': False, 'error': 'Insufficient IMU data'}
            real_world_motion = self.integrate_imu_motion(imu_data, timestamps)
            camera_motion = self.calculate_camera_motion(camera_poses)
            if camera_motion > 0.001:
                scale = real_world_motion / camera_motion
                if 0.1 <= scale <= 10.0:
                    confidence = self.calculate_imu_confidence(imu_data, real_world_motion)
                    return {
                        'success': True,
                        'scale': scale,
                        'confidence': confidence
                    }
            return {'success': False, 'error': 'Invalid motion calculation'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    def integrate_imu_motion(self, imu_data, timestamps):
        total_motion = 0.0
        gravity = np.array([0, 0, -9.81])
        for i, imu_frame in enumerate(imu_data):
            if i == 0:
                continue
            dt = (timestamps[i] - timestamps[i-1]) / 1000.0
            accel_data = imu_frame.get('accelerometer', [])
            if accel_data:
                accel_readings = np.array([[d['x'], d['y'], d['z']] for d in accel_data])
                avg_accel = np.mean(accel_readings, axis=0)
                linear_accel = avg_accel - gravity
                motion_magnitude = np.linalg.norm(linear_accel) * dt * dt * 0.5
                total_motion += motion_magnitude
        return total_motion
    def calculate_camera_motion(self, camera_poses):
        if len(camera_poses) < 2:
            return 0.0
        total_motion = 0.0
        for i in range(1, len(camera_poses)):
            motion = np.linalg.norm(
                camera_poses[i][:3, 3] - camera_poses[i-1][:3, 3]
            )
            total_motion += motion
        return total_motion
    def calculate_imu_confidence(self, imu_data, motion):
        base_confidence = 0.8
        if motion < 0.05:
            base_confidence *= 0.5
        elif motion > 5.0:
            base_confidence *= 0.7
        total_readings = sum(len(frame.get('accelerometer', [])) for frame in imu_data)
        if total_readings < 50:
            base_confidence *= 0.6
        return min(base_confidence, 1.0)
    async def estimate_scale_from_metadata(self, metadata, depth_maps):
        try:
            scale_estimates = []
            for i, meta in enumerate(metadata):
                if 'FocalLength' not in meta or 'ImageWidth' not in meta:
                    continue
                focal_length_mm = float(meta['FocalLength'])
                image_width_px = meta['ImageWidth']
                sensor_width_mm = self.estimate_sensor_width(meta)
                fov_radians = 2 * np.arctan(sensor_width_mm / (2 * focal_length_mm))
                if i < len(depth_maps):
                    avg_depth = np.median(depth_maps[i])
                    scale = (image_width_px * avg_depth * np.tan(fov_radians / 2)) / (sensor_width_mm / 1000)
                    if 0.1 <= scale <= 10.0:
                        scale_estimates.append(scale)
            if scale_estimates:
                final_scale = np.median(scale_estimates)
                confidence = 0.7 * (1.0 - np.std(scale_estimates) / np.mean(scale_estimates))
                return {
                    'success': True,
                    'scale': final_scale,
                    'confidence': max(0.1, confidence)
                }
            return {'success': False, 'error': 'No valid metadata estimates'}
        except Exception as e:
            return {'success': False, 'error': str(e)}
    def estimate_sensor_width(self, metadata):
        sensor_widths = {
            'iPhone': 4.8,
            'Samsung': 5.6,
            'Google': 5.0,
            'OnePlus': 5.4,
            'Xiaomi': 5.2
        }
        device_model = metadata.get('Model', '').lower()
        for brand, width in sensor_widths.items():
            if brand.lower() in device_model:
                return width
        return 5.0
    async def estimate_scale_from_geometry(self, camera_poses, point_cloud):
        try:
            if len(camera_poses) < 2:
                return {'success': False, 'error': 'Insufficient camera poses'}
            baselines = []
            for i in range(len(camera_poses)):
                for j in range(i + 1, len(camera_poses)):
                    baseline = np.linalg.norm(
                        camera_poses[i][:3, 3] - camera_poses[j][:3, 3]
                    )
                    baselines.append(baseline)
            median_baseline = np.median(baselines)
            typical_motion_range = np.array([0.3, 0.8])  # meters
            scale_estimates = typical_motion_range / median_baseline
            scale = np.mean(scale_estimates)
            baseline_std = np.std(baselines)
            baseline_consistency = 1.0 - (baseline_std / median_baseline)
            confidence = 0.6 * max(0.1, baseline_consistency)
            return {
                'success': True,
                'scale': scale,
                'confidence': confidence
            }
        except Exception as e:
            return {'success': False, 'error': str(e)}
    def fuse_scale_estimates(self, scale_estimates, confidences):
        if not scale_estimates:
            return 1.0, 0.0
        total_weight = 0.0
        weighted_sum = 0.0
        for method, scale in scale_estimates.items():
            confidence = confidences.get(method, 0.0)
            base_weight = self.confidence_weights.get(method, 0.1)
            weight = confidence * base_weight
            weighted_sum += scale * weight
            total_weight += weight
        if total_weight > 0:
            final_scale = weighted_sum / total_weight
            final_confidence = total_weight / sum(self.confidence_weights.values())
        else:
            final_scale = 1.0
            final_confidence = 0.0
        return final_scale, min(final_confidence, 1.0)
