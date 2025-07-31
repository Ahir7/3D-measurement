import numpy as np
from scipy.spatial.transform import Rotation
from scipy.signal import butter, filtfilt
from typing import List, Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class SensorFusion:
    def __init__(self, sampling_rate: float = 100.0):
        self.sampling_rate = sampling_rate
        self.gravity = np.array([0, 0, -9.81])
        
    def fuse_imu_data(self, imu_sequence: List[Dict], timestamps: List[float]) -> Dict:
        """
        Advanced IMU sensor fusion with Kalman filtering
        """
        try:
            # Extract and organize sensor data
            accel_data = self._extract_sensor_data(imu_sequence, 'accelerometer')
            gyro_data = self._extract_sensor_data(imu_sequence, 'gyroscope')
            mag_data = self._extract_sensor_data(imu_sequence, 'magnetometer')
            
            # Apply sensor fusion algorithm
            fused_data = self._apply_madgwick_filter(accel_data, gyro_data, mag_data, timestamps)
            
            # Calculate motion metrics
            motion_metrics = self._calculate_motion_metrics(fused_data, timestamps)
            
            return {
                'fused_orientation': fused_data['orientation'],
                'linear_acceleration': fused_data['linear_accel'],
                'motion_metrics': motion_metrics,
                'quality_score': self._assess_data_quality(accel_data, gyro_data)
            }
            
        except Exception as e:
            logger.error(f"IMU sensor fusion failed: {e}")
            return {}
    
    def _extract_sensor_data(self, imu_sequence: List[Dict], sensor_type: str) -> np.ndarray:
        """Extract specific sensor data from IMU sequence"""
        data_list = []
        
        for frame in imu_sequence:
            sensor_readings = frame.get(sensor_type, [])
            if sensor_readings:
                # Average multiple readings within the frame
                frame_data = np.array([[r['x'], r['y'], r['z']] for r in sensor_readings])
                avg_reading = np.mean(frame_data, axis=0)
                data_list.append(avg_reading)
            else:
                # Use zero if no data available
                data_list.append(np.array([0.0, 0.0, 0.0]))
        
        return np.array(data_list)
    
    def _apply_madgwick_filter(self, accel: np.ndarray, gyro: np.ndarray, 
                              mag: np.ndarray, timestamps: List[float]) -> Dict:
        """
        Apply Madgwick AHRS algorithm for sensor fusion
        """
        n_samples = len(accel)
        if n_samples < 2:
            return {'orientation': np.array([]), 'linear_accel': np.array([])}
        
        # Initialize quaternion
        q = np.array([1.0, 0.0, 0.0, 0.0])  # w, x, y, z
        orientations = []
        linear_accels = []
        
        # Madgwick filter parameters
        beta = 0.1  # Algorithm gain
        
        for i in range(n_samples):
            if i == 0:
                dt = 1.0 / self.sampling_rate
            else:
                dt = (timestamps[i] - timestamps[i-1]) / 1000.0  # Convert to seconds
            
            # Current sensor readings
            a = accel[i]
            g = gyro[i]
            m = mag[i] if len(mag) > i else np.array([0.0, 0.0, 0.0])
            
            # Normalize accelerometer and magnetometer
            if np.linalg.norm(a) > 0:
                a = a / np.linalg.norm(a)
            if np.linalg.norm(m) > 0:
                m = m / np.linalg.norm(m)
            
            # Apply Madgwick algorithm
            q = self._madgwick_update(q, g, a, m, dt, beta)
            orientations.append(q.copy())
            
            # Calculate linear acceleration (remove gravity)
            gravity_world = self._rotate_vector_by_quaternion(self.gravity, q)
            linear_accel = accel[i] - gravity_world
            linear_accels.append(linear_accel)
        
        return {
            'orientation': np.array(orientations),
            'linear_accel': np.array(linear_accels)
        }
    
    def _madgwick_update(self, q: np.ndarray, gyro: np.ndarray, accel: np.ndarray, 
                        mag: np.ndarray, dt: float, beta: float) -> np.ndarray:
        """Single step of Madgwick AHRS algorithm"""
        qw, qx, qy, qz = q
        
        # Gyroscope integration
        gx, gy, gz = gyro
        qDot1 = 0.5 * (-qx * gx - qy * gy - qz * gz)
        qDot2 = 0.5 * (qw * gx + qy * gz - qz * gy)
        qDot3 = 0.5 * (qw * gy - qx * gz + qz * gx)
        qDot4 = 0.5 * (qw * gz + qx * gy - qy * gx)
        
        # Accelerometer correction
        if np.linalg.norm(accel) > 0:
            ax, ay, az = accel
            
            # Objective function
            f1 = 2 * (qx * qz - qw * qy) - ax
            f2 = 2 * (qw * qx + qy * qz) - ay
            f3 = 1 - 2 * (qx * qx + qy * qy) - az
            
            # Jacobian
            J11 = -2 * qy
            J12 = 2 * qz
            J13 = -2 * qw
            J14 = 2 * qx
            J21 = 2 * qx
            J22 = 2 * qw
            J23 = 2 * qz
            J24 = 2 * qy
            J31 = 0
            J32 = -4 * qx
            J33 = -4 * qy
            J34 = 0
            
            # Gradient
            step1 = J11 * f1 + J21 * f2 + J31 * f3
            step2 = J12 * f1 + J22 * f2 + J32 * f3
            step3 = J13 * f1 + J23 * f2 + J33 * f3
            step4 = J14 * f1 + J24 * f2 + J34 * f3
            
            # Normalize gradient
            norm = np.sqrt(step1*step1 + step2*step2 + step3*step3 + step4*step4)
            if norm > 0:
                step1 /= norm
                step2 /= norm
                step3 /= norm
                step4 /= norm
            
            # Apply feedback
            qDot1 -= beta * step1
            qDot2 -= beta * step2
            qDot3 -= beta * step3
            qDot4 -= beta * step4
        
        # Integrate quaternion
        q[0] += qDot1 * dt
        q[1] += qDot2 * dt
        q[2] += qDot3 * dt
        q[3] += qDot4 * dt
        
        # Normalize quaternion
        norm = np.linalg.norm(q)
        if norm > 0:
            q /= norm
        
        return q
    
    def _rotate_vector_by_quaternion(self, vector: np.ndarray, q: np.ndarray) -> np.ndarray:
        """Rotate a vector by a quaternion"""
        qw, qx, qy, qz = q
        vx, vy, vz = vector
        
        # Quaternion rotation formula
        rotated_x = vx*(qw*qw + qx*qx - qy*qy - qz*qz) + vy*(2*qx*qy - 2*qw*qz) + vz*(2*qx*qz + 2*qw*qy)
        rotated_y = vx*(2*qx*qy + 2*qw*qz) + vy*(qw*qw - qx*qx + qy*qy - qz*qz) + vz*(2*qy*qz - 2*qw*qx)
        rotated_z = vx*(2*qx*qz - 2*qw*qy) + vy*(2*qy*qz + 2*qw*qx) + vz*(qw*qw - qx*qx - qy*qy + qz*qz)
        
        return np.array([rotated_x, rotated_y, rotated_z])
    
    def _calculate_motion_metrics(self, fused_data: Dict, timestamps: List[float]) -> Dict:
        """Calculate motion-related metrics"""
        try:
            linear_accel = fused_data['linear_accel']
            
            if len(linear_accel) < 2:
                return {}
            
            # Calculate velocity by integrating acceleration
            velocities = []
            positions = []
            
            velocity = np.array([0.0, 0.0, 0.0])
            position = np.array([0.0, 0.0, 0.0])
            
            for i in range(1, len(linear_accel)):
                dt = (timestamps[i] - timestamps[i-1]) / 1000.0
                
                # Simple integration (could be improved with better methods)
                velocity += linear_accel[i] * dt
                position += velocity * dt
                
                velocities.append(velocity.copy())
                positions.append(position.copy())
            
            total_distance = np.linalg.norm(positions[-1]) if positions else 0.0
            max_velocity = np.max([np.linalg.norm(v) for v in velocities]) if velocities else 0.0
            
            return {
                'total_distance': float(total_distance),
                'max_velocity': float(max_velocity),
                'positions': np.array(positions),
                'velocities': np.array(velocities)
            }
            
        except Exception as e:
            logger.error(f"Motion metrics calculation failed: {e}")
            return {}
    
    def _assess_data_quality(self, accel_data: np.ndarray, gyro_data: np.ndarray) -> float:
        """Assess the quality of IMU data"""
        try:
            quality_score = 1.0
            
            # Check for data completeness
            if len(accel_data) == 0 or len(gyro_data) == 0:
                return 0.0
            
            # Check for excessive noise
            accel_std = np.std(accel_data, axis=0)
            gyro_std = np.std(gyro_data, axis=0)
            
            # Penalize high noise
            if np.mean(accel_std) > 5.0:  # High accelerometer noise
                quality_score *= 0.7
            if np.mean(gyro_std) > 2.0:  # High gyroscope noise
                quality_score *= 0.8
            
            # Check for data gaps or outliers
            accel_outliers = np.sum(np.abs(accel_data) > 50, axis=1)  # Extreme accelerations
            if np.sum(accel_outliers) > len(accel_data) * 0.1:  # More than 10% outliers
                quality_score *= 0.6
            
            return max(0.0, min(1.0, quality_score))
            
        except Exception as e:
            logger.error(f"Data quality assessment failed: {e}")
            return 0.5
