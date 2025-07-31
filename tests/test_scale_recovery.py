import pytest
import numpy as np
from server.models.scale_recovery import MultiMethodScaleRecovery

class TestScaleRecovery:
    def setup_method(self):
        """Setup test fixtures"""
        self.scale_recovery = MultiMethodScaleRecovery()
        
        # Create mock data
        self.mock_imu_data = [
            {
                'accelerometer': [
                    {'x': 0.1, 'y': 0.2, 'z': 9.8},
                    {'x': 0.15, 'y': 0.25, 'z': 9.9}
                ],
                'gyroscope': [
                    {'x': 0.01, 'y': 0.02, 'z': 0.01}
                ]
            }
        ] * 10
        
        self.mock_timestamps = list(range(0, 10000, 1000))  # 10 timestamps, 1 second apart
        
        self.mock_camera_poses = [
            np.eye(4),
            np.array([[1, 0, 0, 0.1],
                     [0, 1, 0, 0.2],
                     [0, 0, 1, 0.3],
                     [0, 0, 0, 1]])
        ]
        
        self.mock_metadata = [
            {
                'FocalLength': 4.25,
                'ImageWidth': 3072,
                'ImageHeight': 4096,
                'Model': 'iPhone 12'
            }
        ]
    
    def test_imu_motion_integration(self):
        """Test IMU motion integration"""
        motion = self.scale_recovery.integrate_imu_motion(
            self.mock_imu_data, 
            self.mock_timestamps
        )
        
        assert isinstance(motion, float)
        assert motion >= 0
    
    def test_camera_motion_calculation(self):
        """Test camera motion calculation"""
        motion = self.scale_recovery.calculate_camera_motion(self.mock_camera_poses)
        
        assert isinstance(motion, float)
        assert motion > 0  # Should detect motion between poses
    
    def test_imu_confidence_calculation(self):
        """Test IMU confidence calculation"""
        confidence = self.scale_recovery.calculate_imu_confidence(
            self.mock_imu_data, 
            0.5  # Mock motion value
        )
        
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1
    
    @pytest.mark.asyncio
    async def test_imu_scale_estimation(self):
        """Test IMU-based scale estimation"""
        result = await self.scale_recovery.estimate_scale_from_imu(
            self.mock_imu_data,
            self.mock_camera_poses,
            self.mock_timestamps
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'scale' in result
            assert 'confidence' in result
            assert isinstance(result['scale'], float)
            assert isinstance(result['confidence'], float)
    
    def test_sensor_width_estimation(self):
        """Test sensor width estimation"""
        width = self.scale_recovery.estimate_sensor_width(self.mock_metadata[0])
        
        assert isinstance(width, float)
        assert width > 0
        # Should return iPhone sensor width
        assert width == 4.8
    
    @pytest.mark.asyncio
    async def test_metadata_scale_estimation(self):
        """Test metadata-based scale estimation"""
        mock_depth_maps = [np.ones((100, 100)) * 2.0]  # Mock depth map
        
        result = await self.scale_recovery.estimate_scale_from_metadata(
            self.mock_metadata,
            mock_depth_maps
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
    
    @pytest.mark.asyncio
    async def test_geometric_scale_estimation(self):
        """Test geometric scale estimation"""
        mock_point_cloud = np.random.rand(1000, 3)
        
        result = await self.scale_recovery.estimate_scale_from_geometry(
            self.mock_camera_poses,
            mock_point_cloud
        )
        
        assert isinstance(result, dict)
        assert 'success' in result
        
        if result['success']:
            assert 'scale' in result
            assert 'confidence' in result
    
    def test_scale_fusion(self):
        """Test scale estimate fusion"""
        scale_estimates = {
            'imu': 1.2,
            'metadata': 1.1,
            'geometric': 1.3
        }
        
        confidences = {
            'imu': 0.8,
            'metadata': 0.6,
            'geometric': 0.5
        }
        
        final_scale, final_confidence = self.scale_recovery.fuse_scale_estimates(
            scale_estimates, confidences
        )
        
        assert isinstance(final_scale, float)
        assert isinstance(final_confidence, float)
        assert final_scale > 0
        assert 0 <= final_confidence <= 1
    
    def test_empty_scale_fusion(self):
        """Test scale fusion with empty estimates"""
        final_scale, final_confidence = self.scale_recovery.fuse_scale_estimates({}, {})
        
        assert final_scale == 1.0
        assert final_confidence == 0.0

if __name__ == "__main__":
    pytest.main([__file__])
