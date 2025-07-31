import pytest
import numpy as np
from unittest.mock import Mock, patch
from server.main import DUSt3RDimensionCalculator

class TestDUSt3RProcessor:
    def setup_method(self):
        """Setup test fixtures"""
        # Mock the model loading to avoid actual model dependencies in tests
        with patch('server.main.AsymmetricCroCo3DStereo'):
            self.calculator = DUSt3RDimensionCalculator(device="cpu")
    
    def test_extract_metadata(self):
        """Test metadata extraction"""
        # This would require actual image files, so we'll mock it
        with patch('PIL.Image.open') as mock_open:
            mock_img = Mock()
            mock_img._getexif.return_value = {
                'Model': 'iPhone 12',
                'FocalLength': 4.25
            }
            mock_img.width = 3072
            mock_img.height = 4096
            mock_open.return_value.__enter__.return_value = mock_img
            
            metadata = self.calculator.extract_metadata("fake_path.jpg")
            
            assert isinstance(metadata, dict)
            assert 'Model' in metadata
            assert 'ImageWidth' in metadata
            assert metadata['ImageWidth'] == 3072
    
    def test_extract_depth_maps(self):
        """Test depth map extraction"""
        # Create mock 3D points
        mock_points = [
            torch.tensor(np.random.rand(100, 100, 3)),
            torch.tensor(np.random.rand(100, 100, 3))
        ]
        
        depth_maps = self.calculator.extract_depth_maps(mock_points)
        
        assert isinstance(depth_maps, list)
        assert len(depth_maps) == 2
        assert all(isinstance(dm, np.ndarray) for dm in depth_maps)
    
    def test_calculate_quality_score(self):
        """Test quality score calculation"""
        mock_point_cloud = np.random.rand(5000, 3)
        mock_poses = [np.eye(4) for _ in range(5)]
        
        quality = self.calculator.calculate_quality_score(mock_point_cloud, mock_poses)
        
        assert isinstance(quality, float)
        assert 0 <= quality <= 1
    
    def test_quality_score_insufficient_data(self):
        """Test quality score with insufficient data"""
        small_point_cloud = np.random.rand(100, 3)
        few_poses = [np.eye(4)]
        
        quality = self.calculator.calculate_quality_score(small_point_cloud, few_poses)
        
        assert isinstance(quality, float)
        assert quality < 0.5  # Should be lower quality
    
    @pytest.mark.asyncio
    async def test_calculate_dimensions_insufficient_images(self):
        """Test dimension calculation with insufficient images"""
        with pytest.raises(ValueError, match="At least 2 images required"):
            await self.calculator.calculate_dimensions(["single_image.jpg"])
    
    def test_sanitize_for_json(self):
        """Test JSON sanitization"""
        from server.main import sanitize_for_json
        
        # Test with various data types
        test_data = {
            'string': 'test',
            'number': 42,
            'float': 3.14,
            'list': [1, 2, 3],
            'nested': {
                'inner': 'value'
            }
        }
        
        sanitized = sanitize_for_json(test_data)
        
        assert isinstance(sanitized, dict)
        assert sanitized['string'] == 'test'
        assert sanitized['number'] == 42
        
        # Test JSON serialization works
        import json
        json_str = json.dumps(sanitized)
        assert isinstance(json_str, str)

if __name__ == "__main__":
    pytest.main([__file__])
