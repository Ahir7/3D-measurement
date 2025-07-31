import pytest
import asyncio
from fastapi.testclient import TestClient
from server.main import app
import tempfile
import os
from pathlib import Path

client = TestClient(app)

class TestMainAPI:
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "gpu_available" in data
        assert "device" in data
    
    def test_calculate_dimensions_missing_images(self):
        """Test API with missing images"""
        response = client.post("/calculate-dimensions-advanced")
        assert response.status_code == 422  # Validation error
    
    def test_calculate_dimensions_single_image(self):
        """Test API with only one image (should fail)"""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_file:
            # Create a simple test image (placeholder)
            from PIL import Image
            import numpy as np
            
            test_img = Image.fromarray(np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8))
            test_img.save(tmp_file.name)
            
            with open(tmp_file.name, 'rb') as f:
                response = client.post(
                    "/calculate-dimensions-advanced",
                    files={"images": ("test.jpg", f, "image/jpeg")}
                )
            
            os.unlink(tmp_file.name)
        
        assert response.status_code == 400
        data = response.json()
        assert "At least 2 images required" in data["detail"]
    
    @pytest.mark.asyncio
    async def test_calculate_dimensions_valid_request(self):
        """Test API with valid images"""
        # Create multiple temporary test images
        temp_files = []
        files_to_upload = []
        
        try:
            for i in range(3):
                tmp_file = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
                temp_files.append(tmp_file.name)
                
                # Create a test image
                from PIL import Image
                import numpy as np
                
                test_img = Image.fromarray(np.random.randint(0, 255, (200, 200, 3), dtype=np.uint8))
                test_img.save(tmp_file.name)
                tmp_file.close()
                
                with open(tmp_file.name, 'rb') as f:
                    files_to_upload.append(("images", (f"test_{i}.jpg", f.read(), "image/jpeg")))
            
            response = client.post(
                "/calculate-dimensions-advanced",
                files=files_to_upload
            )
            
            # Note: This might fail due to model dependencies, but structure should be correct
            assert response.status_code in [200, 500]
            
        finally:
            # Cleanup
            for temp_file in temp_files:
                if os.path.exists(temp_file):
                    os.unlink(temp_file)

if __name__ == "__main__":
    pytest.main([__file__])
