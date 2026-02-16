import numpy as np

from src.utils.capture_quality import analyze_capture_quality


def test_capture_quality_output_shape(tmp_path):
    image_paths = []
    for index in range(3):
        image = np.full((120, 160, 3), 110 + index * 15, dtype=np.uint8)
        path = tmp_path / f"img_{index:02d}.jpg"
        import cv2
        cv2.imwrite(str(path), image)
        image_paths.append(path)

    report = analyze_capture_quality(image_paths)

    assert "summary" in report
    assert "images" in report
    assert "overlap_scores" in report
    assert report["summary"]["num_images"] == 3
    assert len(report["images"]) == 3
    assert len(report["overlap_scores"]) == 2
    assert 0.0 <= report["summary"]["quality_score"] <= 1.0
