
import unittest
from processor import VideoProcessor

class TestObstruction(unittest.TestCase):
    def setUp(self):
        self.processor = VideoProcessor(model_type='s3fd') # Model type doesn't matter for this test
        self.img_w = 1000
        self.img_h = 1000

    def test_negative_margin(self):
        # Box touching the edge: x1=0
        box = (0, 200, 100, 300)
        
        # With 0 margin, this should be valid (obstruction ratio 0 if it's just touching? 
        # Wait, check_obstruction logic:
        # margin_w = 0
        # safe_x1 = 0
        # inter_x1 = max(0, 0) = 0
        # inter_x2 = min(100, 1000) = 100
        # inter_w = 100
        # obstructed = 0
        # So at 0 margin, touching edge is fine.
        
        # Let's try a box partially OUTSIDE the image (which shouldn't happen with valid detections usually, but logic should handle it)
        # Or let's say we have a positive margin first.
        
        # 10% margin -> safe zone 100 to 900
        # Box at 50 (center) width 100 -> x1=0, x2=100
        # safe_x1 = 100
        # inter_x1 = 100
        # inter_x2 = 100
        # inter_w = 0
        # obstructed = 100% -> True (Obstructed)
        self.assertTrue(self.processor.check_obstruction(box, self.img_w, self.img_h, margin_ratio=0.1))

        # Now with negative margin -10% -> safe zone -100 to 1100
        # Box at 0-100
        # safe_x1 = -100
        # inter_x1 = max(0, -100) = 0
        # inter_x2 = min(100, 1100) = 100
        # inter_w = 100
        # obstructed = 0 -> False (Not obstructed)
        self.assertFalse(self.processor.check_obstruction(box, self.img_w, self.img_h, margin_ratio=-0.1))

    def test_box_partially_out(self):
        # Box -50 to 50 (center 0)
        box = (-50, 200, 50, 300)
        # Area = 100 * 100 = 10000
        
        # Margin 0
        # safe 0-1000
        # inter_x1 = 0
        # inter_x2 = 50
        # inter_w = 50
        # obstructed = 5000 (50%)
        # > 0 -> True
        self.assertTrue(self.processor.check_obstruction(box, self.img_w, self.img_h, margin_ratio=0.0))
        
        # Margin -0.1 (-100 to 1100)
        # inter_x1 = max(-50, -100) = -50
        # inter_x2 = min(50, 1100) = 50
        # inter_w = 100
        # obstructed = 0 -> False
        self.assertFalse(self.processor.check_obstruction(box, self.img_w, self.img_h, margin_ratio=-0.1))

if __name__ == '__main__':
    unittest.main()
