import sys
import os

# Add current directory to path so we can import processor
sys.path.append(os.getcwd())

from processor import VideoProcessor

def test_obstruction():
    print("Testing Strict Obstruction Logic...")
    processor = VideoProcessor(model_type='s3fd')
    
    img_w = 1000
    img_h = 1000
    
    # Case 1: Fully inside
    # Box: 100, 100 to 200, 200
    box_inside = (100, 100, 200, 200)
    is_obs = processor.check_obstruction(box_inside, img_w, img_h, margin_ratio=0.0)
    print(f"Case 1 (Inside): Obstructed? {is_obs} (Expected: False)")
    assert is_obs == False, "Failed: Fully inside box should not be obstructed"

    # Case 2: Partially outside (touching edge)
    # Box: -10, 100 to 90, 200 (10 pixels out on left)
    box_partial = (-10, 100, 90, 200)
    is_obs = processor.check_obstruction(box_partial, img_w, img_h, margin_ratio=0.0)
    print(f"Case 2 (Partial Out): Obstructed? {is_obs} (Expected: True)")
    assert is_obs == True, "Failed: Partially outside box should be obstructed in strict mode"

    # Case 3: Margin test
    # Box: 10, 100 to 110, 200. 
    # If margin is 5% (50px), this box (starting at 10) is inside the image but outside the safe zone (starts at 50).
    box_margin = (10, 100, 110, 200)
    is_obs = processor.check_obstruction(box_margin, img_w, img_h, margin_ratio=0.05)
    print(f"Case 3 (Margin Violation): Obstructed? {is_obs} (Expected: True)")
    assert is_obs == True, "Failed: Box in margin should be obstructed"

    print("All tests passed!")

if __name__ == "__main__":
    test_obstruction()
