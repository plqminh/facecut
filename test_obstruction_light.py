import sys

def check_obstruction(box, img_w, img_h, margin_ratio=0.0):
    """
    Check if the bounding box is significantly obstructed by the frame edge.
    Box format: x1, y1, x2, y2
    margin_ratio: float 0.0-1.0, fraction of dimension to use as margin.
    Returns True if obstructed (>0% area in margin/out of bounds), False otherwise.
    """
    x1, y1, x2, y2 = box
    box_w = x2 - x1
    box_h = y2 - y1
    box_area = box_w * box_h
    
    if box_area <= 0:
        return True

    # Define Safe Zone
    margin_w = int(img_w * margin_ratio)
    margin_h = int(img_h * margin_ratio)
    
    safe_x1 = margin_w
    safe_y1 = margin_h
    safe_x2 = img_w - margin_w
    safe_y2 = img_h - margin_h
    
    # Calculate Intersection with Safe Zone
    inter_x1 = max(x1, safe_x1)
    inter_y1 = max(y1, safe_y1)
    inter_x2 = min(x2, safe_x2)
    inter_y2 = min(y2, safe_y2)
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    # Calculate Obstruction Ratio
    # Obstruction is the part of the box NOT in the intersection
    obstructed_area = box_area - inter_area
    obstruction_ratio = obstructed_area / box_area
    
    # Strict: any obstruction is too much
    if obstruction_ratio > 0.0:
        return True
        
    return False

def test_obstruction():
    print("Testing Strict Obstruction Logic (Lightweight)...")
    
    img_w = 1000
    img_h = 1000
    
    # Case 1: Fully inside
    box_inside = (100, 100, 200, 200)
    is_obs = check_obstruction(box_inside, img_w, img_h, margin_ratio=0.0)
    print(f"Case 1 (Inside): Obstructed? {is_obs} (Expected: False)")
    assert is_obs == False, "Failed: Fully inside box should not be obstructed"

    # Case 2: Partially outside (touching edge)
    box_partial = (-10, 100, 90, 200)
    is_obs = check_obstruction(box_partial, img_w, img_h, margin_ratio=0.0)
    print(f"Case 2 (Partial Out): Obstructed? {is_obs} (Expected: True)")
    assert is_obs == True, "Failed: Partially outside box should be obstructed in strict mode"

    # Case 3: Margin test
    box_margin = (10, 100, 110, 200)
    is_obs = check_obstruction(box_margin, img_w, img_h, margin_ratio=0.05)
    print(f"Case 3 (Margin Violation): Obstructed? {is_obs} (Expected: True)")
    assert is_obs == True, "Failed: Box in margin should be obstructed"

    print("All tests passed!")

if __name__ == "__main__":
    test_obstruction()
