import os
import xml.etree.ElementTree as ET

# Directory containing annotation files
ANNOTATIONS_DIR = "data/cat_dog_dataset/annotations"

def check_annotations():
    for filename in os.listdir(ANNOTATIONS_DIR):
        if filename.endswith(".xml"):
            filepath = os.path.join(ANNOTATIONS_DIR, filename)
            tree = ET.parse(filepath)
            root = tree.getroot()
            
            # Find all bounding boxes
            bboxes = root.findall(".//bndbox")
            
            # Check if there are more than 1 bounding box
            if len(bboxes) > 1:
                print(f"File with more than 1 bbox: {filepath}")
            
            # # Check for bounding boxes with height or width < 112
            # for bbox in bboxes:
            #     xmin = int(bbox.find("xmin").text)
            #     ymin = int(bbox.find("ymin").text)
            #     xmax = int(bbox.find("xmax").text)
            #     ymax = int(bbox.find("ymax").text)
                
            #     width = xmax - xmin
            #     height = ymax - ymin
                
            #     if width < 112 or height < 112:
            #         print(f"File: {filepath}, bbox with small dimensions: {xmin, ymin, xmax, ymax}")

if __name__ == "__main__":
    check_annotations()