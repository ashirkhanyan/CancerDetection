#!/usr/bin/env python
# coding: utf-8

# In[ ]:


def get_box_parameters(box):
    x1, y1 = box[0]  # Top-left corner coordinates
    x2, y2 = box[1]  # Bottom-right corner coordinates

    # Calculate box parameters
    x_center = (x1 + x2) / 2
    y_center = (y1 + y2) / 2
    width = x2 - x1
    height = y2 - y1

    return x_center, y_center, width, height

# Given bounding box coordinates
#box = [[219.24291497975707, 131.39676113360323],[531.7935222672064, 298.1983805668016]]

# Get box parameters
#x_center, y_center, width, height = get_box_parameters(box)

# Print the results
#print("x_center:", x_center)
#print("y_center:", y_center)
#print("width:", width)
#print("height:", height)

