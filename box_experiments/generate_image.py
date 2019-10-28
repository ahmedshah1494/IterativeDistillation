import cv2
import numpy as np 
import matplotlib.pyplot as plt
def generate_box_in_box(size=257):
    canvas = np.zeros((size,size,3), dtype='uint8')
    center = size//2
    diag_len = size
    outer_pts = [
            [center,center+diag_len//2],            
            [center+diag_len//2,center],
            [center,center-diag_len//2],
            [center-diag_len//2,center]
    ]            
    outer_pts = np.array(outer_pts)
    inner_pts = [(outer_pts[i]+outer_pts[(i+1)%len(outer_pts)])/2 for i in range(len(outer_pts))]

    outer_pts = np.array(outer_pts, dtype='int32').reshape(-1,1,2)
    inner_pts = np.array(inner_pts, dtype='int32').reshape(-1,1,2)
    
    cv2.fillPoly(canvas, [outer_pts], (255,255,255))
    cv2.fillPoly(canvas, [inner_pts], (0,0,0))    

    plt.imsave('box_in_box.png', canvas)

generate_box_in_box()
