# -*- coding: utf-8 -*-
"""
Created on Sun Jul  4 00:41:42 2021

@author: Wisit
"""
import cv2
import face_recognition
import numpy as np
import matplotlib.pyplot as plt


def detection_circle(canny_img, cut_img):
    print("start")
    print(canny_img.shape[0])
    print(canny_img.shape[1])
    size_zoomin =5
    size_zoomout =0.2
    if canny_img.shape[0]<100:
        size_zoomin=10
        size_zoomout=0.1
    canny_img = cv2.resize(canny_img, (0,0),None,size_zoomin,size_zoomin)
    cut_img = cv2.resize(cut_img, (0,0),None,size_zoomin,size_zoomin)
    print("canny chage")
    print(canny_img.shape[0])
    print(canny_img.shape[1])
    plt.imshow(cut_img)
    plt.show()
    plt.imshow(cv2.cvtColor(canny_img, cv2.COLOR_GRAY2BGR))
    plt.show()
    plt.imshow(cv2.cvtColor(cut_img, cv2.COLOR_BGR2RGB))
    plt.show()
    circles = cv2.HoughCircles(canny_img, cv2.HOUGH_GRADIENT, 
                               dp=5, minDist=canny_img.shape[0], 
                               minRadius=int((canny_img.shape[0])/8), 
                               maxRadius=int((canny_img.shape[0])/2.5))
    print("circles")
    print(circles[0])
   
    eye_circles = np.copy(cut_img)
    
    x, y, r = 0, 0, 0
    if circles is not None and len(circles) > 0:
        print("found_circle")
        circles = circles[0]
        for (x, y, r) in circles:
            x, y, r = int(x), int(y), int(r)
            print("xr=")
            print( x)
            print("yr=")
            print(y)
            print(r)
            eye_circles=cv2.circle(eye_circles, (x, y), r, (255, 255, 0), 4)
            print("draw_circle")
        plt.imshow(cv2.cvtColor(eye_circles, cv2.COLOR_BGR2RGB))
        plt.show()
        
    print("fisnidetection_circle")
    
    
    x=int(x*size_zoomout)
    y=int(y*size_zoomout)
    r=int(r*size_zoomout)
    print(x)
    print(y)
    print(r)
    return x, y, r

def skin(img_skin, x_left,y_left, x_rigth, y_rigth):
    x = (x_left + x_rigth) / 2
    y = (y_left + y_rigth) / 2
    
    cv2.imshow("im_skin",img_skin)

    color = [[0 for i in range(3)] for j in range(9)]
    sum = [[0 for i in range(2)] for j in range(9)]

    for i in range(3):
        for j in range(3):
            idx = int(i * 3 + j)
            for k in range(3):
                color[idx][k] = int(img_skin[int(y + (-10 + i * 10)), int(x + (-10 + j * 10)), k])
            sum[idx][0] = color[idx][0] + color[idx][1] + color[idx][2]
            sum[idx][1] = idx
    print("color")    
    print(color) 
    print("sum")   
    print(sum)      
    sum.sort()
    print("sum after sort") 
    print(sum) 
    print("color[sum[0][1]]") 
    print(color[sum[0][1]]) 
    return color[sum[0][1]]

def set(img_change, h, w, x_pos_eye ,y_pos_eye ,modelrgb, skin,trans):
    
    alpha = modelrgb[:,:,3] # extract it
    cv2.imshow("alpha",alpha)
    binary = ~alpha   # invert b/w
    cv2.imshow('invert',img_change)
    for y in range(modelrgb.shape[1]):
        for x in range(modelrgb.shape[0]):
            img_y = int(y_pos_eye - modelrgb.shape[1] / 2 + y)
            img_x = int(x_pos_eye - modelrgb.shape[0] / 2 + x)
          
            
            if img_y > 0 and img_y < h and img_x > 0 and img_y < w:
                red = img_change[img_y, img_x, 2]
                green = img_change[img_y, img_x, 1]
                blue = img_change[img_y, img_x, 0]
                
                if binary[y][x]==0:
                    img_change[img_y, img_x, 0] = int(modelrgb[y,x, 0]*trans)
                    img_change[img_y, img_x, 1] = int(modelrgb[y,x, 1]*trans)
                    img_change[img_y, img_x, 2] = int(modelrgb[y,x, 2]*trans)
# =============================================================================
#                             print( "img[img_y, img_x, 3]")
#                             print( img_change[img_y, img_x, 3])
# =============================================================================
                            

    return

# Load the jpg file into a numpy array


img = cv2.imread("cony.jpg")



print("type(img)")
print(type(img))
imgOriginal = img.copy()
rgba_image = img.copy()
plt.imshow(img)




# =============================================================================
# for y in range(rgba_image.shape[1]):
#     for x in range(rgba_image.shape[0]):
#             print("****************************************************")
#             print(y)
#             print(x)                
#             print(rgba_image[y, x, 0])             
#             print(rgba_image[y, x, 1])
#             print(rgba_image[y, x, 2])
#             print(rgba_image[y, x, 3])
# =============================================================================
            


model = cv2.imread("model/anime_6.png",cv2.IMREAD_UNCHANGED)





# =============================================================================
# img = cv2.resize(img, (0,0),None,1,1)
# =============================================================================

img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
print("type(img)")
print(type(img_gray))
cv2.imshow("img_gray",img_gray)
print("img_gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)/////////////////////////////////////////////")
# Find all facial features in all the faces in the image

face_landmarks_list = face_recognition.face_landmarks(img_gray)

mask2 = np.zeros_like(img_gray)
for face_landmarks in face_landmarks_list:
    print( face_landmarks['left_eye'] )
    print( face_landmarks['right_eye'] )
    
    print( face_landmarks['left_eye'][0] )
    print( face_landmarks['left_eye'][0][0] )
    print( face_landmarks['left_eye'][0][1] )
    
    # draw mask
# =============================================================================
#     points = np.array(face_landmarks['left_eye'],np.int32)
#     convexhull = cv2.convexHull(points)
#     cv2.polylines(imgOriginal, [convexhull], True,(255,0,0),1)
#     cv2.fillConvexPoly(mask2, convexhull, 255)
#     
#     points = np.array(face_landmarks['right_eye'],np.int32)
#     convexhull = cv2.convexHull(points)
#     cv2.polylines(imgOriginal, [convexhull], True,(255,0,0),1)
#     cv2.fillConvexPoly(mask2, convexhull, 255)
# =============================================================================
    
    #crop image left

    x_min_l=min([ x[0] for x in face_landmarks['left_eye']])
    x_max_l=max([ x[0] for x in face_landmarks['left_eye']])
    y_min_l=min([ x[1] for x in face_landmarks['left_eye']])
    y_max_l=max([ x[1] for x in face_landmarks['left_eye']])
    print("min Max")
    print(x_min_l)
    print(x_max_l)
    print(y_min_l)
    print(y_max_l)
        
    border_img_crop = face_landmarks['left_eye'][1][0]-face_landmarks['left_eye'][0][0];
    
    print("border_img_crop")
    print(border_img_crop)

    cut_img_l = imgOriginal[y_min_l-border_img_crop:y_max_l+border_img_crop,x_min_l-border_img_crop:x_max_l+border_img_crop]
    print('cut_img_l')
    print(cut_img_l.shape[0])
    print(cut_img_l.shape[1])
    cv2.imshow("cut_img", cut_img_l)
   
    #cut_img_l = cv2.resize(cut_img_l, (0,0),None,10,10)
    img_preprocessed_l = cv2.cvtColor(cv2.GaussianBlur(cut_img_l, (9, 9), 0), cv2.COLOR_BGR2GRAY)
    canny_img_l = cv2.Canny(img_preprocessed_l, threshold1=50, threshold2=80)
    cv2.imshow("canny_img_l", canny_img_l)

    
    xl, yl, length_l = detection_circle(canny_img_l, cut_img_l)
    xl+=x_min_l-border_img_crop
    yl+=y_min_l-border_img_crop
    length_l=int(length_l*2)
    
# =============================================================================
#     xl=int(xl*0.1)
#     yl=int(yl*0.1)
#     length_l=int(length_l*0.1)
# =============================================================================
    
    print("xl")
    print( xl)
    print("yl")
    print(yl)
    print('length_l')
    print(length_l)
    
    print("print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)print(length_l)")
    #crop image_right
    x_min_r=min([ x[0] for x in face_landmarks['right_eye']])
    x_max_r=max([ x[0] for x in face_landmarks['right_eye']])
    y_min_r=min([ x[1] for x in face_landmarks['right_eye']])
    y_max_r=max([ x[1] for x in face_landmarks['right_eye']])
    print("min Max")
    print(x_min_r)
    print(x_max_r)  
    print(y_min_r)
    print(y_max_r)
        
    border_img_crop = face_landmarks['right_eye'][1][0] - face_landmarks['right_eye'][0][0];
    
    cut_img_r = imgOriginal[y_min_r-border_img_crop:y_max_r+border_img_crop,x_min_r-border_img_crop:x_max_r+border_img_crop]
    print('cut_img_r')
    print(cut_img_r.shape[0])
    print(cut_img_r.shape[1])
    cv2.imshow("cut_img-R", cut_img_r)
   
    #cut_img_r = cv2.resize(cut_img_r, (0,0),None,10,10)
    img_preprocessed_r = cv2.cvtColor(cv2.GaussianBlur(cut_img_r, (9, 9), 0), cv2.COLOR_BGR2GRAY)
    canny_img_r = cv2.Canny(img_preprocessed_r, threshold1=50, threshold2=80)
    cv2.imshow("canny_img_r", canny_img_r)
    
    xr, yr, length_r = detection_circle(canny_img_r, cut_img_r)
   
    xr+=x_min_r-border_img_crop
    yr+=y_min_r-border_img_crop
    length_r=int(length_r*2)
    
# =============================================================================
#     xr=int(xr*0.1)
#     yr=int(yr*0.1)
#     length_r=int(length_r*0.1)
# =============================================================================
    
    print("xr")
    print( xr)
    print("yr")
    print(yr)
    print('length_r')
    print(length_r)
    
    #set skin
    Skin = skin(imgOriginal, xl ,yl, xr, yr)
    print('type( Skin)')
    print(type( Skin))
    print(Skin)
                
  
 
    if length_l>length_r :
        model_rgb_left = cv2.resize(model, (length_l, length_l),interpolation = cv2.INTER_AREA)
        model_rgb_right = cv2.resize(model, (length_l, length_l),interpolation = cv2.INTER_AREA)
    else:
        model_rgb_left = cv2.resize(model, (length_r, length_r),interpolation = cv2.INTER_AREA)
        model_rgb_right = cv2.resize(model, (length_r, length_r),interpolation = cv2.INTER_AREA)
    
    
    print('model_rgb_leftmodel_rgb_leftmodel_rgb_leftmodel_rgb_left')
    
    print(model_rgb_left.shape[0])
    print(model_rgb_left.shape[1])
 
    
    
# =============================================================================
#     for y in range(model_rgb_left.shape[1]):
#         for x in range(model_rgb_left.shape[0]):
#             print("****************************************************")
#             print(y)
#             print(x)   
#             
#             print(model_rgb_left[y, x, 0])
#             print(model_rgb_left[y, x, 1])
#             print(model_rgb_left[y, x, 2])
# =============================================================================
    
 
    

    cv2.imshow(" model_rgb_left", model_rgb_left)
    
    
    cv2.imshow("model_rgb_right",model_rgb_right)
     
    transparent=1
    
    if transparent>0 and transparent<=1:
        set(rgba_image, rgba_image.shape[0], rgba_image.shape[1], xl, yl ,model_rgb_left, Skin,   transparent)
        set(rgba_image, rgba_image.shape[0], rgba_image.shape[1], xr, yr , model_rgb_right, Skin,   transparent)
   
    cv2.imwrite('cups-circles_imgOriginal.jpg',rgba_image)
    cv2.imshow("imgOriginal",cv2.resize(rgba_image, (540, 720)))

    #show mask
# =============================================================================
# cv2.imshow("outoutout",imgOriginal)
# cv2.imshow("mask2",mask2)
# =============================================================================


    
cv2.waitKey(0)
cv2.destroyAllWindows()

