import cv2

img= cv2.imread('src/shivji.jpg', 0)

# h,w= img.shape
h=225
w=224

img_resize=cv2.resize(img, (h,w), interpolation=cv2.INTER_CUBIC)

# blur_img = cv2.GaussianBlur(img_resize,ksize=(3,3),sigmaX=0,sigmaY=0)

#sobel edge detection
# sobelx= cv2.Sobel(src=img_resize, ddepth=cv2.CV_64F, dx=1, dy=0, ksize=5)
# sobely= cv2.Sobel(src=img_resize, ddepth=cv2.CV_64F, dx=0, dy=1, ksize=5)



cv2.imshow("shivji",img_resize)


cv2.waitKey()
cv2.destroyAllWindows()

