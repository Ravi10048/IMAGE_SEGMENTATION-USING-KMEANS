import numpy as np
import pandas as pd
import cv2
from PIL import Image
import streamlit as st

st.title("IMAGE SEGMENTATION USING KMEANS ALGO:")
k=st.number_input("Enter the value of k")
# uploaded_photo=st.file_uploader("upload photo")
# cv2.imwrite("uploaded.jpg",uploaded_photo)
if(st.button("Enter")):
    show_img=Image.open("flower.jpg")
    # img_resized = show_img.resize((200, 200))
    st.image(show_img)
    img=cv2.imread("flower.jpg")
    print(img.shape)
    img2=img.reshape((-1,3))
    print(img2.shape)
    img2=np.float32(img2)
    criteria=(cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER,10,1.0)
    k=4

    attempts=10

    ret,label,center=cv2.kmeans(img2,k,None,criteria,attempts,cv2.KMEANS_PP_CENTERS)
    centers = np.uint8(center)
    segmented_data = centers[label.flatten()]
    
    # reshape data into the original image dimensions
    segmented_image = segmented_data.reshape((img.shape))
    # segmented_image = show_img.resize((200, 200))

    # cv2.imshow('Window',segmented_image) # nort work
    # cv2.waitKey(0)
    # saving the grayscale image
    cv2.imwrite('segmented.jpg',segmented_image)
    st.write("\n")
    st.write("\n")
    st.write("\n")
    st.image(segmented_image)

    # plt.imshow(segmented_image)
    # cv2.imwrite('segmented.jpg',segmented_image)  
    # plt.show()
    # show_img=Image.open("segmented.jpg")
    # st.image(show_img)
    # img=cv2.imread("cell.jpg")
