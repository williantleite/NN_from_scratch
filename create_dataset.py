import os
import cv2
import numpy as np
def create_dataset(img_folder,IMG_WIDTH=32,IMG_HEIGHT=32):
    """
    Takes the path to the folder containing all images.
    Returns a list of images and a list of classes.
    """
    img_data_array=[]
    class_name=[]
    for dir1 in os.listdir(img_folder):
        for file in os.listdir(os.path.join(img_folder,dir1)):
            image_path= os.path.join(img_folder,dir1,file)
            image= cv2.imread(image_path,cv2.COLOR_BGR2RGB)#For grayscale use cv2.IMREAD_GRAYSCALE
            if image is None:
                continue
            if len(image.shape) ==2: #This effectively removes the grayscale images since they do not have a third dimension
                continue
            image=cv2.resize(image,(IMG_HEIGHT, IMG_WIDTH),interpolation=cv2.INTER_AREA)
            image=np.array(image)
            image=image.astype('float32')
            image/=255 
            img_data_array.append(image)
            class_name.append(dir1)
    return img_data_array, class_name
