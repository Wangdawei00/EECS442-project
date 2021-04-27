import numpy as np
import cv2

def main():

    # Get Test Images  
    img_list = ["9.png", "10.png", "20.png", "72.png"]

    # Begin testing loop 
    print("Begin Test Loop ...")
    
    for img_name in img_list:

        img = cv2.imread(img_name,cv2.IMREAD_GRAYSCALE)
        img = cv2.applyColorMap(img, cv2.COLORMAP_JET)
        cv2.imwrite(img_name.split(".")[0]+"_recolor.png", img)

        print("Recoloring {} done.".format(img_name))


if __name__ == "__main__":
    main()