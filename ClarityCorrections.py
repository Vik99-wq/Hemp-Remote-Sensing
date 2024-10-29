import cv2
import numpy as np
import glob
from natsort import natsorted

imageNames = []
for img in natsorted(glob.glob('TrainingImagesSet2/*.jpg')):   
    imageNames.append(img.split("\\",1)[1])

for imageName in imageNames:
    image = cv2.imread(f'TrainingImagesSet2/{imageName}')

    # Create a black image of the same size
    black_image = np.zeros_like(image)

    # Darkening factor (0.8 means 80% original image, 20% black)
    darkening_factor = 0.8

    # Blend the original image with the black image
    darkened_image = cv2.addWeighted(image, darkening_factor, black_image, 1 - darkening_factor, 0)

    # Save the result
    cv2.imwrite(f'DarkenedImages/{imageName}_DARKENED.jpg', darkened_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()