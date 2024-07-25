import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage import morphology
from PIL import Image
import timeit

start = timeit.default_timer()

DOMINANT_COLORS = 50
FILEPATH = "hemp1.jpeg"
NUM_LUMA = 4

def generateOriginalImage():

    im = cv2.imread(FILEPATH) # Reads an image into BGR Format
    im = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)

    plt.imshow(im) # as RGB Format
    plt.suptitle("Original Image")
    plt.imsave('Original.png', im)
    #plt.show()
    plt.close()
    return im

def kMeans(im):
    # Flatten Each channel of the Image
    all_pixels = im.reshape((-1,3))

    km = KMeans(n_clusters=DOMINANT_COLORS)
    km.fit(all_pixels)

    centers = km.cluster_centers_ # In RGB Format
    centers = np.array(centers,dtype='uint8')
    return km, centers

def Colors(centers):
    plt.figure(0,figsize=(8,2))

    # Storing info in color array
    colors = []
    k = 1
    for each_col in centers:
        plt.subplot(1,DOMINANT_COLORS,k)
        plt.axis("off")
        k+=1
        colors.append(each_col)

        # Color Swatch
        colorSwatch = np.zeros((100,100,3),dtype='uint8')
        colorSwatch[:,:,:] = each_col
        plt.imshow(colorSwatch)
    plt.suptitle("Color Swatch")  
    plt.savefig('colorSwatch.png')
    #plt.show()
    plt.close()
    return colors

def initNewImg(im):
    original_shape = im.shape
    new_img = np.zeros((original_shape[0]*original_shape[1],3),dtype='uint8')
    return new_img, original_shape

def getLightestColor(colors):
    luma = 0
    index = 0
    indexTemp = 0
    for color in colors:
        red = color[0]
        green = color[1]
        blue = color[2]
        lumaTemp = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
        if lumaTemp > luma:
            luma = lumaTemp
            index = indexTemp
        indexTemp += 1
    return index

def generateSegmentedImage(new_img):
    segmentedFilepath = 'segmented.png'
    # Iterate over the image
    for ix in range(new_img.shape[0]):
        new_img[ix] = colors[km.labels_[ix]]
        
    new_img = new_img.reshape((original_shape))
    plt.imshow(new_img)
    #plt.suptitle(f"K={DOMINANT_COLORS}")
    plt.axis('off')
    plt.savefig(segmentedFilepath, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()
    return new_img, segmentedFilepath

def generateBWImage(bw_img, km, colors, original_shape):
    BW1Filepath = 'BW1.png'
    grayscaleColors = colors
    for i in range(len(grayscaleColors)):
        if i == lightestIndex:
            grayscaleColors[i] = [255,255,255]
        else:
            grayscaleColors[i] = [0,0,0]

    # Iterate over the image
    for ix in range(bw_img.shape[0]):
        bw_img[ix] = grayscaleColors[km.labels_[ix]]

    bw_img = bw_img.reshape((original_shape))

    plt.imshow(bw_img)
    #plt.suptitle(f"BW, K={DOMINANT_COLORS}")
    plt.axis('off')
    plt.savefig(BW1Filepath, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()
    return bw_img, BW1Filepath

def imageOpening(img, imageNum, times):
    openedFilepath = f'BW{imageNum}.png'
    opened_img = morphology.opening(img)
    plt.imshow(opened_img)
    #plt.suptitle(f"{times}x Opening, K={DOMINANT_COLORS}")
    plt.axis('off')
    #plt.savefig(openedFilepath, bbox_inches='tight')
    plt.savefig(openedFilepath, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()
    return opened_img, openedFilepath

def imageClosing(img, imageNum, times):
    closedFilepath = f'BW{imageNum}.png'
    closedImg = morphology.closing(img)
    plt.imshow(closedImg)
    #plt.suptitle(f"{times}x Closing, K={DOMINANT_COLORS}")
    plt.axis('off')
    #plt.savefig(closedFilepath, bbox_inches='tight')
    plt.savefig(closedFilepath, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()
    return closedImg, closedFilepath

def imageDilation(img, imageNum, times):
    dilatedFilepath = f'BW{imageNum}.png'
    dilatedImg = morphology.dilation(img)
    plt.imshow(dilatedImg)
    #plt.suptitle(f"{times}x Closing, K={DOMINANT_COLORS}")
    plt.axis('off')
    #plt.savefig(closedFilepath, bbox_inches='tight')
    plt.savefig(dilatedFilepath, bbox_inches='tight', pad_inches = 0)
    #plt.show()
    plt.close()
    return dilatedImg, dilatedFilepath

def generateTransparentImage(filepath):
    transparentFilepath = 'transparent_image.png'
    BW = Image.open(filepath)
    BWrgba = BW.convert('RGBA')
    datas = BWrgba.getdata()
    newData = []
    for item in datas:
        if item[0] == 0 and item[1] == 0 and item[2] == 0:
            newData.append((255, 255, 255, 0))
        else:
            newData.append(item)

    BWrgba.putdata(newData)
    BWrgba.save(transparentFilepath, 'PNG')
    return transparentFilepath

def overlayImages(segmentedFilepath, transparentFilepath):
    overlayFilepath = 'overlayed.png'
    '''
    background = cv2.imread(segmentedFilepath)
    overlay = cv2.imread(transparentFilepath)

    #dst = cv2.addWeighted(background,0.5,overlay,0.5,0)
    dst = cv2.addWeighted(overlay,0.6,background,0.4,0)
    cv2.imshow('dst',dst)
    # Saving the image 
    cv2.imwrite('overlay.png', dst) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    '''
    background = cv2.imread(segmentedFilepath, cv2.IMREAD_UNCHANGED)
    foreground = cv2.imread(transparentFilepath, cv2.IMREAD_UNCHANGED)

    # normalize alpha channels from 0-255 to 0-1
    alpha_background = background[:,:,3] / 255.0
    alpha_foreground = foreground[:,:,3] / 255.0

    # set adjusted colors
    for color in range(0, 3):
        background[:,:,color] = alpha_foreground * foreground[:,:,color] + \
            alpha_background * background[:,:,color] * (1 - alpha_foreground)

    # set adjusted alpha and denormalize back to 0-255
    background[:,:,3] = (1 - (1 - alpha_foreground) * (1 - alpha_background)) * 255

    # display the image
    #cv2.imshow("Composited image", background)
    cv2.imwrite(overlayFilepath, background) 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return overlayFilepath

def findBlobs(img_filepath):
    cannyFilepath = 'Canny_Edges_After_Contouring.png'
    contourFilepath = 'Contours.png'
    img = cv2.imread(img_filepath)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edged = cv2.Canny(imgray, 30, 200) 
    #contours, hierarchy = cv2.findContours(edged,cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    contours, hierarchy = cv2.findContours(edged,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.imwrite(cannyFilepath, edged) 
    print("Number of Contours found = " + str(len(contours))) 
    cv2.drawContours(img, contours, -1, (255, 255, 255), thickness=cv2.FILLED) 
    cv2.imwrite(contourFilepath, img) 
    cv2.destroyAllWindows()
    return cannyFilepath, contourFilepath

im = generateOriginalImage()
km, centers = kMeans(im)
colors = Colors(centers)
new_img, original_shape = initNewImg(im)
lightestIndex = getLightestColor(colors)
bw_img = new_img
new_img, segmentedFilepath = generateSegmentedImage(new_img)
bw_img, BW1Filepath = generateBWImage(bw_img, km, colors, original_shape)
opened_img1x, openedFilepath1x = imageOpening(bw_img, 2, 1)
closed_img1x, closedFilepath1x = imageClosing(opened_img1x, 3, 1)
closed_img2x, closedFilepath2x = imageClosing(closed_img1x, 4, 2)
closed_img3x, closedFilepath3x = imageClosing(closed_img1x, 5, 3)
dilated_img1x, dilatedFilepath1x = imageDilation(closed_img1x, 6, 1)
dilated_img2x, dilatedFilepath2x = imageDilation(closed_img1x, 7, 2)
dilated_img3x, dilatedFilepath3x = imageDilation(closed_img1x, 8, 3)
dilated_img4x, dilatedFilepath4x = imageDilation(closed_img1x, 9, 4)
#transparentFilepath = generateTransparentImage(closedFilepath2x)
#transparentFilepath = generateTransparentImage(dilatedFilepath4x)
#overlayFilepath = overlayImages(segmentedFilepath, transparentFilepath)
cannyFilepath, contourFilepath = findBlobs(dilatedFilepath4x)
transparentFilepath = generateTransparentImage(contourFilepath)
overlayFilepath = overlayImages(segmentedFilepath, transparentFilepath)


stop = timeit.default_timer()
print('Runtime: ', round((stop - start), 2))