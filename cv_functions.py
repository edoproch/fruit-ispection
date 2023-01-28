
import cv2 
import numpy as np
from scipy.spatial.distance import mahalanobis, euclidean
import matplotlib.pyplot as plt



#Function used to plot the gray scale level of the input images
def plot_hists(img, figsize=(7,15), bins=256):
    plt.figure(figsize = figsize)
    for i in range(len(img)):
        plt.subplot(len(img),1,i+1)
        plt.hist(img[i].flatten(),bins=bins);
        plt.title(f"img{i+1} gray-level histogram")
    plt.show();

#Function used to plot the images, for example the binary masks or the fruits
def show_fruit(first_group, x_size, y_size, title = '', second_group = None):
    cols = 1 if second_group is None else 2
    i=0
    j = 0
    fig = plt.figure(figsize=(x_size,y_size))
    fig.suptitle(title)
    while j < len(first_group):
        if cols == 2:
            plt.subplot(len(first_group), cols,i+1)
            plt.imshow(first_group[j], cmap='gray')
            plt.axis('off')
            plt.subplot(len(second_group), cols,i+2)
            plt.imshow(second_group[j])
            i += 2
        else:
            plt.subplot(cols, len(first_group),i+1)
            plt.imshow(first_group[j], cmap='gray')
            i+=1
        plt.axis('off')
        j += 1
    plt.show()

#This function threshold the given images and return a list of binary masks (one for each input images)
def thresholding(img, th=-1, filter_val = None):
    """
    USAGE
    input_imgs: list of input images
    th: threshold value, pass an integer>=0 to use the binary thresholding with th as threshold value,
        pass -1 to use asaptive thresholding,
        pass -2 to use Otsu's algorithm
    filter_val: pass an integer to appy a bilateral filter with this size for the neighborhood
    """
    thresholded_imgs = np.zeros_like(img)
    if filter_val is not None:
        img = cv2.bilateralFilter(img, filter_val,80,80).astype(np.uint8)
    if th == -1:
        thresholded_imgs = cv2.adaptiveThreshold(img, 1, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 8)
    elif th == -2:
        _, m =  cv2.threshold(img, th, 1, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        thresholded_imgs = m
    else:
        _, m =  cv2.threshold(img, th, 1, cv2.THRESH_BINARY)
        thresholded_imgs = m

    return thresholded_imgs

#Apply the flood fill algorith and return a binary mask containing all the identified holes
def apply_flood_fil(img):
    detected_holes = img.copy()
    mask = np.zeros((img.shape[0] + 2, img.shape[1] + 2), np.uint8)
    detected_holes = detected_holes.astype(np.uint8)
    cv2.floodFill(detected_holes, mask, (0, 0), 255)
    return detected_holes==0

#Using a bit-wise or, this function
# merges the mask identified by the flood fill algorithm with the one identified by the thresholding 
def get_complete_shape(shapes, thresholded_imgs):
    return [t_img | shapes[i] for i, t_img in enumerate(thresholded_imgs)]

#using a binary mask cut a nir or a colored image from the background simply doing a multiplication 
#between the mask and the image 
def cut_background(img, binary_mask, flag='nir'):
    if flag=='nir': return img * binary_mask 
    elif flag=='rgb': return img * np.repeat(binary_mask[:,:,np.newaxis],3,axis=2)


def get_new_color_space(color_space):
    new_space=''
    if color_space == 'hsv':
        new_space = cv2.COLOR_RGB2HSV
    elif color_space == 'hls':
        new_space = cv2.COLOR_RGB2HLS
    elif color_space == 'luv':
        new_space = cv2.COLOR_RGB2LUV
    elif color_space == 'lab':
        new_space = cv2.COLOR_RGB2LAB
    return new_space

#given some samples and a color space compute the mean anche the covariance matrix
#of the samples
def get_mean_cov(samples, color_space):
    covariance = np.zeros((3, 3), dtype="float64")
    mean = np.zeros((1, 3), dtype="float64")
    for sample in samples:
        if color_space != 'rgb':
            new_space = get_new_color_space(color_space)
            if new_space != '': 
                sample = cv2.cvtColor(sample,new_space)
        sample = sample.reshape(-1, 3)
        cov, m = cv2.calcCovarMatrix(sample, None, cv2.COVAR_NORMAL |  cv2.COVAR_ROWS)
        covariance += cov
        mean += m

    return mean / len(samples), covariance

#given the image produced by the connected component algorithm and a lable, this funcion returns a binary mask which contains
#only the component associated with the given label to which the flood fill algorithm was applied
def isolate_defect(img, num):
    isolated_defect = np.zeros((img.shape[0],img.shape[1]), int)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if img[x][y] == num:
                isolated_defect[x][y] = 1
    isolated_defect = isolated_defect.astype(np.uint8)
    flooded_defect = apply_flood_fil(isolated_defect)
    return flooded_defect.astype(np.uint8) | isolated_defect

#using the 'isolate_defect' function, this procedure returns a binary mask with all the defect identified on a single image
def get_defects(edge):
    num_labels, labels_im = cv2.connectedComponents(edge,connectivity=8)
    defects_img = np.zeros((labels_im.shape[0],labels_im.shape[1]), int)
    areas = []
    single_image_defects = []
    for num in range(1, num_labels):
        defect = isolate_defect(labels_im,num)
        areas.append(cv2.countNonZero(defect))
        single_image_defects.append(defect)
    border_index = np.argmax(areas)
    for i, image_defect in enumerate(single_image_defects):
        #insted of exluding the component with the biggest area we could use a renge to exlude also the defects that are too little since in a real world application colud be allowed
        if i != border_index:
            defects_img |= image_defect
    return defects_img

#This function return a binary mask that highlights the parts in a image which have a color 
#similar to the 'russet_color_sample' parameter
def get_russet_mask(covariance_matrix, color_space, russet_color_sample, img, distance, formula='mahalanobis'):
    """
    USAGE
    covariance_matrix and russet_color_sample: you can use the outputs of get_mean_cov
    color_space: a color space (between rgb, hsv, hls,luv and lab) to use 
    COLOR_imgs: the images
    distance: the threshold to decide if a pixel is sufficently near to the sample
    formula: you can use the mahalanobis or the euclidean
    """
    inverce_cov = np.linalg.inv(covariance_matrix)
    russet_masks = np.zeros((img.shape[0],img.shape[1]))
    if color_space != 'rgb':
        new_space = get_new_color_space(color_space)
        if new_space != '': 
            img = cv2.cvtColor(img,new_space)
    for x in range(img.shape[0]):
        for y in range(img.shape[1]):
            if formula == 'mahalanobis':
                d = mahalanobis(img[x][y], russet_color_sample, inverce_cov)
            elif formula == 'euclidean':
                d = euclidean(img[x][y], russet_color_sample)
            if d<distance:
                russet_masks[x][y] = 255
    return russet_masks

#this function returns the edges of a gray levels image using the Canny edge detector
def get_edges(img, th1, th2, filter_type=None ,filter_val=None, sigma=2, s=80):
    """
    USAGE
    src: the input image
    th1: first threshold
    th2: second threshold
    filter_type: None to don't apply filters to the input image, 
                'bilateral' to use a bilateral filter,
                'gaussian' to use a gaussian filter
    filter_val: the size of the neighborhood for the bilateral 
                filter or the kernel size for the gaussian filter
    sigma: sigmaColor parameter for bilateral filter or sigmaX parameter for gaussian filter
    s =  sigmaSpace parameter for the bilateral filter
    """
    if (filter_val is not None) and (filter_type is not None):
        img = img.astype(np.uint8)
        if filter_type == 'bilateral':
            img = cv2.bilateralFilter(img, filter_val,sigma,s).astype(np.uint8)
        elif filter_type == 'gaussian':
            img = cv2.GaussianBlur(img, filter_val, sigma).astype(np.uint8)
    return cv2.Canny(img,threshold1 = th1,threshold2 = th2, L2gradient = True)

#given a binary mask and the images it uses the findContours function and the fitEllipse function
#to draw  an ellipse arround each element of the image.
#Optionally it uses a gaussian blur that permits todrow the ellipses slightly bigger
def draw_ellipses(mask, images, kernel_size = (3,3), sigma=2):
    circled_imgs = []
    for i,img in enumerate(mask):
        if kernel_size is not None:
            img = cv2.GaussianBlur(img, kernel_size, sigma)
        contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if(len(images[i].shape) == 3):
            circled_imgs.append(images[i].copy())
        else: 
            circled_imgs.append(cv2.cvtColor( images[i].copy(),cv2.COLOR_GRAY2RGB))
        for cnt in contours:
            ellipse = cv2.fitEllipse(cnt)
            cv2.ellipse(circled_imgs[i],ellipse, (255,69,0), 2)
    return circled_imgs

#this funcion is used to load the images at the beginnig of each task
def load_images(path,names):
    NIR_imgs = []
    COLOR_imgs = []
    for name in names:
        NIR_imgs.append(cv2.imread('fruit-inspection-images/' + path + '/C0_0000' + name + '.png', cv2.IMREAD_GRAYSCALE))
        COLOR_imgs.append(cv2.cvtColor(cv2.imread('fruit-inspection-images/' + path + '/C1_0000' + name + '.png', cv2.IMREAD_COLOR),cv2.COLOR_BGR2RGB))
    return NIR_imgs, COLOR_imgs


