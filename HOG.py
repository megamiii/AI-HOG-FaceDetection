import cv2
import numpy as np
import matplotlib.pyplot as plt


def get_differential_filter():
    # TODO: implement this function
    # sobel filter: derivative of gaussian filter
    # sobel filter for x direction
    filter_x = np.array([[-1, 0, 1],
                         [-2, 0, 2],
                         [-1, 0, 1]])
    
    # sobel filter for y direction
    filter_y = np.array([[-1, -2, -1],
                         [ 0,  0,  0],
                         [ 1,  2,  1]])
    return filter_x, filter_y


def filter_image(im, filter):
    # TODO: implement this function
    # get image dimensions
    h, w = im.shape
    
    # get filter dimensions
    fh, fw = filter.shape
    
    #  pad zeros on the boundary on the input image
    padded_im = np.pad(im, ((1, 1), (1, 1)), mode='constant')
    
    # initialize the filtered image
    im_filtered = np.zeros_like(im)
    
    # convolution
    for i in range(h):
        for j in range(w):
            im_filtered[i, j] = np.sum(padded_im[i:i+fh, j:j+fw] * filter)
            
    return im_filtered


def get_gradient(im_dx, im_dy):
    # TODO: implement this function
    # gradient magnitude
    grad_mag = np.sqrt(im_dx ** 2 + im_dy ** 2)
    
    # gradient angle
    grad_angle = np.arctan2(im_dy, im_dx)
    
    # the range of angle must be [0, pi)
    grad_angle[grad_angle < 0] += np.pi
    return grad_mag, grad_angle


def build_histogram(grad_mag, grad_angle, cell_size):
    # TODO: implement this function
    # number of cells along y and x axes
    m, n = grad_mag.shape
    M = m // cell_size
    N = n // cell_size
    
    # convert radians to degrees ([0, 180])
    deg_angle = np.degrees(grad_angle) % 180
    
    # bin edges for histogram computation based on the provided theta ranges
    bins = np.array([15, 45, 75, 105, 135, 165])
    
    # bin index for each gradient angle in the image
    bins_index = np.digitize(deg_angle, bins)
    
    # initialize histogram (a 3D tensor with size M x N x 6)
    ori_histo = np.zeros((M, N, 6))
    
    # compute the histogram
    for i in range(M):
        for j in range (N):
            # orientations and magnitudes for gradients in the current cell
            cell_orientations = bins_index[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_magnitudes = grad_mag[i*cell_size:(i+1)*cell_size, j*cell_size:(j+1)*cell_size]
            cell_orientations[cell_orientations == 6] = 0
            # put magnitudes into the corresponding histogram bins
            for k in range(6):
                ori_histo[i, j, k] = np.sum(cell_magnitudes[cell_orientations  == k])
                
    return ori_histo


def get_block_descriptor(ori_histo, block_size):
    # TODO: implement this function
    # normalization constant
    e = 0.001
    
    # calculate the size of the normalized histogram
    M, N, _ = ori_histo.shape
    len_x = M - (block_size - 1)
    len_y = N - (block_size - 1)
    
    # initialize the normalized histogram
    ori_histo_normalized = np.zeros((len_x, len_y, 6 * block_size**2))
    
    # iterate through each block in the input histogram
    for i in range(len_x):
        for j in range(len_y):
            # extract and flatten the block
            block = ori_histo[i:i+block_size, j:j+block_size, :].flatten()
            # calculate the L2 norm
            norm = (np.sqrt(np.sum(block**2) + e**2))
            # assign the normalized histogram to ori_histo_normalized
            ori_histo_normalized[i, j, :] = block / norm
            
    return ori_histo_normalized


# visualize histogram of each block
def visualize_hog(im, hog, cell_size, block_size):
    num_bins = 6
    max_len = 7  # control sum of segment lengths for visualized histogram bin of each block
    im_h, im_w = im.shape
    num_cell_h, num_cell_w = int(im_h / cell_size), int(im_w / cell_size)
    num_blocks_h, num_blocks_w = num_cell_h - block_size + 1, num_cell_w - block_size + 1
    histo_normalized = hog.reshape((num_blocks_h, num_blocks_w, block_size**2, num_bins))
    histo_normalized_vis = np.sum(histo_normalized**2, axis=2) * max_len  # num_blocks_h x num_blocks_w x num_bins
    angles = np.arange(0, np.pi, np.pi/num_bins)
    mesh_x, mesh_y = np.meshgrid(np.r_[cell_size: cell_size*num_cell_w: cell_size], np.r_[cell_size: cell_size*num_cell_h: cell_size])
    mesh_u = histo_normalized_vis * np.sin(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    mesh_v = histo_normalized_vis * -np.cos(angles).reshape((1, 1, num_bins))  # expand to same dims as histo_normalized
    plt.imshow(im, cmap='gray', vmin=0, vmax=1)
    for i in range(num_bins):
        plt.quiver(mesh_x - 0.5 * mesh_u[:, :, i], mesh_y - 0.5 * mesh_v[:, :, i], mesh_u[:, :, i], mesh_v[:, :, i],
                   color='red', headaxislength=0, headlength=0, scale_units='xy', scale=1, width=0.002, angles='xy')
    # plt.show()
    plt.savefig('hog.png')


def extract_hog(im, visualize=False, cell_size=8, block_size=2):
    # TODO: implement this function
    # convert the grey-scale image to float format and normalize to range [0, 1]
    im_float = im.astype('float') / 255.0
    
    # get differential images using get_differential_filter and filter_image
    dx_filter, dy_filter = get_differential_filter()
    im_dx = filter_image(im_float, dx_filter)
    im_dy = filter_image(im_float, dy_filter)
    
    # compute the gradients using get_gradient
    grad_mag, grad_angle = get_gradient(im_dx, im_dy)
    
    # build the histogram of oriented gradients for all cells using build_histogram
    ori_histo = build_histogram(grad_mag, grad_angle, cell_size)
    
    # build the descriptor of all blocks with normalization using get_block_descriptor
    ori_histo_normalized = get_block_descriptor(ori_histo, block_size)
    
    # return a long vector (hog) by concatenating all block descriptors
    hog = ori_histo_normalized.flatten()
    
    return hog


def face_recognition(I_target, I_template):
    # TODO: implement this function
    # extract HOG from the template
    template_HOG = extract_hog(I_template).flatten()
    
    # normalize the template's HOG descriptor
    template_HOG -= np.mean(template_HOG)
    template_norm = np.linalg.norm(template_HOG)

    # target image and template dimensions
    img_h, img_w = I_target.shape
    box_h, box_w = I_template.shape

    # thresholding on the NCC score will produce many overlapping bounding boxes
    NCC_bounding_boxes = []
    
    # thresholds
    NCC_threshold = 0.49
    NMS_threshold = 0.5
    
    # stride for the sliding window (bounding box patch)
    stride = 3

    # sliding window/patch accross the target image
    for y in range(0, img_h - box_h + 1, stride):
        for x in range(0, img_w - box_w + 1, stride):
            # extract window from the target image
            patch = I_target[y:y+box_h, x:x+box_w]
            
            # extraxt its HOG
            patch_HOG = extract_hog(patch).flatten()
            
            # normalize its HOG
            patch_HOG -= np.mean(patch_HOG)
            patch_norm = np.linalg.norm(patch_HOG)

            # calculate the NCC score
            score = np.dot(template_HOG, patch_HOG) / (template_norm * patch_norm)
            if score > NCC_threshold:
                NCC_bounding_boxes.append((x, y, score))
                
    all_scores = [s for (_, _, s) in NCC_bounding_boxes]
    
    # non-maximum suppression (NMS)
    NMS_bounding_boxes = []
    
    for i, box_i in enumerate(NCC_bounding_boxes):
        discard = False
        x1, y1, _ = box_i

        for j, box_j in enumerate(NCC_bounding_boxes):
            if i == j:
                continue

            x2, y2, _ = box_j
            
            # intersection coordinates
            x_inter1 = max(x1, x2)
            y_inter1 = max(y1, y2)
            x_inter2 = min(x1+box_w, x2+box_w)
            y_inter2 = min(y1+box_h, y2+box_h)
            
            # area of intersection and union
            intersection = max(0, x_inter2 - x_inter1) * max(0, y_inter2 - y_inter1)
            union = box_w * box_h + box_w * box_h - intersection
            
            # intersection over union (IOU)
            iou = intersection / union
            
            if iou > NMS_threshold and all_scores[i] < all_scores[j]:
                discard = True
                break

        if not discard:
            NMS_bounding_boxes.append(box_i)

    # convert NMS_bounding_boxes to a numpy array
    bounding_boxes = np.array(NMS_bounding_boxes)
    
    return bounding_boxes


def visualize_face_detection(I_target, bounding_boxes, box_size):

    hh,ww,cc=I_target.shape

    fimg=I_target.copy()
    for ii in range(bounding_boxes.shape[0]):

        x1 = bounding_boxes[ii,0]
        x2 = bounding_boxes[ii, 0] + box_size 
        y1 = bounding_boxes[ii, 1]
        y2 = bounding_boxes[ii, 1] + box_size

        if x1<0:
            x1=0
        if x1>ww-1:
            x1=ww-1
        if x2<0:
            x2=0
        if x2>ww-1:
            x2=ww-1
        if y1<0:
            y1=0
        if y1>hh-1:
            y1=hh-1
        if y2<0:
            y2=0
        if y2>hh-1:
            y2=hh-1
        fimg = cv2.rectangle(fimg, (int(x1),int(y1)), (int(x2),int(y2)), (255, 0, 0), 1)
        cv2.putText(fimg, "%.2f"%bounding_boxes[ii,2], (int(x1)+1, int(y1)+2), cv2.FONT_HERSHEY_SIMPLEX , 0.5, (0, 255, 0), 2, cv2.LINE_AA)

    plt.figure(3)
    plt.imshow(fimg, vmin=0, vmax=1)
    plt.imsave('result_face_detection.png', fimg, vmin=0, vmax=1)
    plt.show()


if __name__=='__main__':

    im = cv2.imread('cameraman.tif', 0)
    hog = extract_hog(im, visualize=False)

    I_target= cv2.imread('target.png', 0) # MxN image

    I_template = cv2.imread('template.png', 0) # mxn  face template

    bounding_boxes = face_recognition(I_target, I_template)

    I_target_c= cv2.imread('target.png') # MxN image (just for visualization)
    
    visualize_face_detection(I_target_c, bounding_boxes, I_template.shape[0]) # visualization code