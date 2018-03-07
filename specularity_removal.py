import cv2 as cv
import numpy as np
import logging
import os



# FLANN - Fast Library for Approximate Nearest Neighbors
FLANN_INDEX_KDTREE = 0
SCH_PARAM_CHECKS = 50
INDEX_PARAM_TREES = 5

GOOD_MATCH_THRESHOLD = 0.6
MIN_MATCH_COUNT = 20

RESULTS_DIR = 'results/'

DIFF_THRESHOLD = 20

def homography(img1, img2, visualize=False):
    """
    Finds Homography matrix from Image1 to Image2.
        Two images should be a plane and can change in viewpoint

    :param img1: Source image
    :param img2: Target image
    :param visualize: Flag to visualize the matched pixels and Homography warping
    :return: Homography matrix. (or) Homography matrix, Visualization image - if visualize is True
    """
    sift = cv.xfeatures2d.SIFT_create()
    kp1, desc1 = sift.detectAndCompute(img1, None)
    kp2, desc2 = sift.detectAndCompute(img2, None)

    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=INDEX_PARAM_TREES)
    # number of times the trees in the index should be recursively traversed
    # Higher values gives better precision, but also takes more time
    sch_params = dict(checks=SCH_PARAM_CHECKS)
    flann = cv.FlannBasedMatcher(index_params, sch_params)

    matches = flann.knnMatch(desc1, desc2, k=2)
    logging.debug('{} matches found'.format(len(matches)))

    # select good matches
    matches_arr = []
    good_matches = []
    for m, n in matches:
        if m.distance < GOOD_MATCH_THRESHOLD * n.distance:
            good_matches.append(m)
        matches_arr.append(m)

    if len(good_matches) < MIN_MATCH_COUNT:
        raise (Exception('Not enough matches found'))
    else:
        logging.debug('{} of {} are good matches'.format(len(good_matches), len(matches)))

    src_pts = [kp1[m.queryIdx].pt for m in good_matches]
    src_pts = np.array(src_pts, dtype=np.float32).reshape((-1, 1, 2))
    dst_pts = [kp2[m.trainIdx].pt for m in good_matches]
    dst_pts = np.array(dst_pts, dtype=np.float32).reshape((-1, 1, 2))

    homo, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5)

    if visualize:
        res = visualize_homo(img1, img2, kp1, kp2, matches, homo, mask)
        return homo, res

    return homo


def visualize_homo(img1, img2, kp1, kp2, matches, homo, mask):
    h, w, d = img1.shape
    pts = [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
    pts = np.array(pts, dtype=np.float32).reshape((-1, 1, 2))
    dst = cv.perspectiveTransform(pts, homo)

    img2 = cv.polylines(img2, [np.int32(dst)], True, [255, 0, 0], 3, 8)

    matches_mask = mask.ravel().tolist()
    draw_params = dict(matchesMask=matches_mask,
                       singlePointColor=None,
                       matchColor=(0, 255, 0),
                       flags=2)
    res = cv.drawMatches(img1, kp1, img2, kp2, matches, None, **draw_params)
    return res


def remove_specularity(img_files):
    """
    Removes highlights/specularity in Images from multiple view points

    :param img_files: File names of input images in horizontal order (important)
    """
    # read images from file names
    imgs = read_images(img_files)

    # solve each pair of image.
    # assumption: Input images are in order
    for i in range(len(imgs) - 1):
        logging.debug('processing images {} and {}'.format(i+1, i+2))
        imgs[i], imgs[i+1] = _solve(imgs[i], imgs[i + 1])

    for i, path in enumerate(img_files):
        fname = os.path.basename(path) #返回/或\后的文件名
        res_file = os.path.join(RESULTS_DIR, fname)

        logging.info('saving the results in {}'.format(res_file))
        cv.imwrite(res_file, imgs[i])


def read_images(img_files):
    """
    Reads images from the given file locations.
    Also does basic validations like:
        - All images should be of same size

    :param img_files: Image file names
    :return: Read image objects
    """
    imgs = list()
    for fname in img_files:
        img = cv.imread(fname)
        imgs.append(img)

    # validations
    if len(imgs) > 0:
        h, w = imgs[0].shape[:2]
        logging.debug('Images are of resolution: {}x{}'.format(w, h))
        if h > 1000 or w > 1200:
            logging.warning('Image resolution is too high. It might take longer time to process. '
                            'Try reducing the resolution')
        for i in range(1, len(imgs)):
            if imgs[i].shape[0] != h or imgs[i].shape[1] != w:
                logging.error('Images are not of same size')
                raise(Exception('Sorry! This method works for same sized images only'))

    return imgs


def _solve(img1, img2):
    h, w, d = img1.shape

    # step 1: Find homography of 2 images
    homo = homography(img2, img1)

    # step 2: warp image2 to image1 frame
    img2_w = cv.warpPerspective(img2, homo, (w, h))

    # step 3: resolve highlights by picking the best pixels out of two images
    im1 = _resolve_spec(img1, img2_w)

    # step 4: repeat the same process for Image2 using warped Image1
    im_w = cv.warpPerspective(im1, np.linalg.inv(homo), (w, h))
    im2 = _resolve_spec(img2, im_w)

    return im1, im2


def _resolve_spec(im1, im2):
    im = im1.copy()

    img1 = cv.cvtColor(im1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(im2, cv.COLOR_BGR2GRAY)

    # Best pixel selection criteria
    #   1. Pixel difference should be more than 20. (just an experimentally value. Free to change!)
    #   2. Best pixel should have less intensity
    #   3. pixel should not be pure black. (just an additional constraint
    #       to remove black background created by warping)
    mask = np.logical_and((img1 - img2) > DIFF_THRESHOLD, img1 > img2)
    mask = np.logical_and(mask, img2 != 0)

    im[mask] = im2[mask]
    return im
