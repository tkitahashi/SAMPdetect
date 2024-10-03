import cv2, os
import time
import numpy as np


# Average Hash Algorithm
def aHash(img):
    img = cv2.resize(img, (8, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    np_mean = np.mean(gray)  # Calculate the average value of numpy.ndarray
    ahash_01 = (gray > np_mean) + 0  # Greater than the average value = 1, otherwise = 0
    ahash_list = ahash_01.reshape(1, -1)[0].tolist()  # Flatten -> Convert to a list
    ahash_str = "".join([str(x) for x in ahash_list])
    return ahash_str


def pHash(img):
    img = cv2.resize(img, (64, 64))  # Default interpolation=cv2.INTER_CUBIC
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    dct = cv2.dct(np.float32(gray))
    dct_roi = dct[0:8, 0:8]  # Masking operation implemented by OpenCV

    avreage = np.mean(dct_roi)
    phash_01 = (dct_roi > avreage) + 0
    phash_list = phash_01.reshape(1, -1)[0].tolist()
    phash_str = "".join([str(x) for x in phash_list])
    return phash_str


def dHash(img):
    img = cv2.resize(img, (9, 8))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # If the previous pixel in each row is greater than the next pixel, it is 1, otherwise it is 0, generating a hash
    hash_str0 = []
    for i in range(8):
        hash_str0.append(gray[:, i] > gray[:, i + 1])
    hash_str1 = np.array(hash_str0) + 0
    hash_str2 = hash_str1.T
    hash_str3 = hash_str2.reshape(1, -1)[0].tolist()
    dhash_str = "".join([str(x) for x in hash_str3])
    return dhash_str


def hammingDist(s1, s2):
    assert len(s1) == len(s2)
    return sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)])


# Calculate similarity by obtaining histograms of each RGB channel
def classify_hist_with_split(image1, image2, size=(256, 256)):
    # After resizing the image, separate it into three RGB channels, and then calculate the similarity value of each channel
    image1 = cv2.resize(image1, size)
    image2 = cv2.resize(image2, size)
    sub_image1 = cv2.split(image1)
    sub_image2 = cv2.split(image2)
    sub_data = 0
    for im1, im2 in zip(sub_image1, sub_image2):
        sub_data += calculate(im1, im2)
    sub_data = sub_data / 3
    return sub_data


# Calculate the similarity value of the histogram of a single channel
def calculate(image1, image2):
    hist1 = cv2.calcHist([image1], [0], None, [256], [0.0, 255.0])
    hist2 = cv2.calcHist([image2], [0], None, [256], [0.0, 255.0])
    # Calculate the overlap degree of histograms
    degree = 0
    for i in range(len(hist1)):
        if hist1[i] != hist2[i]:
            degree = degree + (1 - abs(hist1[i] - hist2[i]) / max(hist1[i], hist2[i]))
        else:
            degree = degree + 1
    degree = degree / len(hist1)
    return degree


def is_same_frame_by_diff(last_frame, frame):
    # Calculate the difference between the current frame and the previous frame
    frame_delta = cv2.absdiff(last_frame, frame)

    # Convert the result to a grayscale image
    thresh = cv2.cvtColor(frame_delta, cv2.COLOR_BGR2GRAY)

    # Binarize the image
    thresh = cv2.threshold(thresh, 25, 255, cv2.THRESH_BINARY)[1]

    """ 
    #Remove image noise, first erode and then dilate (morphological opening operation)
    thresh=cv2.erode(thresh,None,iterations=1) 
    thresh = cv2.dilate(thresh, None, iterations=2) 
    """
    # Contour positions on the threshold image
    contours = cv2.findContours(
        thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )[-2]

    # Traverse the contours
    for contour in contours:
        # Ignore small contours to eliminate errors
        if cv2.contourArea(contour) < 5:
            continue
        else:
            return False

    return True


if __name__ == "__main__":

    start = time.time()

    raw_img1 = "2019Sep11_pic\\pic00000001.jpg"

    for root, dirs, files in os.walk("2019Sep11_pic", topdown=False):
        pass
    for file in files:
        raw_img2 = os.path.join(root, file)
        img1 = cv2.imread(raw_img1)
        img2 = cv2.imread(raw_img2)

        ahash_str1 = aHash(img1)
        ahash_str2 = aHash(img2)

        phash_str1 = pHash(img1)
        phash_str2 = pHash(img2)

        dhash_str1 = dHash(img1)
        dhash_str2 = dHash(img2)
        a_score = 1 - hammingDist(ahash_str1, ahash_str2) * 1.0 / (32 * 32 / 4)
        p_score = 1 - hammingDist(phash_str1, phash_str2) * 1.0 / (32 * 32 / 4)
        d_score = 1 - hammingDist(dhash_str1, dhash_str2) * 1.0 / (32 * 32 / 4)

        n = classify_hist_with_split(img1, img2)
        # print('Similarity of the three-histogram algorithm：', n)
        # print('a_score:{},p_score:{},d_score{}'.format(a_score,p_score,d_score))
        print(
            "%s,    %.8f,    %.8f,    %.8f,    %.8f"
            % (file, n, a_score, p_score, d_score)
        )
        raw_img1 = raw_img2

    end = time.time()
    print("Total Spend time：", str((end - start) / 60)[0:6] + "minutes")
