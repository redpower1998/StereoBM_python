import time
import numpy
import math
import os.path
import cv2

# sobel filtering for preprocessing


def sobel_filter(image):
    height, width = image.shape
    out_image = numpy.zeros((height, width))

    table_x = numpy.array(([-1, -2, -1], [0, 0, 0], [1, 2, 1]))
    table_y = numpy.array(([-1, 0, 1],  [-2, 0, 2], [-1, 0, 1]))

    for y in range(2, width-2):
        for x in range(2, height-2):
            cx, cy = 0, 0
            for offset_y in range(0, 3):
                for offset_x in range(0, 3):
                    pix = image[x + offset_x -
                                1, y + offset_y - 1]
                    if offset_x != 1:
                        cx += pix * table_x[offset_x, offset_y]
                    if offset_y != 1:
                        cy += pix * table_y[offset_x, offset_y]
            out_pix = math.sqrt(cx ** 2 + cy ** 2)
            out_image[x, y] = out_pix if out_pix > 0 else 0
    numpy.putmask(out_image, out_image > 255, 255)
    return out_image

# Calculate left disparity


def calc_left_disparity(gray_left, gray_right, num_disparity=128, block_size=11):

    height, width = gray_right.shape
    disparity_matrix = numpy.zeros((height, width), dtype=numpy.float32)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):
        print("%d%% " % (i * 100 // height), end=' ', flush=True)

        for j in range(half_block, width - half_block):
            left_block = gray_left[i - half_block:i +
                             half_block, j - half_block:j + half_block]
            diff_sum = 32767
            disp = 0

            for d in range(0, min(j - half_block - 1, num_disparity)):
                right_block = gray_right[i - half_block:i +
                                  half_block, j - half_block - d:j + half_block - d]
                sad_val = sum(sum(abs(right_block - left_block)))

                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d

            disparity_matrix[i - half_block, j - half_block] = disp
    print('100%')
    return disparity_matrix


# Calculate right disparity


def calc_right_disparity(gray_left, gray_right, num_disparity=128, block_size=11):

    height, width = gray_right.shape
    disparity_matrix = numpy.zeros((height, width), dtype=numpy.float32)
    half_block = block_size // 2

    for i in range(half_block, height - half_block):
        print("%d%% " % (i * 100 // height), end=' ', flush=True)

        for j in range(half_block, width - half_block):
            right_block = gray_right[i - half_block:i +
                              half_block, j - half_block:j + half_block]
            diff_sum = 32767
            disp = 0

            for d in range(0, min(width - j - half_block, num_disparity)):

                left_block = gray_left[i - half_block:i +
                                 half_block, j - half_block + d:j + half_block + d]
                sad_val = sum(sum(abs(right_block - left_block)))

                if sad_val < diff_sum:
                    diff_sum = sad_val
                    disp = d

            disparity_matrix[i - half_block, j - half_block] = disp
    print('100%')
    return disparity_matrix


# Left-right verification


def left_right_check(disp_left, disp_right):
    height, width = disp_left.shape
    out_image = disp_left

    for h in range(1, height-1):
        for w in range(1, width-1):
            left = int(disp_left[h, w])
            if w - left > 0:
                right = int(disp_right[h, w - left])
                dispDiff = left - right
                if dispDiff < 0:
                    dispDiff = -dispDiff
                elif dispDiff > 1:
                    out_image[h, w] = 0
    return out_image


print('Starting StereoBM')
num_disparity = 32  # 视差范围
left_image_path = 'left.jpg'
right_image_path = 'right.jpg'

print('Read images')
start_time = time.time()

# Read left and right images
image_left = cv2.imread(os.path.join(left_image_path))
image_right = cv2.imread(os.path.join(right_image_path))

# Convert to grayscale
gray_left = numpy.mean(image_left, 2)
gray_right = numpy.mean(image_right, 2)

# Preprocessing
sobel_left = sobel_filter(gray_left)
sobel_right = sobel_filter(gray_right)

cv2.imwrite('sobel_left.bmp', sobel_left)

print('Start LeftBM')
# Calculate left disparity
disparity_left = calc_left_disparity(
    sobel_left, sobel_right, num_disparity, 21)
cv2.imwrite('disparity_left.bmp', disparity_left)
disparity_left_color = cv2.applyColorMap(cv2.convertScaleAbs(
    disparity_left, alpha=256/num_disparity), cv2.COLORMAP_JET)
cv2.imwrite('disparity_leftRGB.bmp', disparity_left_color)

print('Start RightBM')
# Calculate right disparity
disparity_right = calc_right_disparity(
    sobel_left, sobel_right, num_disparity, 21)
disparity_right_color = cv2.applyColorMap(cv2.convertScaleAbs(
    disparity_right, alpha=256/num_disparity), cv2.COLORMAP_JET)
cv2.imwrite('disparity_rightRGB.bmp', disparity_right_color)


print('Start LRCheck')
# Post-processing
disparity = left_right_check(disparity_left, disparity_right)

print('Duration: %s seconds\n' % (time.time() - start_time))

# Save disparity map to file.
cv2.imwrite('disparity.bmp', disparity_right)


# Generate color image and save to file
disparity_color = cv2.applyColorMap(cv2.convertScaleAbs(
    disparity, alpha=256/num_disparity), cv2.COLORMAP_JET)
cv2.imwrite('disparityRGB.bmp', disparity_color)

# Display result
cv2.imshow('Left', image_left)
cv2.imshow('Disparity RGB', disparity_color)
cv2.waitKey(60000)
