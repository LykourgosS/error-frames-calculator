import cv2
import numpy as np


# returns a list of video frames (in RGB format)
def get_colored_frames_from(video_path):
    cap = cv2.VideoCapture(video_path)
    print("\nSplitting video to frames...")
    list_of_frames = []
    frame_no = 0
    while cap.isOpened():
        retVal, frame = cap.read()
        if retVal:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            list_of_frames.append(frame)
            frame_no += 1
        else:
            break
    cap.release()
    print("Finished! (selected video has %d frames)" % len(list_of_frames))
    return np.array(list_of_frames)


# returns the closest bigger multiple of size to i
def rnd_to_nxt_mul(size, i):
    return ((i - 1) // size + 1) * size


# returns a black pixel
def black_pixel():
    # colored (RGB)
    return [[0, 0, 0]]


# returns an image having width multiple of n, by adding black pixels at the end of each row
def add_black_columns_to(image):
    tmp_image = []
    width = image.shape[1]
    # adding black pixels for every row of image
    black_pixels = np.array(black_pixel() * (rnd_to_nxt_mul(16, width) - width))
    for row in image:
        tmp_row = np.append(row, black_pixels, axis = 0)
        tmp_image.append(tmp_row)
    return np.array(tmp_image)


# returns an image having height multiple of n, by adding lines of black pixels at the end of the image
def add_black_rows_to(image):
    tmp_image = list(image)
    height = image.shape[0]
    width = image.shape[1]
    # adding rows of black pixels at the end of the image
    black_line = np.array(black_pixel() * rnd_to_nxt_mul(16, width))
    for j in range(rnd_to_nxt_mul(16, height) - height):
        tmp_image.append(black_line)
    return np.array(tmp_image)


# if needed returns a numpy array of extended video frames, so their dimensions will be multiple of 16 (macroblock size)
def extend_frames(images):
    ext_images = images
    height = images.shape[1]
    width = images.shape[2]
    print("\nThe dimensions of selected video is %d x %d" % (width, height))
    if width % 16 != 0:
        ext_images = []
        for img in images:
            img = add_black_columns_to(img)
            ext_images.append(img)
        images = ext_images
        width = images.shape[2]
    if height % 16 != 0:
        ext_images = []
        for img in images:
            img = add_black_rows_to(img)
            ext_images.append(img)
        height = np.array(ext_images).shape[1]
    # b_per_f = width * height // 16 // 16
    print("The dimensions of the extended frames is %d x %d." % (width, height))
    return np.array(ext_images)


# returns an array of the image divided to macroblocks
def frame2macroblocks(n, frame):
    blocks = []
    coordinates = []
    height = frame.shape[0]
    width = frame.shape[1]
    for row in range(0, height - 1, n):
        for pixel in range(0, width - 1, n):
            macroblock = frame[row: row + n, pixel: pixel + n]
            blocks.append(macroblock)
            coordinates.append((row, pixel))
    return np.array(blocks, dtype = 'int32'), coordinates


# split all video frames to divided frames of macroblocks (16 x 16)
def divide_to_macroblocks(frames):
    div_frames = []
    div_frames_coordinates = []
    print("\nDividing all video frames to macroblocks (16 x 16)...")
    for i in range(len(frames)):
        macroblock, coord = frame2macroblocks(16, frames[i])
        div_frames.append(macroblock)
        div_frames_coordinates.append(coord)
    print("Finished! (%d macroblocks per frame)" % len(macroblock))
    return np.array(div_frames), np.array(div_frames_coordinates)


# given an array of macroblocks returns an array of the image (opposite of frame2macroblocks(...))
def macroblocks2frame(n, shape, div_frame):
    constucted_frame = np.empty(shape = shape, dtype = 'int32')
    height = shape[0]
    width = shape[1]
    for row in range(0, height - 1, n):
        for pixel in range(0, width - 1, n):
            constucted_frame[row: row + n, pixel: pixel + n] = div_frame[(width // n) * (row // n) + (pixel // n)]
    return constucted_frame


# calculating SAD for a given position (pos), using calibrated formula for contemporary CRT phosphors:
# Y = 0.2125 R + 0.7154 G + 0.0721 B (RGB to gray-scale)
def sad(n, target_block, ref_block):
    sum = 0
    for p in range(n):
        for q in range(n):
            # sum += abs(
            #     0.2125 * ref_block[p][q][0] + 0.7154 * ref_block[p][q][1] + 0.0721 * ref_block[p][q][2] - 0.2125 *
            #     target_block[p][q][0] - 0.7154 * target_block[p][q][1] - 0.0721 * target_block[p][q][2])
            sum += abs(ref_block[p][q][0] - target_block[p][q][0])
    return sum


# returns a block from a frame with the coordinates of its upper left pixel equal to coords
def get_block(n, coords, frame):
    return np.array(frame[coords[0]: coords[0] + n, coords[1]: coords[1] + n])


# finding the best fitted block using logarithmic search (SAD metric is being used)
def best_fitted_block(ref_frame, block, coords):
    k = 16
    satisfied_sad = False
    while k > 1:
        best_sad = 16 * 16 * 255  # max SAD for a macroblock of size 16 x 16
        threshold = best_sad * threshold_coefficient
        best_block = None
        # finding the coordinates of the 8 + 1 positions for current k
        positions = [(i + coords[0], j + coords[1]) for i in range(-k // 2, k // 2 + 1, k // 2) for j in
                     range(-k // 2, k // 2 + 1, k // 2)]
        # finding the block with the best SAD for current k
        for pos in positions:
            if pos[0] >= 0 and pos[1] >= 0 and pos[0] <= (ref_frame.shape[0] - 16) and pos[1] <= ref_frame.shape[
                1] - 16:
                tmp_block = get_block(16, pos, ref_frame)
                tmp_sad = sad(16, block, tmp_block)
                if tmp_sad <= best_sad:
                    best_sad = tmp_sad
                    best_block = tmp_block
                    if best_sad <= threshold:
                        satisfied_sad = True
                        break
        if satisfied_sad:
            break
        k //= 2
    return best_block


# returns the block based difference between the given reference and target frame
def single_calc_block_based_frame_diff(ref_frame, div_target_frame, div_target_frame_coords):
    block_based_diff = []
    for i in range(len(div_target_frame)):
        block_based_diff.append(
            abs(div_target_frame[i] - best_fitted_block(ref_frame, div_target_frame[i], div_target_frame_coords[i])))
    block_based_diff = macroblocks2frame(16, ref_frame.shape, np.array(block_based_diff))
    return block_based_diff


# calculating for all video frames, the block-based difference
def calc_block_based_frame_diff_for(images, div_images, div_images_coords):
    print("\nCalculating block based error frames (%d in total)..." % (len(images) - 1))
    block_based_diff = []
    for n in range(0, len(images) - 1):
        print("Complete frame difference #%d" % (n + 1))
        block_based_diff.append(
            single_calc_block_based_frame_diff(images[n], div_images[n + 1], div_images_coords[n + 1]))
    print("Finished!")
    return np.array(block_based_diff), "b_b_err_(%s)_" % str(threshold_coefficient)


# calculating for all video frames, the difference between the n and n+1 frame as whole frames (not block-based)
def calc_whole_frame_diff_for(images):
    print("\nCalculating whole error frames...")
    whole_diff = []
    for n in range(0, len(images) - 1):
        whole_diff.append(abs(images[n + 1] - images[n]))
    print("Finished!")
    return np.array(whole_diff), "wh_err_"


# export difference frames to video
def to_video(images_diff, new_fn):
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    new_filename = new_fn + filename
    out = cv2.VideoWriter(new_filename, fourcc, 25.0, (images_diff.shape[2], images_diff.shape[1]))
    print("\nCreating video file with name \"%s\"..." % new_filename)
    for img_d in images_diff:
        out.write(img_d.astype('uint8'))
    print("The video has been added in your run folder.")
    out.release()


#  start of program

# get filename from the user
filename = input("\nGive the file name of your video... (it should be in the same folder with this .py file)\n")

# splitting given video to frames
frames = get_colored_frames_from(filename)

# select type of difference
while True:
    print("\nSelect what kind of difference you want to plot:")
    selection = input(" a. As whole frames.\n b. Block-based.\nSelect an option from above, typing a, b: ").lower()
    if selection in ["a", "b"]:
        if selection == "a":
            # calculate error frames: difference as whole frames
            diff_to_video, new_fn = calc_whole_frame_diff_for(frames)
        else:
            # extend frames, by adding black pixels when needed
            ext_frames = extend_frames(frames)
            # divide all video frames to macroblocks (n x n)
            div_frames, div_frames_coords = divide_to_macroblocks(ext_frames)
            # selecting the algorithms's indulgence
            coef_list = [0, 0.0001, 0.001, 0.01, 0.1]
            threshold_coefficient = coef_list[
                int(input(
                    "\nGive algorithm indulgence [0(strict) - 4(lenient)] (big indulgence for faster execution): "))]
            # calculate error frames: block-based difference
            diff_to_video, new_fn = calc_block_based_frame_diff_for(ext_frames, div_frames, div_frames_coords)
        to_video(diff_to_video, new_fn)
        break
    else:
        print("Invalid input...")
