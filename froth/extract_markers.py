import cv2
import numpy as np

from markers import get_markers
from preprocessing import preprocess


# frame = cv2.imread("frames/1.png")
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# # gray = cv2.equalizeHist(gray)
# #
# # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 39, 2)
# # cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
# # cnts = cnts[0] if len(cnts) == 2 else cnts[1]
# # frame1 = frame.copy()
#
# # cv2.imwrite("markers/1.png", thresh)

def extract_markers(grayscale_frame,
                    markers_out_path: str = None,
                    highlits_out_path: str = None,
                    preprocessed_out_path: str = None):
    preprocessed_frame = preprocess(grayscale_frame)
    highlights = get_markers("small", preprocessed_frame)
    # highlights = cv2.imread("markers/single.png")[..., 0]   # test
    contours, _ = cv2.findContours(highlights, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    markers = np.zeros(highlights.shape, dtype=np.uint8)
    for contour in contours:
        M = cv2.moments(contour)
        if M['m00'] != 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            cx_sum = 0
            cy_sum = 0
            n_dots = contour.shape[0]
            for i in range(n_dots):
                cx_sum += contour[i, 0, 0]
                cy_sum += contour[i, 0, 1]
            cx_sum /= n_dots
            cy_sum /= n_dots
            cx = np.round(cx_sum).astype(np.uint32)
            cy = np.round(cy_sum).astype(np.uint32)

        markers[cy, cx] = 255

    cv2.imwrite(f"{highlits_out_path}", highlights)    # for tests
    cv2.imwrite(f"{preprocessed_out_path}", preprocessed_frame)
    if markers_out_path is not None:
        cv2.imwrite(f"{markers_out_path}", markers)

    markers_list = np.where(markers == 255)
    markers_list = np.stack([markers_list[0], markers_list[1]], axis=-1)

    return markers_list


# drawContours(frame1, cnts)
# show(frame, frame1)