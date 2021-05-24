# -*- coding: utf-8 -*-
import os.path

import cv2

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('img')
    args = parser.parse_args()
    img_file = args.img

    out_filename = os.path.splitext(os.path.basename(img_file))[0] + '.canny' + '.jpg'
    out_dir = './output'
    os.makedirs(out_dir, exist_ok=True)
    out_filepath = os.path.join(out_dir, out_filename)
    img = cv2.imread(img_file, 0)
    cv2.imwrite(out_filepath, cv2.Canny(img, 200, 300))
    cv2.imshow("canny", cv2.imread(out_filepath))
    while True:
        if cv2.waitKey(0) & 0xFF == 27:
            cv2.destroyAllWindows()
            break
