import cv2
import os
import numpy as np
import json
import imutils


result_dic = {}


# tmp = {}

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


def search_returnPoint(img, template, template_size):
    img_size = img.shape[:2]
    img_height, img_width = img_size[0], img_size[1]
    height, width = template_size[0], template_size[1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_ = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if height > img_height or width > img_width:
        return None, None, None, None, None

    for scale in np.linspace(1, min(img_height, img_width), num=20):
        resized = imutils.resize(template_, width=int(template_.shape[1]*scale))
        height, width = resized.shape[0], resized.shape[1]
        if height > img_height or width > img_width:
            break

        for rotate in np.linspace(0, 360, num=40):
            M = cv2.getRotationMatrix2D((width/2, height/2), rotate, 1)
            rotated = cv2.warpAffine(resized, M, (width, height))
            height, width = rotated.shape[0], rotated.shape[1]

            result = cv2.matchTemplate(img_gray, rotated, cv2.TM_CCOEFF_NORMED)
            threshold = 0.95
            # res大于99%
            loc = np.where(result >= threshold)
            # 使用灰度图像中的坐标对原始RGB图像进行标记
            point = ()
            for pt in zip(*loc[::-1]):
                cv2.rectangle(img, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 3)
                point = pt
            if point == ():
                return None, None, None, None, None
            return img, point[0], point[1], point[0] + width, point[1] + height
    return None, None, None, None, None


def getdic(path):
    temp = {}
    test_image = cv2.imread(path)
    test_image_size = test_image.shape[:3]
    test_image_height = test_image_size[0]
    test_image_width = test_image_size[1]
    test_image_depth = test_image_size[2]
    test_im = path.replace("test\\", "")
    temp.setdefault(test_im, {})["height"] = test_image_height
    temp.setdefault(test_im, {})["width"] = test_image_width
    temp.setdefault(test_im, {})["deep"] = test_image_depth
    temp.setdefault(test_im, {})["objects"] = {}
    for root, dirs, files in os.walk('samples'):
        for file in files:
            object = {}
            box = []
            template_path = os.path.join(root, file)
            template = cv2.imread(template_path)
            template_size = template.shape[:3]
            category_name = root.replace("samples\\", "")
            img, xmin, ymin, xmax, ymax = search_returnPoint(test_image, template, template_size)
            if xmin is not None:
                object['category'] = category_name
                box.append(xmin)
                box.append(ymin)
                box.append(xmax)
                box.append(ymax)
                object["bbox"] = box
                print(object)
                temp.setdefault(test_im, {})["objects"][file.replace(".jpg", "")] = object
    return temp


with open("test.json", 'w') as tj:
    for test_root, test_dirs, test_files in os.walk('test\\'):
        for test_file in test_files:
            test_path = os.path.join(test_root, test_file)
            temp = getdic(test_path)
            print(temp)
            json.dump(temp, tj, cls=NpEncoder)
print("Write done")
            
