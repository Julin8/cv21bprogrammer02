import cv2
import os
import numpy as np
import json
from matplotlib import pyplot as plt

# img_path = "train\\train\\1173.jpg"
# img = cv2.imread("train\\train\\1173.jpg")
# img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# img_size = img.shape[:3]
# img_height, img_width, img_depth = img_size[0], img_size[1], img_size[2]
# dic = {}
# im = img_path.replace("train\\train\\", "")
# dic.setdefault(im, {})["height"] = img_height
# dic.setdefault(im, {})["width"] = img_width
# dic.setdefault(im, {})["deep"] = img_depth
# dic.setdefault(im, {})["objects"] = {}
# dic.setdefault(im, {})["objects"]["1"] = {"name": "karry", "age": 21}
# dic.setdefault(im, {})["objects"]["2"] = {"name": "julin", "age": 21}
# print(dic)

# for root, dirs, files in os.walk('samples'):
#    for file in files:
#        template_path = os.path.join(root, file)
# template = cv2.imread('samples\\bowl\\4307258.jpg')
# template_size = template.shape[:2]
# print(template_size)

result_dic = {}


# tmp = {}


def search_returnPoint(img, template, template_size):
    img_size = img.shape[:2]
    img_height, img_width = img_size[0], img_size[1]
    height, width = template_size[0], template_size[1]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    template_ = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    if height > img_height or width > img_width:
        return None, None, None, None, None
    result = cv2.matchTemplate(img_gray, template_, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    # res大于99%
    loc = np.where(result >= threshold)
    # 使用灰度图像中的坐标对原始RGB图像进行标记
    point = ()
    for pt in zip(*loc[::-1]):
        # cv2.rectangle(img, pt, (pt[0] + template_size[1], pt[1] + + template_size[0]), (7, 249, 151), 3)
        cv2.rectangle(img, pt, (pt[0] + width, pt[1] + height), (0, 0, 255), 3)
        point = pt
    if point == ():
        return None, None, None, None, None
    return img, point[0], point[1], point[0] + width, point[1] + height


# img, xmin, ymin, xmax, ymax = search_returnPoint(img, template, template_size)
# print(xmin, ymin, xmax, ymax)

for test_root, test_dirs, test_files in os.walk('test\\'):
    for test_file in test_files:
        temp = {}
        test_image_path = os.path.join(test_root, test_file)
        test_image = cv2.imread(test_image_path)
        test_image_size = test_image.shape[:3]
        test_image_height = test_image_size[0]
        test_image_width = test_image_size[1]
        test_image_depth = test_image_size[2]
        test_im = test_image_path.replace("test\\", "")
        temp.setdefault(test_im, {})["height"] = test_image_height
        temp.setdefault(test_im, {})["width"] = test_image_width
        temp.setdefault(test_im, {})["deep"] = test_image_depth
        temp.setdefault(test_im, {})["objects"] = {}

        for root, dirs, files in os.walk('samples'):
            for file in files:
                object = {}
                count = 0
                box = []
                template_path = os.path.join(root, file)
                template = cv2.imread(template_path)
                template_size = template.shape[:3]
                category_name = root.replace("samples\\", "")
                img, xmin, ymin, xmax, ymax = search_returnPoint(test_image, template, template_size)
                # if xmin is None:
                #    print(template_path + " None")
                if xmin is not None:
                    # print("karry")
                    # print(category_name)
                    object['catagory'] = category_name
                    box.append(xmin)
                    box.append(ymin)
                    box.append(xmax)
                    box.append(ymax)
                    # print(box)
                    object["bbox"] = box
                    # print(object)
                    temp.setdefault(test_im, {})["objects"][file.replace(".jpg", "")] = object
                    count += 1
                if count == 30:
                    continue
        print(temp)
        result_dic.update(temp)

# print(dic)
with open("test.json", 'w') as tj:
    json.dump(result_dic, tj)

# img, xmin, ymin, xmax, ymax = search_returnPoint(img, template, template_size)
# if img is None:
#    print("None")
# else:
#    print("bbox:"+str(xmin)+" "+str(ymin)+" "+str(xmax)+" "+str(ymax))
#    plt.figure()
#    plt.imshow(img, animated=True)
