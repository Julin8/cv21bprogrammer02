import cv2
import json
import os
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt


def createDirectory(name):
    path = 'samples/'
    isExist = os.path.exists(path + name)
    if not isExist:
        os.makedirs(path + name)
    dir_path = path + name
    return dir_path


def cropImage(image_path, box, dir, object_id):
    image = Image.open(image_path)
    img = image.crop(box)
    img.save(dir + "/" + str(object_id) + '.jpg')


# img_rgb = cv2.imread('train/train/1173.jpg')
# img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2GRAY)
# im = Image.open('train/train/1173.jpg')
# draw = ImageDraw.Draw(im)

# with open("./train/train.json", 'r') as load_json:
#    load_dict = json.load(load_json)
#    object_dic = load_dict["1173.jpg"]["objects"]
#    dic = load_dict["1173.jpg"]["objects"]['4234238']['bbox']
#    xmin = dic[0]
#    ymin = dic[1]
#    xmax = dic[2]
#    ymax = dic[3]

# cropImage('train/train/1173.jpg', [xmin, ymin,xmax,ymax],'samples',1)

train_path = 'train/train'
train = json.load(open('train/train.json', 'r'))
category_mapping = []
for image_name in train:
    image_path = os.path.join('train/train/', image_name)
    for object_id in train[image_name]['objects']:
        # print(object_id)
        category = train[image_name]['objects'][object_id]['category']
        box = train[image_name]['objects'][object_id]['bbox']
        category_dir = ''
        if category not in category_mapping:
            category_mapping.append(category)
            category_dir = createDirectory(category)
        else:
            category_dir = 'samples/'+category
        cropImage(image_path, box, category_dir, object_id)
# print(category_mapping)
# for root, dirs, files in os.walk(train_path):
# for file in files:
#   image_path = os.path.join(root, file)
# image_dic = load_dict[file]['objects']
#   image = cv2.imread(image_path)
# print(image.shape)
# print(image_dic)

# print(img_rgb.shape)

# cv2.rectangle(img_rgb, (x-long/2, y+width/2), (x+long/2, y-width/2), (0, 255, 0), 3)
# cv2.imshow("Image", img_rgb)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# draw.line([(xmin, ymin),
#          (xmin, ymax),
#          (xmax, ymax),
#          (xmax, ymin),
#          (xmin, ymin)],
#         width=1, fill='red')
print("OK")
# im.save('test1.jpg')
# bbox = (xmin, ymin, xmax, ymax)
# ng = im.crop(bbox)
# ng.save('1.jpg')
