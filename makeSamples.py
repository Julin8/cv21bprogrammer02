import json
import os
from PIL import Image


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


train_path = 'train/train'
train = json.load(open('train/train.json', 'r'))
category_mapping = []
for image_name in train:
    image_path = os.path.join('train/train/', image_name)
    for object_id in train[image_name]['objects']:
        category = train[image_name]['objects'][object_id]['category']
        box = train[image_name]['objects'][object_id]['bbox']
        category_dir = ''
        if category not in category_mapping:
            category_mapping.append(category)
            category_dir = createDirectory(category)
        else:
            category_dir = 'samples/'+category
        cropImage(image_path, box, category_dir, object_id)


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
