import json
import os


"""
cd datasets
mkdir -p mix_det/annotations
cp mot/annotations/val_half.json mix_det/annotations/val_half.json
cp mot/annotations/test.json mix_det/annotations/test.json
cd mix_det
ln -s ../mot/train mot_train
ln -s ../crowdhuman/CrowdHuman_train crowdhuman_train
ln -s ../crowdhuman/CrowdHuman_val crowdhuman_val
ln -s ../Cityscapes cp_train
ln -s ../ETHZ ethz_train
cd ..
"""
#MOT17 train, CrowdHuman train and val, Cityperson train, ETHZ train

# 读取mot数据集的标注信息
mot_json = json.load(open('datasets/mot/annotations/train.json','r'))

img_list = list()
for img in mot_json['images']:
    img['file_name'] = 'mot_train/' + img['file_name']
    img_list.append(img)

ann_list = list()
for ann in mot_json['annotations']:
    ann_list.append(ann)

video_list = mot_json['videos']
category_list = mot_json['categories']


print('mot17')

max_img = 10000
max_ann = 2000000
max_video = 10

# 读取Crowdhuman数据集的标注信息
crowdhuman_json = json.load(open('datasets/crowdhuman/annotations/train.json','r'))
# 将crowdhuman数据集的图片和标注信息添加到mix_mot_ch数据集中
img_id_count = 0
for img in crowdhuman_json['images']:
    img_id_count += 1
    img['file_name'] = 'crowdhuman_train/' + img['file_name']
    img['frame_id'] = img_id_count
    # the same image id
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in crowdhuman_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

video_list.append({
    'id': max_video,
    'file_name': 'crowdhuman_train'
})

print('crowdhuman_train')

max_img = 30000
max_ann = 10000000

crowdhuman_val_json = json.load(open('datasets/crowdhuman/annotations/val.json','r'))
img_id_count = 0
for img in crowdhuman_val_json['images']:
    img_id_count += 1
    img['file_name'] = 'crowdhuman_val/' + img['file_name']
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in crowdhuman_val_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

video_list.append({
    'id': max_video,
    'file_name': 'crowdhuman_val'
})

print('crowdhuman_val')

max_img = 40000
max_ann = 20000000

ethz_json = json.load(open('datasets/ETHZ/annotations/train.json','r'))
img_id_count = 0
for img in ethz_json['images']:
    img_id_count += 1
    img['file_name'] = 'ethz_train/' + img['file_name'][5:]
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in ethz_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

print('ETHZ')

video_list.append({
    'id': max_video,
    'file_name': 'ethz'
})

max_img = 50000
max_ann = 25000000

cp_json = json.load(open('datasets/Cityscapes/annotations/train.json','r'))
img_id_count = 0
for img in cp_json['images']:
    img_id_count += 1
    img['file_name'] = 'cp_train/' + img['file_name'][11:]
    img['frame_id'] = img_id_count
    img['prev_image_id'] = img['id'] + max_img
    img['next_image_id'] = img['id'] + max_img
    img['id'] = img['id'] + max_img
    img['video_id'] = max_video
    img_list.append(img)
    
for ann in cp_json['annotations']:
    ann['id'] = ann['id'] + max_ann
    ann['image_id'] = ann['image_id'] + max_img
    ann_list.append(ann)

print('Cityscapes')

video_list.append({
    'id': max_video,
    'file_name': 'cityperson'
})

mix_json = dict()
mix_json['images'] = img_list
mix_json['annotations'] = ann_list
mix_json['videos'] = video_list
mix_json['categories'] = category_list
json.dump(mix_json, open('datasets/mix_det/annotations/train.json','w'))
