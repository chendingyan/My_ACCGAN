import i2v
import os
from PIL import Image

tag = ['blonde hair', 'brown hair', 'black hair', 'blue hair', 'pink hair',
       'purple hair', 'green hair', 'red hair', 'silver hair', 'white hair', 'orange hair',
       'aqua hair', 'grey hair', 'long hair', 'short hair', 'twintails', 'drill hair', 'ponytail', 'blush',
       'smile', 'open mouth', 'hat', 'ribbon', 'glasses', 'blue eyes', 'red eyes', 'brown eyes',
       'green eyes', 'purple eyes', 'yellow eyes', 'pink eyes', 'aqua eyes', 'black eyes', 'orange eyes', ]
anime_path = '../face_detector/getchu_detector'
year_path = '../web_crawler/resource/getchu_datas.txt'


def get_tag_map():
    tag_map = dict()
    for i, j in enumerate(tag):
      tag_map[j] = i
    return tag_map




def get_year_dict():
    info_list = []
    with open(year_path, 'r') as fin:
        info_list = fin.readlines()
    info_list = list(map(lambda each: each.strip('\n'), info_list))
    info_list = list(map(lambda each: each.split(' '), info_list))
    year_dict = dict()
    for i, d in enumerate(info_list):
        year_dict[d[0]] = d[1][0:4]

    return year_dict


if __name__ == '__main__':
    illust2vec = i2v.make_i2v_with_chainer(
        "illust2vec_tag_ver200.caffemodel", "tag_list.json")
    tag_map = get_tag_map()
    year_dict = get_year_dict()
    files = os.listdir(anime_path)  # 得到文件夹下的所有文件名称
    for f in files:
        print(f)
        f1 = f.split('_')

        year = year_dict[f1[0]]
        path = anime_path+'/'+f
        print(path)
        img = Image.open(path)



        tags = illust2vec.estimate_specific_tags([img], tag_map)
        sort = sorted(tags[0].items(), key=lambda d: d[1], reverse=True)
        print(sort[0][1])
        if sort[0][1] > 0.25:
            tag_name = sort[0][0]
            ff = open("tags_new.txt", "a+")  # 以追加的方式
            ff.write(f[:-4] + '\t' + year + '\t' + tag_name + '\t' + path+ '\n')
        else:
            print('No tag detect!!!!')
    ff.close()