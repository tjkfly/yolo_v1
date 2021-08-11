# 将图片的名字写到txt文件中 train 、val 、 test
import random
import os

path = './dataset/voc2007'
# xml 标注文件夹
root_dir = os.path.join(path + '/Annotations')
# print(root_dir)
write_path = open(path + '/ImageSets/Main/pscalvoc.txt', 'w')
for dir_path, dir_name, file_names in os.walk(root_dir):
    # print(len(file_names))   #  9963 个
    for file_name in file_names:
        # os.path.splitext() 将文件名和扩展名分开
        if os.path.splitext(file_name)[1] == '.xml':
            # 把文件名字写入txt
            write_path.write(os.path.splitext(file_name)[0] + '\n', )
write_path.close()

annotation_path = path + u"/ImageSets/Main/pscalvoc.txt"
train_path = path + u"/ImageSets/Main/train.txt"
val_path = path + u"/ImageSets/Main/val.txt"
test_path = path + u"/ImageSets/Main/test.txt"

train_file = open(train_path, "w")
val_file = open(val_path, "w")
test_file = open(test_path, "w")
anno = open(annotation_path, 'r')

result = []
my_dict = {}
cnt = 0
for line in anno:
    my_dict[cnt] = line
    cnt += 1
total_num = cnt


# 7 : 2 : 1
train_num = int(total_num * 0.7)
val_num = int(total_num * 0.2)
test_num = total_num - train_num - val_num

test_set = set()
val_set = set()
train_set = set()

while len(test_set) < test_num:
    x = random.randint(0, total_num)
    if x not in test_set:
        test_set.add(x)

while len(val_set) < val_num:
    x = random.randint(0,total_num)
    if x in test_set:
        continue
    if x not in val_set:
        val_set.add(x)
for x in range(total_num):
    if x in val_set or x in test_set:
        continue
    else:
        train_set.add(x)

index = 0

for i in range(cnt):
    strs = my_dict[i]
    if i in train_set:
        train_file.write(strs)
    elif i in val_set:
        val_file.write(strs)
    else:
        test_file.write(strs)
    index += 1

train_file.close()
val_file.close()
test_file.close()
