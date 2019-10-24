import os
import settings

def read_file(file_path):
    with open(file_path,'r',encoding='utf-8') as f:
        contents = f.readlines()
        i = 1
        for content in contents:
            print("day la content thu {}".format(i))
            i += 1
    return contents

contents = read_file(os.path.join(settings.DATA_TRAIN_PATH,'1.txt'))
print(len(contents))
