import json
import sys
import os
with open(sys.argv[1], 'r') as fd:
    coco = json.load(fd)
images = coco['images']
for idx, img in enumerate(images):
    prefix, img['file_name'] = img['file_name'].rsplit('_', 1)
    assert prefix == 'COCO_val2014', prefix

save_name = os.path.basename(sys.argv[1]) + '.2017'
with open(save_name, 'w') as fd:
    json.dump(coco, fd)
print(f'processed {sys.argv[1]}, saved to {save_name}')
