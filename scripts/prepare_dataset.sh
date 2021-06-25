set -e

PROJ_DIR==$HOME/RefineMask
cd $PROJ_DIR

COCO_ROOT=...
LVIS_ROOT=...

rm -rf data

mkdir data && cd data
mkdir coco && cd coco

ln -s $COCO_ROOT/train2017 ./
ln -s $COCO_ROOT/val2017 ./
ln -s $COCO_ROOT/test2017 ./

ln -s $COCO_ROOT/annotations ./

cd ..
mkdir lvis && cd lvis
mkdir annotations

ln -s $LVIS_ROOT/lvis_v0.5_train.json annotations/

# fix lvis name prefix
python $PROJ_DIR/tools/lvis_filename_to2017.py $LVIS_ROOT/lvis_v0.5_val.json
mv lvis_v0.5_val.json.2017 annotations/lvis_v0.5_val.json

# add lvis v1.0 anno
ln -s $LVIS_ROOT/lvis_v1.0/lvis_v1_train.json annotations/
ln -s $LVIS_ROOT/lvis_v1.0/lvis_v1_val.json annotations/

ln -s $COCO_ROOT/train2017 ./
ln -s $COCO_ROOT/val2017 ./
ln -s $COCO_ROOT/test2017 ./

cd ../../

set +e
