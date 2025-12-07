""" GN-GCN: Grid neighborhood-based graph convolutional network for spatio-temporal knowledge graph reasoning
Bing Han, Tengteng Qu, Jie Jiang
https://doi.org/10.18170/DVN/UIS4VC
"""


### Train models

0. Make dictionary to save models
```
mkdir models
```



1. Train models
```
cd src

attention:According to the content of the dataset (ICEWS14s、ICEWS18、ICEWS05-15), replace the locgrid_one2one_level7-14s, locgrid_one2one_level7-18 or locgrid_one2one_level7-15 to main.py in src.


The training, testing, and validation sets of the dataset are train.csv, test.csv, valid.cs. Meanwhile, they respectively includes 80%, 10%, and 10% of gridneighbor. csv and entity loc_neighbor. csv used for spatial neighborhood learning.



python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0

python main.py -d ICEWS18 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0


python main.py -d ICEWS05-15 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0

```








### Evaluate models


```
python main.py -d ICEWS14s --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test
```

python main.py -d ICEWS18 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test

python main.py -d ICEWS05-15 --train-history-len 3 --test-history-len 3 --dilate-len 1 --lr 0.001 --n-layers 2 --evaluate-every 1 --gpu=0 --n-hidden 200 --self-loop --decoder convtranse --encoder uvrgcn --layer-norm --weight 0.5  --entity-prediction --relation-prediction --add-static-graph --angle 10 --discount 1 --task-weight 0.7 --gpu 0 --test