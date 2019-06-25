| dataset | model name | Overall Acc |
| :----: | :----: | :----: |
| duke | resnet50/net_last | 0.8777 |
| duke | resnet50/net_49 | 0.8805 | 
| duke | resnet50/net_39 | 0.8796 | 
| duke | resnet50/net_29 | 0.8796 | 
| duke | resnet50/net_19 | 0.8764 | 
| duke | resnet50/net_9 | 0.8806 | 



| dataset | model name | Overall Acc |
| :----: | :----: | :----: |
| duke | densenet/net_last | 0.8583 |
| duke | densenet/net_59 | 0.8583 | 
| duke | densenet/net_49 | 0.8589 | 
| duke | densenet/net_39 | 0.8585 | 
| duke | densenet/net_29 | 0.8550| 
| duke | densenet/net_19 | 0.8571 | 
| duke | densenet/net_9 | 0.8383 | 

### duke的label顺序
'backpack'
'bag'
'handbag'
'boots'
'gender'
'hat'
'shoes'
'top'
'upblack'
'upwhite'
'upred'
'uppurple'
'upgray'
'upblue'
'upgreen'
'upbrown'
'downblack'
'downwhite'
'downred'
'downgray'
'downblue'
'downgreen'
'downbrown'







下衣颜色 上衣颜色 衣袖长短  下衣类型长短 背包  拎东西 帽子 性别
backpack:0
bag:0
handbag:0
down:1
up:1
hair:1
hat:0
gender:1
upblack:0
upwhite:1
upred:0
uppurple:0
upyellow:0
upgray:0
upblue:0
upgreen:0
upbrown:-1
downblack:0
downwhite:1
downred:0
downpurple:0
downyellow:0
downgray:0
downblue:0
downgreen:0
downbrown:0



train cmd:
python3  demo.py  --data-path  ./dataset  --dataset  duke  --model  resnet50
python3  train.py  --data-path  ./dataset  --dataset  duke  --model  resnet50_softmax