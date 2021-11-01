# bert-for-korean-spacing

### BERT Pretrained model을 이용한 한국어 띄어쓰기

NLP과정에서는 한국어 텍스트 정보의 의미를 효과적으로 처리하기 위해서는 각 단어를 정확하게 파악하는 것이 중요합니다. 특히 잘못 된 띄어쓰기는 단어 구별 오류의 원인이 되며 문장 해석의 어려움 증가로 성능 저하를 초래합니다.   

bert-for-korean-spacing은 위와 같은 문제를 해결하기 위해 띄어쓰기를 학습 한 모델입니다.


## Dataset
* 모두의 말뭉치 문어 데이터 53,201문장

<img width="50%" src="https://user-images.githubusercontent.com/77109972/139565991-163adba8-06d6-454f-b807-ca38417a24b3.png"/>

  파라미터 max_len과 문장의 길이가 얼만나 고르게 분포되어 있는지가 학습에 중요한 요소로 판단합니다.  
  띄어쓰기가 제대로 된 데이터를 수집하는 것이 중요합니다. (예: 국립국어원 데이터, 교과서 

## Train

#### Requirements
<a href="https://download.pytorch.org/whl/torch_stable.html" target="_blank">
<img src="https://img.shields.io/badge/PyTorch-1.5.1-EE4C2C?style=flat-square&logo=PyTorch&logoColor=white"/>
</a>
```
pip install torch===1.5.1 torchvision===0.6.1 -f https://download.pytorch.org/whl/torch_stable.html -i https://pypi.douban.com/simple
```
#### train.txt, test.txt, val.txt 

```python
from sklearn.model_selection import train_test_split
import os
from tqdm import tqdm

# 아무 말뭉치 데이터 수집
TRAIN_DATA_FILE = '말뭉치.txt'
sents = [s[:-1] for s in open(TRAIN_DATA_FILE, encoding='utf-8').readlines()]

results=[]
for i in tqdm(sents):
    if i =='':
        continue
    results.append(i)

# train, test 비율을 9:1로 맞춘다.
test_size = int(len(results) * 0.1)
train, val = train_test_split(results, test_size=test_size, random_state=111)
train, test = train_test_split(train, test_size=test_size, random_state=111)

def write_dataset(file_name, dataset):
    with open(
        os.path.join("./data", file_name), mode="w", encoding="utf-8"
    ) as f:
        dataset
        for data in dataset:
            f.write(data+'\n')

write_dataset("train.txt", train)
write_dataset("val.txt", val)
write_dataset("test.txt", test)
```

* train_config.yaml

  config/train_config.yaml에서 학습 데이터의 경로를 확인합니다.
```yaml
log_path: logs
bert_model: monologg/kobert
train_data_path: data/train_data.txt
val_data_path: data/val_data.txt
test_data_path: data/test_data.txt
max_len: 128
train_batch_size: 64
eval_batch_size: 64
dropout_rate: 0.1
gpus: 8
distributed_backend: ddp
```
if you don't have gpu, gpus must be change to 0 (gpus:8 -> gpus:0)

#### Run!

```python
cd kospacing
python train.py
```

## Eval
* eval_config.yaml

  data/sample.txt 파일을 만들어서 검사를 할 문장을 입력합니다.
```yaml
bert_model: monologg/kobert
test_data_path: data/sample.txt
ckpt_path: checkpoints/epoch=4_val_acc=0.000000.ckpt
max_len: 128
eval_batch_size: 64
dropout_rate: 0.1
```

- net.py의 test_step_outputs을 변경하여 출력 결과를 선택할 수 있습니다.

```python
test_step_outputs = {"result":prd_result(pred_labels),
            "test_acc": test_acc, 
            "gt_labels": gt_labels,
            "pred_labels": pred_labels}
```

#### Run!

```python
cd kospacing
python eval.py
```
## Results

* testset :  5320
* Accuracy : 0.948 

## Example
> input  : 그냥영풍이라고써있으니까될거같지않냐?  
> output : 그냥 영풍이라고 써 있으니까 될 거 같지 않냐?

> input  : 대표적인미디어문화연구자인더글러스켈너는이렇게말하고있다.  
> output : 대표적인 미디어 문화연구자인 더글러스 켈너는 이렇게 말하고 있다.

> input  : 트렁크룸사업의성장성은이례적이다.	  
> output : 트렁크 룸사업의 성장성은 이례적이다.

## Demo

```python
cd sampleweb
python app.py
```

![화면-기록-2021-10-31-오전-11 52 24](https://user-images.githubusercontent.com/77109972/139566025-5b3771cb-a1c3-43ba-95bc-ec23730ac7c0.gif)

## Reference

#### GitHub

* [KoBERT-Transformers](https://github.com/monologg/KoBERT-Transformers)
* [PyTorchLightning](https://github.com/PyTorchLightning/pytorch-lightning)
* [OmegaConf](https://github.com/omry/omegaconf)

#### Paper

- [한국어 자동 띄어쓰기 시스템을 위한 학습 데이터의 효과적인 표현 방법](https://www.earticle.net/Article/A367406)
- [BERT를 이용한 한국어 자동 띄어쓰기](http://isoft.cnu.ac.kr/paper/[KSC2019]BERT%EB%A5%BC%20%EC%9D%B4%EC%9A%A9%ED%95%9C%20%ED%95%9C%EA%B5%AD%EC%96%B4%20%EC%9E%90%EB%8F%99%20%EB%9D%84%EC%96%B4%EC%93%B0%EA%B8%B0.pdf)

*** 
```
new_bert
└kospacing
	└config
  	   - eval_config.yaml
	   - train_config.yaml
 	└data
	   - sample.txt
	   - test.txt
	   - train.txt
	   - val.txt
	- dataset.py
	...
└sampleweb
	└config
	   - eval_config.yaml
	└static
	└templates
	- app.py
	...
```
