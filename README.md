## bert-for-korean-spacing
BERT Pretrained model을 이용한 한국어 띄어쓰기

## Dataset
* KcBERT 학습용 데이터셋 중 1,250,000건 사용
* Korpora 패키지를 이용해 다운로드

## Train
* train_config.yaml
```yaml
log_path: logs
bert_model: monologg/kobert
train_data_path: data/kcbert/train_data.txt
val_data_path: data/kcbert/val_data.txt
test_data_path: data/kcbert/test_data.txt
max_len: 128
train_batch_size: 64
eval_batch_size: 64
dropout_rate: 0.1
gpus: 8
distributed_backend: ddp
```

```python
python train.py
```

## Eval
* eval_config.yaml
```yaml
bert_model: monologg/kobert
test_data_path: data/kcbert/test_data.txt
ckpt_path: checkpoints/epoch=4_val_acc=0.000000.ckpt
max_len: 128
eval_batch_size: 64
dropout_rate: 0.1
```

```python
python eval.py
```
## Results
* 

## Example
* 

## Reference
* https://github.com/Beomi/KcBERT
* https://github.com/monologg/KoBERT-Transformers
* https://github.com/ko-nlp/Korpora
* https://github.com/PyTorchLightning/pytorch-lightning
* https://github.com/omry/omegaconf