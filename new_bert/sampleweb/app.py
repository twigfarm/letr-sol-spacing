from flask import Flask, render_template, request
import torch
import pytorch_lightning as pl
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from io import StringIO
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

from kospacing import preprocessor
from kospacing import dataset
from kospacing import net

Preprocessor=preprocessor.Preprocessor
CorpusDataset=dataset.CorpusDataset
SpacingBertModel=net.SpacingBertModel

app = Flask(__name__)

def get_dataloader(data_path, transform, batch_size):
    dataset = CorpusDataset(data_path, transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)
    return dataloader


def main(config):

    preprocessor = Preprocessor(config.max_len)
    test_dataloader = get_dataloader(
        config.test_data_path, preprocessor.get_input_features, config.eval_batch_size
    )
    model = SpacingBertModel(config, None, None, test_dataloader)
    checkpoint = torch.load(config.ckpt_path, map_location=lambda storage, loc: storage)
    model.load_state_dict(checkpoint["state_dict"])

    trainer = pl.Trainer()
    res = trainer.test(model)
    return res

# 입력 데이터 공백제거
def remove_spacing(sentences):
    return str(sentences).replace(' ', '')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    input_text = request.form['text']
    text=remove_spacing(input_text)

    with open('../kospacing/data/sample.txt', 'w', encoding='utf-8') as f:
        f.write(text)

    config = OmegaConf.load("./config/eval_config.yaml")
    
    predict=main(config)['result'][0]
  
    return render_template('index.html', inputText=input_text, prediction=predict)


if __name__ == '__main__':
    app.run(debug=True)