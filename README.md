# Named Entity Recognition

中文命名实体识别

数据集：https://github.com/SophonPlus/ChineseNlpCorpus (dh_msra命名实体识别，下载后放在 data 文件夹中)

所用字向量：https://github.com/SophonPlus/ChineseWordVectors (polyglot_wiki 字向量，解压后放在 data 文件夹中)

## Train and Test
快速训练 + 交叉验证 + 测试，具体参数设定见main.py文件
````
python main.py -model_name BiLSTM or BiLSTM-CRF \
                -do_train True \
                -do_cv True \
                -do_test True
````

## Results
超参数一致，没有认真调参，仅供参考。验证集分数为5折交叉验证的F1分数平均值，测试集为5折最优模型的投票结果

| Model_name  | Dev F1 | Test F1 |
| ------------- | ---- | ---- | 
| BiLSTM | 0.8896 | 0.9079 |
| BiLSTM-CRF | 0.9045  | 0.9212 |

## TODO
- Lattice LSTM
