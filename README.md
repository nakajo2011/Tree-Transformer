# What is
「Explainability of Transformer for Large Over-the-Top Media Viewing Logs」のPCFG代理モデルとしてTree構造を生成する実装。

https://github.com/yaushian/Tree-Transformer をベースにしてtvvbertに適応するように改造。

## Dependencies

* python3
* pytorch 1.0
* transformers
* nltk
* svg2png
* cairosvg

We use BERT tokenizer from [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers) to tokenize words. Please install [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers) following the instructions of the repository.  


## Training
For grammar induction training:  
```python3 main.py -train -model_dir [model_dir] -num_step 60000```  

このリポジトリには視聴行動データを600000step分学習させたモデルをtrain_model以下に格納済み。

## Evaluation
For grammar induction testing:  
```python3 main.py -test -seq_length 512```  
The code creates a result directory named model_dir. The result directory includes 'bracket.json' and 'tree.txt'. File 'bracket.json' contains the brackets of trees outputted from the model and they can be used for evaluating F1.
The 'datas' dir include constituent attention weight datas.

## Generate Tree Graph
Exec After evaluation:
```
python3 main.py -graph
```

## Acknowledgements
* Our code is mainly revised from [Annotated Transformer](http://nlp.seas.harvard.edu/2018/04/03/attention.html).  
* The code of BERT optimizer is taken from [PyTorch-Transformers](https://github.com/huggingface/pytorch-transformers).  

## Contact
nakajo@chaintope.com

base creator: king6101@gmail.com
