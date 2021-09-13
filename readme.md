## Weakly-supervised Text Classification Based on Keyword Graph

## How to run?

### Download data
Our dataset follows previous works. For long texts, we follow Conwea. For short texts, we follow LOTClass.   
We transform all their data into unified json format.    
1. Download datasets from:
https://drive.google.com/drive/folders/1D8E9T-vuBE-YdAd9OBy-yS4UW4AptA58?usp=sharing    
    * Long text datasets(follow Conwea):
        * 20Newsgroup Fine(20NF)
        * 20Newsgroup Coarse(20NC)
        * NYT Fine(NYT_25)
        * NYT Coarse(NYT_5)
  
    * Short text datasets(follow LOTClass)
        * Agnews
        * dbpedia
        * imdb
        * amazon

2. Unzip data into './data/processed'


Another way to obtain data (Not recommended):   
You can download long text data from [Conwea](https://github.com/dheeraj7596/ConWea/tree/master/data) and short text data from [LOTClass](https://github.com/yumeng5/LOTClass/tree/master/datasets) and transform data into json format using our code.
The  code is located at 'preprocess_data/process_long.py (process_short.py)
You need to edit the preprocess code to change the dataset path to your downloaded path and change the taskname. The processed data is located in 'data/processed'.
We alse provide preprocess code for [X-class](https://arxiv.org/abs/2010.12794), which is 'process_x_class.py'.


### Requirements
This project is based on python==3.8. The dependencies are as follow:
```
pytorch
DGL
yacs
visdom
transformers
scikit-learn
numpy
scipy
```

### Train and Eval
* Recommend to start visdom to show the results.
```
visdom -p 8888
```
Open the browser to the server_ip:8888 to show visdom panel.

* Train:
    * First edit 'task/pipeline.py'  to specify to config file and CUDA devices you used.    
      Some configuration files are provided in the `config` folder.
    * Start training:
      ```
      python task/pipeline.py
      ```

    * Our code is based on multi GPUs, may be unable to run on single GPU currently.

### Run on your custom dataset.
1. provide datasets to dir `data/processed`.
    * keywords.json    
      keywords for each class. type: dict. key: class_index. value: list containing all keywords for this class.
      See provided datasets for details.
      
    * unlabeled.json    
      unlabeled sentences in our paper. type: list. item: list with 2 items([sentence_i,label_i]).    
      In order to facilitate the evaluation, we are similar to Conwea's settings, where labels of sentences are provided.
      The labels are only used for evaluation.
      
2. provide config to dir `config`. You can copy one of the existing config files and change some fields, like number_classes, classifier.type, data_dir_name etc.

3. Specify the config file name in `pipeline.py` and run the pipeline code.

### Citation
Please cite the following paper if you find our code helpful! Thank you very much.
> Lu Zhang, Jiandong Ding, Yi Xu, Yingyao Liu and Shuigeng Zhou. "Weakly-supervised Text Classification Based on Keyword Graph". EMNLP 2021.
> 
      