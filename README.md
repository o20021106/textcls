# textcls
訓練文章分類器。共有三個模型:CNN、LightGBM、Logistic
<br/>
<br/>
### 1. 分詞與清理資料
先將資料放```data/```(內含title content category_int)轉成csv<br/>
category_int是文章類別<br/>
```python preprocess.py -f filename```
<br/>
<br/>
### 2. 訓練CNN模型<br/>
```python cnn_train.py```
<br/>
<br/>
### 3. 訓練LightGBM模型<br/>
```python lgbm_train.py```
<br/>
<br/>
### 4. 訓練Logistic模型<br/>
```python logistic_train.py```
<br/>
<br/>
### 5. 使用模型預測
```python inference.py -f test --cnn cnn.h5 --lgbm lgbm.pickle --logistic logistic.pickle```
