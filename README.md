# textcls 文章分類器
這是一個簡易版的文章分類器，可用於分類網路文章。
資料預處理的部份包含將文章標題和內容的HTML標籤去除（使用bs4)以及分詞（使用jieba）
資料預處理完後進行模型的訓練，共有三個模型:CNN、LightGBM、Logistic
最後用簡單的blending選擇在三個模型中獲得最多票的分類為最終預測

## 訓練
### 1. 分詞與清理資料
先將資料放```data/input/input_filename.csv```(內含title 、content、category_int)轉成csv<br/>
category_int是文章類別<br/>
```python preprocess.py -f input_filename```
<br/>
### 2. 訓練CNN模型<br/>
```python cnn_train.py```
<br/>
### 3. 訓練LightGBM模型<br/>
```python lgbm_train.py```
<br/>
### 4. 訓練Logistic模型<br/>
```python logistic_train.py```
<br/>
## 預測
將欲預測的資料放到```data/input/input_filename.csv```(內含title 、content)轉成csv<br/>
```python inference.py -f input_filename -p prediction_filename```
預測會使用三個模型預測投票的結果為最終預測結果，並存到```data/predictions/predictionn_filename.csv```
難有三個欄位：title、content、category_int_prediction
