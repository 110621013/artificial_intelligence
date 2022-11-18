# 環境建置
###### tags: `envir`

## 硬體配置
* OS: windows11 x64
* CPU: 11th Gen Intel(R) Core(TM) i7-11800H@2.30GHz
* GPU: GeForce RTX 3060(notebook)
* RAM: 16GB
## 軟體與環境配置
* edit/run: VScode
* envirment: conda
    * python: 3.7.13
    * torch: 1.12.1 (pytorch-mutex: cuda 1.0)
    * numpy: 1.21.6
    * pandas: 1.2.3
    * matplotlib: 3.2.2
    * gym: 0.26.2
    * gym_anytrading: 1.3.2
    * tqdm: 4.64.1
    * stable-baselines3: 1.4.0
    * os
    * json
## 執行方法
train.py內的if __name__ == '__main__'有兩個函數
第一個是main()，其負責執行PolicyGradient的部分，包括環境建立、PolicyGradientNetwork、PolicyGradientAgent、訓練、儲存agent model、show Total Rewards以及測試，注意路徑的部分會吃2330_stock.csv在.py檔案所在之處，並且儲存/吃.ckpt檔案在.py檔案所在之處。
第二個是stable_baselines3_main()，大同小異負責執行stable_baselines3套件的部分，包括環境建立、training、Evaluation、testing和輸出

test.py內則固定參數不動，並且PolicyGradient的Network跟Agent已經跟train.py對齊無須變動，針對.ckpt檔案進行給定範圍的輸出，吃.ckpt檔案在.py檔案所在之處。(test.py只對main()的PolicyGradient部分輸出，stable_baselines3_main()自己會輸出的)