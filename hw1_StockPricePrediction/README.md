# 環境建置
###### tags: `envir`

## 硬體配置
* OS：windows11 x64
* CPU：11th Gen Intel(R) Core(TM) i7-11800H@2.30GHz
* GPU：GeForce RTX 3060(notebook)
* RAM：16GB
## 軟體與環境配置
* edit/run：VScode
* envirment：conda
    * python：3.7.13
    * tensorflow：1.14.0
    * torch：1.12.1 (pytorch-mutex：cuda 1.0)
    * sklearn：1.0.2
    * numpy：1.21.6
    * pandas：1.2.3
    * matplotlib：3.2.2
    * os
    由於tensorflow執行時在load/save model版本與h5py會有不相容，因此上述套件完成安裝後需要進行install h5py==2.10降版本
## 執行方法
在main.py中會先執行import，並且有多個def出來的function，這些function在if __name__ == "__main__":中被執行，可根據report.pdf內提及的題號與對應function進行：
0. multi_linear_test()、dnn_test()、lstm_test()
1. q1()
2. q2()
3. q2()
4. q4()
5. q5()