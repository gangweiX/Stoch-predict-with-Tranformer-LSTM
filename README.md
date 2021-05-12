# stoch-predict-with-Tranformer-LSTM
stock predict with MLP,CNN,RNN,LSTM,Transformer and Transformer-LSTM  
Environment
============
    1.Python 3.8  
    2.PyTorch  
    3.TorchVision
Install  
============
Create a virtual environment and activate it.  
-------------
    conda create -n stock_predict python=3.8  
    conda activate stock_predict  
The code has been tested with PyTorch 1.8 and Cudatoolkit 11.1.  
----------------------
    conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia  
    pip3 install pandas  
    pip3 install matplotlib  
    pip3 install tqdm  
    pip3 install tensorboardX  
Train/evaluate
=============
To train and evaluate our model, you can run  
---------------
    python3 main.py
Plot
======
To plot the figure of stoch predict,you can run  
------------
    python3 plot.py  
    ![image](https://github.com/gangweiX/stoch-predict-with-Tranformer-LSTM/blob/main/plot_figure/model_transformer-lstm.png) 
    
    
