Python 3.10 or newer

**Dataset**

With Data for walker.py
We can download all datasets by running
```bash
source download_datasets.sh
```
OR Link Download: https://people.csail.mit.edu/mlechner/datasets/walker.zip  

This script creates a folder ```data```, where all downloaded datasets are stored.

And script creates a folder ```results```.

In folder ```CLOSED_FORM_CONTINUOUS_TIME_NeuralNetworks```
## 0. Clone the repository:
```bash
git clone https://github.com/inin-panda/CLOSED_FORM_CONTINUOUS_TIME_NeuralNetworks.git
cd CLOSED_FORM_CONTINUOUS_TIME_NeuralNetworks
```

## 1. Create and activate a virtual environment:
- For Unix/macOS:
```bash
python3 -m venv venv
source venv/bin/activate
```
- For Windows:
```bash
python -m venv venv
.\venv\Scripts\activate
```

## 2. Install the required dependencies:
```bash
pip install -r requirement.txt
```

## Module description:
- ```tf_cfc.py``` Implementation of the CfC (various versions) in Tensorflow 2.x
- ```irregular_sampled_datasets.py``` Same splits Datasets



## 3. Training and evaluating the models:
There is exactly one Python module per dataset:
### 3.1. Usage with datasets: XOR, IMDB, Walker2D
All `3.1._` training scripts except the following three flags
- ```minimal``` Runs the CfC direct solution
- ```use_ltc``` Runs an LTC with a semi-implicit ODE solver instead of a CfC 
- ```use_mixed``` Mixes the CfC's RNN-state with a LSTM to avoid vanishing gradients
- If none of these flags are provided, the full CfC model is used
-------------------------------------

Each script accepts the following two arguments:
- ```--minimal | --use_ltc ```
- ```--epochs: number of training epochs (default 20)```
#### 3.1.1- XOR Binary Classification : ```train_xor.py``` 

- Example trains CfC:
```Bash: 
    #train full CfC model: default 20 epochs.  
python train_xor.py  

    #train CfC model CfC direct solution: default 20 epochs. 
python train_xor.py --minimal  

    #train  CfC model CfC direct solution: 10 epochs.
python train_xor.py --minimal --epochs 10 
```

#### 3.1.2- Walker2D Reinforcement Learning: ```train_walker.py``` 
- Example trains CfC:
```Bash: Example trains CfC
    #train full CfC model: default 20 epochs.  
python train_walker.py  

    # train CfC model CfC direct solution: default 20 epochs.  
python train_walker.py --minimal 

    #train  CfC model CfC direct solution: 10 epochs.
python train_walker.py --minimal --epochs 10
```

#### 3.1.2- IMDB : ```train_imdb.py``` 
- Example trains CfC:
```Bash: Example trains CfC
    #train full CfC model: default 20 epochs.  
python train_imdb.py  

    # train CfC model CfC direct solution: default 20 epochs.  
python train_imdb.py --minimal 

    #train  CfC model CfC direct solution: 10 epochs.
python train_imdb.py --minimal --epochs 10
```

### 3.2. Usage with dataset: EtsMNIST
Each script accepts the following two arguments:
- ```--model: cfc | ltc ```
- ```--epochs: number of training epochs (default 50)```
#### - Event-based MNIST for Continuous-Time Models : ```train_et_smnist.py``` Trains the CfC model.

- Example trains CfC:
```Bash: Example trains CfC
    # train CfC + number of training default 50
python train_et_smnist_ltc.py

    # train LTC + number of training default 50
python train_et_smnist_ltc.py --model ltc 

    #train full CfC model: number of training 10 epochs 
python train_et_smnist_ltc.py --model cfc --epochs 10 
```

## References
| Models | References |
| ----- | ----- |
| Liquid time-constant Networks | https://arxiv.org/abs/2006.04439 |
| Neural ODEs | https://papers.nips.cc/paper/7892-neural-ordinary-differential-equations.pdf |
| Continuous-time RNNs | https://www.sciencedirect.com/science/article/abs/pii/S089360800580125X |
| A Tutorial on Liquid Neural Networks including Liquid CfCs| https://ncps.readthedocs.io/en/latest/quickstart.html |
| Closed-form Continuous-time Neural Networks (CfCs)| Paper Open Access: https://www.nature.com/articles/s42256-022-00556-7 |
| Closed-form Continuous-time Neural Networks (CfCs)| Arxiv: https://arxiv.org/abs/2106.13898 |

