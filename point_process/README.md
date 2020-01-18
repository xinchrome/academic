# point process


### Introduction

This project is an experiment for modeling events sequence data based on temporal point process.


### Deployment

On Ubuntu 16.04 (with Python2.7), run the following commands as root in the terminal:

	$ apt update
	$ apt -y install git python-pip python-tk
	$ pip install numpy matplotlib tensorflow keras

If you have already installed these packages, please upgrade them to latest version. To show all of the experiment results, go to the root directory of this project, run:

	$ python draw_pp.py
	$ python draw_spatial.py

Then the output curves will be saved in folder "pic/".

### Experiments

##### Pretraing with MLE only

Firstly, we train our point process by maximizing log likelihood on observed sequence, and predict events in the future time interval by time interval. The learning curve is shown as follows.

We can see that the objective declines as the training proceeds, that the objective decreases dramatically each time we employ our efficient EM algorithm. However, pure MLE may lead to overfitting quickly, i.e. too much training on observed sequence leads to performance degredation on long-term prediction.

To perform pre-training experiment, go to foler 'pp_gan/' , run:

	$ python generator.py

The the training log will be saved in folder "log/".

##### Training with Various Losses

After performing early stopping of MLE, we will further train our model to gain more accuracy.

We *discretize the timestamp of each event*, this approximate approach leads to deterministic expectation of total events in each time interval, which enables the gradients computation of any loss function w.r.t. parameters in point process model, without losing prediction accuracy in the experiment. This method enables further enhance of accuracy provided that MLE has already overfitted. We choose mean squared error on the validation sequence as our objetive functions, empirical results on real data (e.g. paper citations) has shown the efficacy.

We notice that $\beta$ must be fixed in MSE-only experiment, otherwise the loss may be rather unstable and even becomes invalid value during training process. In order to test the potential ability of MSE-only method, we fix $\beta$ to be the optimal value estimated by MLE.

Gaussion noise on the output of generator is useful to mitigate overfitting. The number of samples is always limited, which may distort the real distribution of observed sequence, we could see the noise as a form of random data augmentation. Note that the noise is only active at training time.


### Referrences

 The detailed description for algorithms can be found in 

- [https://www.ijcai.org/Proceedings/16/Papers/380.pdf](https://www.ijcai.org/Proceedings/16/Papers/380.pdf)

- [http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14385/13934](http://aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewFile/14385/13934)

