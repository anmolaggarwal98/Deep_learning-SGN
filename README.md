## Non - Gaussian Behaviour of Stochastic Gradient Noise in Deep Learning

#### Candidate Number: 1040706

In recent years, there has been a growing interest in Stochastic Gradient Descent
(SGD) and its modifications (just as AdaDelta and Adam) in the field of machine learning, mainly due to its
computational efficiency. It is often assumed that gradient noise follows Gaussian
distribution in large data-sets by invoking the __classical Central Limit Theorem__.
However, the results in my report (`will be published here shortly`) shows that this is far from true, in fact we show that
__stochastic gradient noise (SGN)__ follows an alpha-stable distribution, which is a family
of heavy tailed distribution where alpha is a tail index. For validation, we build two models from scratch by just the use of *numpy* for vector operations. We only use *keras* to import MNIST and Fashion-MNIST datasets. The models try to show results on two questions: 

* __Does the choice of activation function have a big effect on distribution of SGN?__: for this we run the tests using `ReLu` and `sigmoid` where the implementation can be found in `model_epoch_vs_alpha.py`. I have run a test on the file `test_epoch_alpha_relu.py` where the graphs can be seen in the folder `MNIST_activation`. Please adjust this file accordingly to change the activation function and datasets. All the documentation are provided in the `doc-strings` in the models. I have also attached the jupyter notebook `epoch_alpha.ipynb` which although outdates and not well documented, shows you my progress and also plots. 

* __Does the choice of learning rate effect the distribution of SGN?__: for this we run the tests using `ReLu` and `sigmoid` and we adjust the learning rate from __0.001 to 0.1__ with an increment which is user defined. The implementation can be found in `model_lr_vs_alpha.py`. I have run a test on the file `test_lr_alpha.py` where the graphs can be seen in the folder `MNIST_lr`. The jupyter notebook attached for this is `lr_epoch.ipynb`

My report is heavily based on the research paper: `http://proceedings.mlr.press/v97/simsekli19a/simsekli19a.pdf` which you meant find useful to understand what and why I have done this project

##### Installation: 

Please feel free to `clone` this repository and play around with the code. I have tried to keep the documentation in the doc strings above the function as understandable as possible.
