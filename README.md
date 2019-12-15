## Non - Gaussian Behaviour of Stochastic Gradient Noise in Deep Learning

#### Candidate Number: 1040706

In recent years, there has been a growing interest in Stochastic Gradient Descent
(SGD) and its modifications in the field of machine learning, mainly due to its
computational efficiency. It is often assumed that gradient noise follows Gaussian
distribution in large data-sets by invoking the __classical Central Limit Theorem__.
However, the results in my report (will be published here shortly) shows that this is far from true, in fact we show that
__stochastic gradient noise (SGN)__ follows an alpha-stable distribution, which is a family
of heavy tailed distribution where alpha is a tail index. For validation, we build two models from scratch by just the use of *numpy* for vector operations. We only use *keras* to import MNIST and Fashion-MNIST datasets. The models try to show results on two questions: 

* __Does the choice of activation function have a big effect on distribution of SGN?__: for this we run the tests using `ReLu` and `sigmoid` where the implementation can be found in `model_epoch_vs_alpha.py`. I have run a test on the file `test_epoch_alpha_relu.py` where the outputs can be seen in the folder `MNIST_activation`. Please adjust this file accordingly to change the activation function and datasets. All the documentation are provided in the `doc-strings` in the models. 
vary the choice of the activation function and learning rate to show that SGN is far
from Gaussian and its distribution is very sensitive to these changes.
