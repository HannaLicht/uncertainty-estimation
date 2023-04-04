This repository contains the source code for the bachelor's thesis "Vergleich ausgewählter Methoden zur
Uncertainty Estimation und deren
Verwendung für aktives Lernen" (=Comparison of selected methods for uncertainty estimation and their use for active learning).

## Abstract

Due to many advances in the field of artificial neural networks, there is a desire to use them in critical situations. For this, trust in model outputs is necessary. High accuracy of the networks is also an important factor. Estimating predictive uncertainty, known as *Uncertainty Estimation*, enables predictions about the correctness of outputs, thereby increasing confidence in the model outputs. Predictive uncertainty can also be used to improve the efficiency of labeling as part of the training process of a network, so that its accuracy grows faster. One such approach to optimize the training is the *Uncertainty-Based Active Learning*.

In this repository, selected methods for estimating predictive uncertainty are implemented and compared based on various quality criteria. The tests focus on the interpretability and the information content of uncertainty estimates as well as the runtime of the methods, among other factors. The tests are conducted on the Cifar10, Cifar100, and Cars datasets. The performance of the methods for Uncertainty-Based Active Learning is additionally evaluated to determine to which extent the Active Learning improves the training process. The goal is also to determine which Uncertainty Estimation methods are best suited for Active Learning. Furthermore, we examine a hybrid query strategie for Active Learning that promises to further increase a network's accuracy faster.


## Project Structure

The implementations of selected methods for uncertainty estimation are located in the "Uncertainty" folder. Models can be trained by running the files in the "Models" folder and are saved in the same folder for subsequent testing. The "Evaluation" folder contains various tests for evaluating the methods with respect to a model. The results are saved in the "Results" folder. Some files in the "Evaluation" folder generate plots and tables for the analysis of these results. The plots created for the bachelor's thesis are located in the "Plots" folder. Afer training some models, you can do a simple uncertainty estimation with some evaluation by running one of the files starting with "test" in the folder "Evaluation".
