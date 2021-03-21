# Surrogate models help black box adversarial attacks
## Machine Learning 2021 Course by E. Burnaev, A. Zaytsev et al., Skoltech

Team members: Matvey Morozov, Anna Klueva, Elizaveta Kovtun, Dmitrii Korzh

### Introduction

Adversarial attack is a way to exploit the non-robustness of deep learning modes, it means that slight modifications of the input may lead to the inability of the model to get the correct answer. In this project, we consider a modification of boundary black-box adversarial attacks on deep neural networks for image classification problem. In the process of generating examples for an attack, we use an additional step based on a surrogate model for the attacked model.

Our implementation for Substitute Boundary Attack is based on FoolBox framework implemenation https://foolbox.readthedocs.io/en/stable/index.html
