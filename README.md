Carlini-Wagner Adversarial Attack on MNIST

This tutorial implements the Carlini-Wagner (C&W) targeted adversarial attack against a pretrained CNN for MNIST digit classification.

Goal

Instead of training the CNN, the objective is to find the smallest perturbation δ that changes the prediction of a correctly classified image from 7 to the target class 3, while keeping the perturbation as small as possible.

Main idea

The original CNN is first trained on the MNIST dataset and then frozen. During the attack, the CNN weights are never updated.

A custom perturbation layer is inserted before the CNN:

Input image
      │
      
Perturbation layer (x + δ)
      │
      
Frozen CNN
      │
      
Logits

The perturbation δ is the only trainable parameter. It has the same spatial dimensions as the input image (28 × 28 × 1), meaning that each pixel has one learnable perturbation value.

Optimization

Unlike normal neural network training, Adam does not update the CNN weights.

Instead, during every epoch:

The current adversarial image x+δ is passed through the frozen CNN.
The logits are used to compute the Carlini-Wagner classification loss.
The perturbation regularizer is added.
Backpropagation computes gradients with respect to δ only.
Adam updates δ while the CNN remains unchanged.

Thus, the same image is repeatedly presented to the model, but the perturbation becomes slightly different after every optimization step.

Carlini-Wagner loss

The attack minimizes two objectives simultaneously:

1. Perturbation regularizer

Task 1:

||δ||²

This encourages the perturbation to remain as small as possible.

Task 2:

The L2 regularizer is replaced with the Hoyer-square regularizer

(sum(|δ|))²
-------------------
sum(δ²) + ε

which encourages sparse perturbations, i.e., changing only a small number of pixels.

2. Classification objective

The attack is targeted, meaning that the image should specifically be classified as digit 3.

The classification loss is

max(highest_other_logit − target_logit, −κ)

where

highest_other_logit is the largest logit among all classes except the target.
target_logit is the logit corresponding to digit 3.
κ controls the confidence of the attack.

Because the optimizer minimizes the loss, it tries to increase the target logit until it becomes larger than every other class.

Hyperparameters

Two important hyperparameters control the attack:

λ (lambda): balances successful misclassification against perturbation size.
Larger λ → stronger attack but usually larger perturbation.
Smaller λ → smaller perturbation but attack may fail.
κ (kappa): controls the confidence of the target prediction.
κ = 0 only requires the target class to become the predicted class.
Larger κ forces the target class to win by a larger margin, usually requiring a stronger perturbation.
Evaluation

The attack should not be evaluated using the training loss alone.

Instead, the important criteria are:

Does the adversarial image get classified as the target class (digit 3)?
What is the confidence (softmax probability) of class 3?
What is the L2 norm of the perturbation?
How many pixels were significantly modified (for the sparse perturbation task)?
What I learned
The CNN remains completely fixed during the attack.
Only the perturbation δ is optimized.
Gradients are still propagated through the frozen CNN, but only δ receives parameter updates.
Keras automatically passes the current model predictions (y_pred) to the custom loss during every epoch.
Splitting the objective into a perturbation regularizer (implemented in the custom layer) and a classification loss (implemented as the custom loss function) simplifies the implementation and follows the original Carlini-Wagner formulation.
