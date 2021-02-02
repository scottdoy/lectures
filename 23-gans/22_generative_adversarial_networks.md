---
# Most of these won't change from one presentation to the next
fontsize: 10pt
documentclass: beamer
classoption:
- xcolor=table
- aspectratio=169
theme: metropolis
slide-level: 2

# Change this stuff for each talk
title: |
    | Generative Adversarial
    | Networks
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
institute: scottdoy@buffalo.edu
date: May 02, 2019
---

# Recap: (Variational) Autoencoders
## Recap: Image Manifold

![Illustration of Image Manifold (CIFAR-10)](../imgs/22_cnn_manifold.png){width=75%}

## Recap: Image Manifold Sampling

![Sampling the Manifold](../imgs/22_cnn_manifold_sample.png){width=75%}

## Recap: Autoencoders

Autoencoders do two things:

1. Build an image-space manifold
2. Sample the manifold to generate samples

## Recap: Autoencoder Diagram

![Autoencoder Schematic](../imgs/22_autoencoder_diagram.pdf){width=90%}

## Recap: Undercomplete Autoencoders

If $\mathbf{x} \in \mathbb{R}^{d}$, then making
$\mathbf{h}\in\mathbb{R}^{\widehat{d}}$, where $\widehat{d}<d$, forces the
encoder to "learn" the most important parts of $\mathbf{x}$.

This is ***bottlenecking*** or ***limited bandwidth***, and results in an
***undercomplete autoencoder***.

## Recap: Training

***Regularization*** takes on many forms:

***Loss Function***: 
: $L(\mathbf{x}, g(f(\mathbf{x})))$

***Sparse Autoencoder***: 
: $L(\mathbf{x}, g(f(\mathbf{x}))) + \Omega(\mathbf{h})$

***Contractive Autoencoder***: 
: $L(\mathbf{x}, g(f(\mathbf{x}))) + \Omega(\mathbf{h}, \mathbf{x})$

***Denoising Autoencoders***: 
: $L(\mathbf{x}, g(f(\widetilde{\mathbf{x}})))$

## Recap: VAE

***Variational Autoencoders*** (VAEs) define the encoder and decoder
probabilistically:

$$ q_{\phi}(\mathbf{h}|\mathbf{x}) \qquad \textrm{Encoder} $$
$$ p_{\theta}(\widetilde{\mathbf{x}} | \mathbf{h}) \qquad \textrm{Decoder} $$

To avoid degenerate solutions, we force the system to implement
a ***conjugate prior*** in the form of a spherical unit Gaussian:

$$ \mathbf{h}\sim\mathcal{N}(0,I) $$

## Recap: MNIST Dataset

![MNIST Digits Database](../imgs/22_mnist_examples.png){width=35%}

## Example: Epoch 10

\begincols

\column{0.4\linewidth}

![VAE Reconstruction, Epoch 10](../imgs/22_vae_reconstruction_10.png){width=100%}

\column{0.6\linewidth}

![VAE Sampling, Epoch 10](../imgs/22_vae_sample_10.png){width=70%}

\stopcols

## Example: Latent Walks

![Latent Walk](../imgs/22_latent_space_30.png){width=45%}

# Adversarial Networks
## Adversarial Training Overview
***Regularization***: The goal is to ensure that we learn a hidden latent space
without over-training (just copying things over).

There's another way to train these kinds of networks, and that's using
***Adversarial*** training.

## Adversarial Training Intuition

Ever hear of the ***Turing test***?

Video: [Brian Christian, "The Most Human Human"](https://youtu.be/lFIW8KphZo0?t=40):

\centering
https://www.youtube.com/watch?v=lFIW8KphZo0

## Judging Generated Outputs

![VAE Reconstruction, Epoch 5](../imgs/22_vae_reconstruction_5.png){width=80%}

When we see the results of a generative network (e.g. an autoencoder), how do we
judge how "good" it is?

## Turing Loss Function

In generative networks, humans look at the output and decide how likely it is
that an AI "drew" the sample, or if a human did.

If it can fool us, then we would say that the network has successfully been
trained.

What if we use that feedback to ***train*** the network in the first place? We
might call this a "Turing Loss" function.

## Human Judges

Obviously, for image-based autoencoders, we can't have a person sift through
millions of generated samples hundreds of times just to provide a numeric loss
function for each output in terms of whether or not it's a "real" or "fake"
generated sample.

## Any Volunteers?

![Who wants to judge each of these samples?](../imgs/22_latent_space_30.png){width=45%}

## Human Judges

That would be tedious, expensive, time-consuming, and error-prone.

If only there was a robot who could sit around and judge generated samples all
day...

## It's AI All the Way Down

![Generative Adversarial Network](../imgs/23_gan.pdf){width=90%}

## Training GANs


The "generator" $G$ is just like the "decoder" that we saw previously.

The "discriminator" $D$ is a standard CNN, where the inputs are images and the
outputs are labels of "authenticity".

## Training GANs

Here are the steps a GAN takes:

1. The generator $G$ takes in a random vector with $\widehat{d}$ dimensions and
   spits out an image.
2. The image is fed into the discriminator $D$, with label $0$, along with a
   stream of real images, each with label $1$.
3. The discriminator generates a probability for each input as to whether it is
   "real" (1) or "fake" (0).

## Training GANs

We have ***two*** networks in this scenario that we are trying to train
simultaneously.

We switch back-and-forth between training the discriminator and training the
generator, each time keeping the other steady.

The loss of the discriminator causes it to ***get better at spotting fake
images***, while the loss of the generated causes it to ***get better at
generating fake images***.

## Training GANs

Since now we're generating two deep networks at once, these take a ***long***
time to train.

There's no real way around this, but the upside is that we can start to see some
franky scary results...

# Examples
## Example Results

You can find implementations of GANs throughout the internet.

I'll show a few different example results, along with links so you can try this
out yourself.

## Neural Face

![Example of Generated Faces](../imgs/23_neural_faces_change6.png){width=35%}

[Neural Face Project](https://carpedm20.github.io/faces/): ***https://carpedm20.github.io/faces/***

[code](https://github.com/carpedm20/DCGAN-tensorflow): ***https://github.com/carpedm20/DCGAN-tensorflow***

## Neural Face Latent Walk

![Just Be Happy!](../imgs/23_neural_faces_change6_closeup.png){width=100%}

The latent walk in the "face-space" shows that it really is learning
higher-level features -- like different facial expressions!

## DC-GAN

![Bedroom Dataset, One Epoch](../imgs/23_bedrooms_one_epoch_samples.png){width=80%}

[DC-GAN](https://github.com/Newmu/dcgan_code): ***https://github.com/Newmu/dcgan_code***

## DC-GAN

![Bedroom Dataset, Five Epochs](../imgs/23_bedrooms_five_epoch_samples.png){width=80%}

[DC-GAN](https://github.com/Newmu/dcgan_code): ***https://github.com/Newmu/dcgan_code***

## Bedroom Latent Walk

![Bedroom Latent Walk](../imgs/23_bedrooms_latent_walk.png){width=40%}

[DC-GAN](https://github.com/Newmu/dcgan_code):
***https://github.com/Newmu/dcgan_code***

## Bedroom Latent Walk

![Bedroom Latent Walk Closeup](../imgs/23_bedrooms_latent_walk_closeup.png){width=100%}

## Latent Space Arithmetic

So this latent space... Is it like any other space? Like, can you add and
subtract components of the latent space?

And if those "vectors" refer to image "features"...

## Latent Space Arithmetic

![](../imgs/23_faces_arithmetic_collage_glasses.png){width=90%}

## Nvidia StyleGAN

Want to never trust anything you see ever again?

- [This Person Does Not Exist](https://thispersondoesnotexist.com/)
- [Nvidia StyleGAN Demonstration](https://www.youtube.com/watch?v=kSLJriaOumA)

## Biology Examples

Okay, so... "Biomedical Data"...

Generative networks are interesting for biology because:

1. They allow us some insight into what a network actually learns
2. They can be used to perturb the image space to generate "fake", but
   realistic, learning examples for classification
3. The features they pick up on could indicate previously unknown morphological
   types
   
Also, they're just... cool.

## Which of These are Real?

\begincols

\column{0.5\linewidth}

![](../imgs/23_nuclei_fake_samples_epoch_001.png){width=80%}

\column{0.5\linewidth}

![](../imgs/23_nuclei_real_samples.png){width=80%}

\stopcols

## Which of These are Real?

\begincols

\column{0.5\linewidth}

![](../imgs/23_nuclei_fake_samples_epoch_001_cropped.png){width=80%}

\column{0.5\linewidth}

![](../imgs/23_nuclei_real_samples_cropped.png){width=80%}

\stopcols

## Which of These are Real?

\begincols

\column{0.5\linewidth}

![FAKE](../imgs/23_nuclei_fake_samples_epoch_001_cropped.png){width=80%} 

\column{0.5\linewidth}

![REAL](../imgs/23_nuclei_real_samples_cropped.png){width=80%}

\stopcols

## Variations on a Theme

\begincols

\column{0.5\linewidth}

![Epoch 001](../imgs/23_nuclei_fake_samples_epoch_001_cropped.png){width=80%} 

\column{0.5\linewidth}

![Epoch 002](../imgs/23_nuclei_fake_samples_epoch_002_cropped.png){width=80%}

\stopcols

## How Many Nuclei?

\begincols

\column{0.5\linewidth}

![](../imgs/23_nuclei_fakecrop01.png){width=80%} 

\column{0.5\linewidth}

![](../imgs/23_nuclei_fakecrop02.png){width=80%}

\stopcols

## How Many Nucleoli?

\begincols

\column{0.5\linewidth}

![](../imgs/23_nucleoli_fakecrop01.png){width=80%} 

\column{0.5\linewidth}

![](../imgs/23_nucleoli_fakecrop02.png){width=80%}

\stopcols
