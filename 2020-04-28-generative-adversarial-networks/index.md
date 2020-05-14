---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade
slide-level: 2

title: Generative Adversarial Networks
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
contact: scottdoy@buffalo.edu
date: 2020-04-28
---

# 
## Recap: (Variational) Autoencoders
## Recap: Image Manifold

![Illustration of Image Manifold (CIFAR-10)](img/cnn_manifold.png){width=75%}

## Recap: Image Manifold Sampling

![Sampling the Manifold](img/cnn_manifold_sample.png){width=75%}

## Recap: Autoencoders

Autoencoders do two things:

<ol>
<li class="fragment">Build an image-space manifold</li>
<li class="fragment">Sample the manifold to generate samples</li>
</ol>

## Recap: Autoencoder Diagram

![Autoencoder Schematic](img/autoencoder_diagram2.svg){width=90%}

## Recap: Undercomplete Autoencoders

If \$\\mathbf{x} \\in \\mathbb{R}\^{d}\$, then making
\$\\mathbf{h}\\in\\mathbb{R}\^{\\widehat{d}}\$, where \$\\widehat{d}<d\$, forces the
encoder to "learn" the most important parts of \$\\mathbf{x}\$.

<p class="fragment">
This is **bottlenecking** or **limited bandwidth**, and results in an
**undercomplete autoencoder**.
</p>

## Recap: Training

**Regularization** takes on many forms:

<ul>
<li class="fragment">**Loss Function**: \$L(\\mathbf{x}, g(f(\\mathbf{x})))\$</li>
<li class="fragment">**Sparse Autoencoder**: \$L(\\mathbf{x}, g(f(\\mathbf{x}))) + \\Omega(\\mathbf{h})\$</li>
<li class="fragment">**Contractive Autoencoder**: \$L(\\mathbf{x}, g(f(\\mathbf{x}))) + \\Omega(\\mathbf{h}, \\mathbf{x})\$</li>
<li class="fragment">**Denoising Autoencoders**: \$L(\\mathbf{x}, g(f(\\widetilde{\\mathbf{x}})))\$</li>
</ul>

## Recap: VAE

**Variational Autoencoders** (VAEs) define the encoder and decoder
probabilistically:

\$ q\_{\\phi}(\\mathbf{h}|\\mathbf{x}) \\qquad \\textrm{Encoder} \$

\$ p\_{\\theta}(\\widetilde{\\mathbf{x}} | \\mathbf{h}) \\qquad \\textrm{Decoder} \$

<p class="fragment">
To avoid degenerate solutions, we force the system to implement
a **conjugate prior** in the form of a spherical unit Gaussian:
</p>

<p class="fragment">\$ \\mathbf{h}\\sim\\mathcal{N}(0,I) \$</p>

## Recap: MNIST Dataset

![MNIST Digits Database](img/mnist_examples.png){width=35%}

## Example: Epoch 10

<div class="l-double">
<div>
![VAE Reconstruction, Epoch 10](img/vae_reconstruction_10.png){width=100%}
</div>
<div>
![VAE Sampling, Epoch 10](img/vae_sample_10.png){width=70%}
</div>
</div>

## Example: Latent Walks

![Latent Walk](img/latent_space_30.png){width=45%}

# 
## Adversarial Networks
## Adversarial Training Overview
**Regularization**: The goal is to ensure that we learn a hidden latent space
without over-training (just copying things over).

<p class="fragment">
There's another way to train these kinds of networks, and that's using
**Adversarial** training.
</p>

## Adversarial Training Intuition

Ever hear of the **Turing test**?

Video: [Brian Christian, "The Most Human Human"](https://youtu.be/lFIW8KphZo0?t=40)

## Judging Generated Outputs

![VAE Reconstruction, Epoch 5](img/vae_reconstruction_5.png){width=80%}

When we see the results of a generative network (e.g. an autoencoder), how do we
judge how "good" it is?

<p class="fragment">In other words: what is the loss function that we are trying to optimize?</p>

## Turing Loss Function

In training generative networks, we compare the input image to the output. 

<p class="fragment">
But in inference or testing, humans look at the output and try to guess whether
an AI "drew" the sample, or if it is a real one.
</p>

<p class="fragment">
If it can fool us, then we would say that the network has successfully been
trained.
</p>

<p class="fragment">
What if we use that feedback to **train** the network in the first place? We
might call this a "Turing Loss" function.
</p>

## Human Judges

Obviously, for image-based autoencoders, we can't have a person sift through
millions of generated samples hundreds of times just to provide a numeric loss
function for each output in terms of whether or not it's a "real" or "fake"
generated sample.

## Any Volunteers?

![Who wants to judge each of these samples?](img/latent_space_30.png){width=45%}

## Human Judges

That would be tedious, expensive, time-consuming, and error-prone.

<p class="fragment">
If only there was a robot who could sit around and judge generated samples all
day...
</p>

## It's AI All the Way Down

![Generative Adversarial Network](img/gan.svg){width=90%}

## Training GANs

The "generator" \$G\$ is just like the "decoder" that we saw previously.

<p class="fragment">
The "discriminator" \$D\$ is a standard CNN, where the inputs are images and the
outputs are labels of "authenticity".
</p>

## Training GANs

Here are the steps a GAN takes:

<ol>
<li class="fragment">The generator \$G\$ takes in a random vector with \$\\widehat{d}\$ dimensions and spits out an image.</li>
<li class="fragment">The image is fed into the discriminator \$D\$, with label \$0\$, along with a stream of real images, each with label \$1\$.</li>
<li class="fragment">The discriminator generates a probability for each input as to whether it is "real" (1) or "fake" (0).</li>
</ol>

## Training GANs

We have **two** networks in this scenario that we are trying to train
simultaneously.

<p class="fragment">
We switch back-and-forth between training the discriminator and training the
generator, each time keeping the other steady.
</p>

<p class="fragment">
The loss of the discriminator causes it to **get better at spotting fake
images**, while the loss of the generated causes it to **get better at
generating fake images**.
</p>

## Training GANs

Since now we're generating two deep networks at once, these take a **long**
time to train.

<p class="fragment">
There's no real way around this, but the upside is that we can start to see some
franky scary results...
</p>

# 
## Examples

## Example Results

You can find implementations of GANs throughout the internet.

<p class="fragment">
I'll show a few different example results, along with links so you can try this
out yourself.
</p>

## Neural Face

![Example of Generated Faces](img/neural_faces_change6.png){width=35%}

[Neural Face Project](https://carpedm20.github.io/faces/): **https://carpedm20.github.io/faces/**

[code](https://github.com/carpedm20/DCGAN-tensorflow): **https://github.com/carpedm20/DCGAN-tensorflow**

## Neural Face Latent Walk

![Just Be Happy!](img/neural_faces_change6_closeup.png){width=100%}

The latent walk in the "face-space" shows that it really is learning
higher-level features -- like different facial expressions!

## DC-GAN

![Bedroom Dataset, One Epoch](img/bedrooms_one_epoch_samples.png){width=80%}

[DC-GAN](https://github.com/Newmu/dcgan_code): **https://github.com/Newmu/dcgan_code**

## DC-GAN

![Bedroom Dataset, Five Epochs](img/bedrooms_five_epoch_samples.png){width=80%}

[DC-GAN](https://github.com/Newmu/dcgan_code): **https://github.com/Newmu/dcgan_code**

## Bedroom Latent Walk

![Bedroom Latent Walk](img/bedrooms_latent_walk.png){width=40%}

[DC-GAN](https://github.com/Newmu/dcgan_code):
**https://github.com/Newmu/dcgan_code**

## Bedroom Latent Walk

![Bedroom Latent Walk Closeup](img/bedrooms_latent_walk_closeup.png){width=100%}

## Latent Space Arithmetic

So this latent space... Is it like any other space? Like, can you add and
subtract components of the latent space?

<p class="fragment">And if those "vectors" refer to image "features"...</p>

## Latent Space Arithmetic

![](img/faces_arithmetic_collage_glasses.png){width=90%}

## Nvidia StyleGAN

Want to never trust anything you see ever again?

- [This Person Does Not Exist](https://thispersondoesnotexist.com/)
- [Nvidia StyleGAN Demonstration](https://www.youtube.com/watch?v=kSLJriaOumA)

## Biology Examples

Okay, so... "Biomedical Data"...

<p class="fragment">Generative networks are interesting for biology because:</p>

<ol>
<li class="fragment">They allow us some insight into what a network actually learns</li>
<li class="fragment">They can be used to perturb the image space to generate "fake", but realistic, learning examples for classification</li>
<li class="fragment">The features they pick up on could indicate previously unknown morphological types</li>
</ol>

<p class="fragment">Also, they're just... cool.</p>

## Which of These are Real?

<div class="l-double">
<div>
![](img/nuclei_fake_samples_epoch_001.png){width=80%}
</div>
<div>
![](img/nuclei_real_samples.png){width=80%}
</div>
</div>

## Which of These are Real?

<div class="l-double">
<div>
![](img/nuclei_fake_samples_epoch_001_cropped.png){width=80%}
</div>
<div>
![](img/nuclei_real_samples_cropped.png){width=80%}
</div>
</div>

## Which of These are Real?

<div class="l-double">
<div>
![FAKE](img/nuclei_fake_samples_epoch_001_cropped.png){width=80%} 
</div>
<div>
![REAL](img/nuclei_real_samples_cropped.png){width=80%}
</div>
</div>

## Variations on a Theme

<div class="l-double">
<div>
![Epoch 001](img/nuclei_fake_samples_epoch_001_cropped.png){width=80%} 
</div>
<div>
![Epoch 002](img/nuclei_fake_samples_epoch_002_cropped.png){width=80%}
</div>
</div>

## How Many Nuclei?

<div class="l-double">
<div>
![](img/nuclei_fakecrop01.png){width=80%} 
</div>
<div>
![](img/nuclei_fakecrop02.png){width=80%}
</div>
</div>

## How Many Nucleoli?

<div class="l-double">
<div>
![](img/nucleoli_fakecrop01.png){width=80%} 
</div>
<div>
![](img/nucleoli_fakecrop02.png){width=80%}
</div>
</div>
