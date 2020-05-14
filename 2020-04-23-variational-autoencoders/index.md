---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade
slide-level: 2

title: Autoencoders
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
contact: scottdoy@buffalo.edu
date: 2020-04-23
---

# 
## Recap: CNNs

## Diagram of CNN In Action: Input

<div class="l-double">
<div>
![Input Image](img/cnn_layer0_input.png){width=80%}
</div>
<div>
![Input Layer](img/cnn_layer0_diagram.png){width=80%}
</div>
</div>
 
## Diagram of CNN In Action: CONV 1

<div class="l-double">
<div>
![CONV 1 Activations](img/cnn_layer1_activation.png){width=80%}
</div>
<div>
![CONV 1 Diagram](img/cnn_layer1_diagram.png){width=80%}
</div>
</div>
 
 
## Diagram of CNN In Action: POOL 1 

<div class="l-double">
<div>
![POOL 1 Activations](img/cnn_layer2_activation.png){width=80%}
</div>
<div>
![POOL 1 Diagram](img/cnn_layer2_diagram.png){width=80%}
</div>
</div>
 
 
## Diagram of CNN In Action: CONV 2 

<div class="l-double">
<div>
![CONV 2 Activations](img/cnn_layer3_activation.png){width=80%}
</div>
<div>
![CONV 2 Diagram](img/cnn_layer3_diagram.png){width=80%}
</div>
</div>
 
 
## Diagram of CNN In Action: POOL 2 

<div class="l-double">
<div>
![POOL 2 Activations](img/cnn_layer4_activation.png){width=80%}
</div>
<div>
![POOL 2 Diagram](img/cnn_layer4_diagram.png){width=80%}
</div>
</div>
 
 
## Diagram of CNN In Action: FC 1 

<div class="l-double">
<div>
![FC 1 Activations](img/cnn_layer5_activation.png){width=80%}
</div>
<div>
![FC 1 Diagram](img/cnn_layer5_diagram.png){width=70%}
</div>
</div>
 
 
## Diagram of CNN In Action: SOFTMAX

<div class="l-double">
<div>
![SOFTMAX Outputs](img/cnn_layer6_softmax.png){width=80%}
</div>
<div>
![SOFTMAX Diagram](img/cnn_layer6_diagram.png){width=80%}
</div>
</div>
 
 
## Diagram of CNN In Action: Output Mapping

<div class="l-double">
<div>
![Tissue Map](img/cnn_layer7_map.png){width=80%}
</div>
<div>
![Output](img/cnn_layer7_diagram.png){width=80%}
</div>
</div>

# 
## Generative Models

## CNNs: Behind the Layers

So far, we've looked at CNNs for **classification**.

<p class="fragment">
Central idea: they learn how to generalize the data they see (training) to
apply labels to a broad, unseen testing set.
</p>

<p class="fragment">
They do this by understanding **higher-order** relationships between the input
data (pixel values).
</p>

<p class="fragment">
Meaningful relationships between inputs and groups of inputs are learned,
essentially, through trial-and-error (i.e. backpropagation): Using errors in
classification to modify the "assumptions" (weights and biases) in the hidden
layers.
</p>

## Image Space and Manifolds

Think about the life of a CNN: the only thing it knows about the universe is
what is contained in the training data.

<p class="fragment">
The CNN is trying to learn a **part of the image space** related to our
problem, which is defined by the training data.
</p>

## Image Space and Manifolds

The input **image space** is finite, meaning there is a limit to how many images you
can create given a fixed grid size. Within that grid, every image -- from birds
to faces to nuclei to genomic arrays -- can be constructed. 

<p class="fragment">
You can think of this image space as a **manifold**, where similar images
appear close to one another and dissimilar images appear far apart (sound familiar?).
</p>

## Image Manifold Example

![Illustration of Image Manifold (CIFAR-10)](img/cnn_manifold.png){width=75%}

## Understanding what CNNs "Know"

The question we have to ask is: Has the CNN **actually** learned these
relationships? Does it "know" what a tumor patch looks like, or is its idea of a
tumor patch fundamentally different from ours?

<p class="fragment">
It would be great if we could ask the system to **draw** for us what it thinks
a particular class is like. 
</p>

## Image Manifold Sampling

![Sampling the Manifold](img/cnn_manifold_sample.png){width=75%}

## Interpreting the Output

<div class="l-double">
<div>
![](img/unknown_reconstruction.png)
</div>
<div>
Has the classifier learned an "image manifold"?
<p class="fragment">What relationships between data exist in this manifold space?</p>
<p class="fragment">Can we use this space to create realistic (but "imagined") additional samples?</p>
</div>
</div>

## Enter: Variational Autoencoders

How do we build these systems? We need a setup that can do two things:

<ol>
<li class="fragment">Build an underlying manifold of image space given a set of training data;</li>
<li class="fragment">Sample from the underlying manifold to reconstruct sample inputs.</li>
</ol>

<p class="fragment">
In this lecture we'll talk about **Variational Autoencoders** as the way to do
this. 
</p>

# 
## Autoencoders

## Second Things First: Autoencoders

Let's start by understanding an "autoencoder". 

<p class="fragment">
Simply put, it's a neural network designed to take an input sample and
reconstruct it. So if you have an input \$\\mathbf{x}\$, the autoencoder will spit
out \$\\hat{\\mathbf{x}}\$, which is it's attempt to recreate \$\\mathbf{x}\$. 
</p>

<p class="fragment">Sounds simple, right?</p>

## Autoencoder Generalizability

An autoencoder consists of: 

<ul>
<li class="fragment">An **encoder**, \$\\mathbf{h} = f(\\mathbf{x})\$: transforms input \$\\mathbf{x}\$ into a **code** (hidden layer); and</li>
<li class="fragment">A **decoder**, \$\\mathbf{r}=g(\\mathbf{h})\$, which produces a reconstruction.</li> 
</ul>

<p class="fragment">
If we want to copy the inputs exactly, we could just say that the hidden layer
has the same dimensionality as the input, and the decoder has the same
dimensionality as the hidden layer, and then set all the weights to 1.
</p>

<p class="fragment">What would this look like?</p>

## Meaningless Autoencoder

![Expensive Copy Machine](img/autoencoder_xerox2.svg){width=100%}

## Designed Imperfection

Obviously that isn't what we want. Instead, we restrict -- or **regularize**
-- the autoencoder so that we can't perfectly replicate the input data.

<p class="fragment">
By doing this, we force the system to learn only the **most important parts**
of the input, so that it can achieve "pretty good" reconstruction using as
little information as possible.
</p>

## Bottlenecking

Example: You are moving from your parents' house into a new apartment. You like
your current bedroom, so you try to build a perfect replica in your new place. 

<p class="fragment">
If you could bring everything you own -- if the "bandwidth" of the moving van
allowed you to bring all your possessions -- then you could build a 1:1 replica
of your room in your new place.
</p>

<p class="fragment">
But if we impose a **bottleneck** -- for example, you can't afford a moving
van and have to cram everything into a tiny car -- then you have to pick and
choose what to bring with you to create the closest possible replica. 
</p>

<p class="fragment">
It won't be perfect, but **what** you choose to bring says a lot about what
you think is important in the "concept" of what your bedroom is like.
</p>

## Undercomplete Autoencoders

One easy way to bottleneck is to specify that our encoder function
\$\\mathbf{h}=f(\\mathbf{x})\$ must have a lower dimension than the input
\$\\mathbf{x}\$. Thus, we are by definition throwing away some data when going from
\$\\mathbf{x}\\rightarrow\\mathbf{h}\$. 

## Autoencoder: Diagram

![Autoencoder Schematic](img/autoencoder_diagram2.svg){width=100%}

## Undercomplete Loss Function

This is called an **undercomplete** autoencoder, and as with all neural
networks, it's trained by defining a loss function: 

\$ L(\\mathbf{x}, g(f(\\mathbf{x}))) \$

... which penalizes the system if the reconstruction output is dissimilar from
the input.

## Generalization of PCA

If \$L\$ is the mean squared error and the decoder function is linear, then the
subspace learned by the autoencoder is similar to PCA. 

<p class="fragment">
If the decoder function is nonlinear, then we can build a more generalizable and
powerful version of PCA -- this is good!
</p>

<p class="fragment">
But if the encoder and decoder have too much **capacity**, then they will not
extract any useful data from the training. Instead, they will just "memorize"
the training set, mapping each input to a single value in the hidden layer --
this is bad!
</p>

<p class="fragment">So... how do we know how much capacity to give our network?</p>

# 
## Regularization

## Motivations for Regularization

By designing our system cleverly, we can address the issue of "capacity" by
making sure that our system has the right goals (defined by the loss function)
to learn important features of the input data **without** memorizing them. 

<p class="fragment">
There are a bunch of ways to do this, and a lot of them involve changing the
loss function so that the network needs to learn to do something **in
addition** to copying the inputs.
</p>

## Sparse Autoencoders

**Sparse Autoencoders** add a sparsity term to the loss, so that the loss
function becomes: 

\$ L(\\mathbf{x}, g(f(\\mathbf{x}))) + \\Omega(\\mathbf{h})\$

<p class="fragment">
The sparsity term \$\\Omega(\\mathbf{h})\$ is designed such that it forces the
system to **learn classification** in addition to the reconstruction.
</p>

## Denoising Autoencoders

**Denoising Autoencoders** work by modifying the loss function to be:

\$ L(\\mathbf{x}, g(f(\\widetilde{\\mathbf{x}}))) \$

<p class="fragment">
where \$\\widetilde{\\mathbf{x}}\$ is a copy of \$\\mathbf{x}\$ with some random noise added to
it. So now, the second task is to **remove the noise component from the
sample** as well as reconstruct it from the code layer.
</p>

## Denoising Autoencoder Example

![From the "Deep Learning Book"](img/denoising_autoencoder.png){width=80%}

## Penalizing Derivatives

You can also create a version of a sparse encoder where the error term
takes both \$\\mathbf{x}\$ and \$\\mathbf{h}\$ as parameters:

\$ L(\\mathbf{x}, g(f(\\mathbf{x}))) + \\Omega(\\mathbf{h}, \\mathbf{x}) \$

<p class="fragment">In this version, \$\\Omega(\\mathbf{h}, \\mathbf{x})\$ is of the form:</p>

<p class="fragment">\$ \\Omega(\\mathbf{h}, \\mathbf{x}) = \\gamma \\sum\_{i} |\\nabla\_{\\mathbf{x}} h\_{i}|\^{2} \$</p>

<p class="fragment">
What this means is that if the gradient of \$\\mathbf{x}\$ is small, then the
sparsity term is small and the classifier is close to calculating the
reconstruction loss exclusively. If the gradient is large, then the
loss will have to include a giant term, leading to more regularization.
</p>

<p class="fragment">This is called a **contractive autoencoder**.</p>

## Autoencoders Galore

As you can imagine, there are a bunch of different ways you can try to
manipulate autoencoders to give you what you want: A meaningful representation
of the input data, encoded as \$\\mathbf{h}\$, that represents a low-dimensional
set of features related to the "image-space".

<p class="fragment">However, you can do more than just add terms to the loss function...</p>

# 
## Variational Autoencoders

## VAE: Bayesian Approach

So far, the autoencoders are **deterministic** -- meaning that a particular
input is always mapped to the same code, and the same code is always mapped to a
particular output.

<p class="fragment">
A VAE takes a slightly different approach by defining the encoder and decoder
probabilistically: 
</p>

<p class="fragment">\$ q\_{\\phi}(\\mathbf{h}|\\mathbf{x}) \\qquad \\textrm{Encoder} \$</p>
<p class="fragment">\$ p\_{\\theta}(\\widetilde{\\mathbf{x}} | \\mathbf{h}) \\qquad \\textrm{Decoder} \$</p>

<p class="fragment">
This means that, given an input \$\\mathbf{x}\$, there is a **probability** that
we will observe \$\\mathbf{h}\$, and given a particular \$\\mathbf{h}\$, there is
another probability that we will observe \$\\widetilde{\\mathbf{x}}\$. These
probabilities are parameterized by \$\\phi\$ and \$\\theta\$, respectively.
</p>

## Conjugate Priors

We still need to "regularize", since the probabilities could learn to be
arbitrarily small (i.e. the distributions governing the mapping could just
become delta functions, with zero variance). So we force the system to implement
a **conjugate prior** in the form of a spherical unit Gaussian:

\$ \\mathbf{h}\\sim\\mathcal{N}(0,I) \$

Why is this useful?

## Bayesian Inference

In a Bayesian world, the things that we observe are random variables. This means
that there is some underlying distribution or "law" that says that certain
observations -- like a particular nuclear radius -- occur with a specific
probability. 

<p class="fragment">
The goal in Bayes is to try and estimate that probability through observation or
training. 
</p>

## Image Features, Abstracted

By applying this process to autoencoders, we're saying that each image is a
**sample of a distribution** of possible images, and the dimensions of the
latent space encode **salient features of the image**. 

<p class="fragment">
This means that in our low-dimensional code \$\\mathbf{h}\$, one dimension might
refer to the "tilt" of the numbers, one might refer to the sharpness of its
corners, etc.
</p>

## Sampling the Code

So now, we can do two things:

<ul>
<li class="fragment">First, we can view a low-dimensional representation of the input image data</li>
<li class="fragment">Second, we can **sample** this low-dimensional space and have the system generate samples that are maximally-likely to have come from that region</li>
</ul>

<p class="fragment">I think some examples are in order...</p>

## Example: MNIST

![MNIST Digits Database](img/mnist_examples.png){width=35%}

## Example: VAE model

![PyTorch Example Code](img/torch_model.png){width=90%}

## Example: Training

![Reconstruction Loss vs. Epochs](img/vae_validation_loss.svg){width=75%}

## Example: Epoch 1

<div class="l-double">
<div>
![VAE Reconstruction, Epoch 1](img/vae_reconstruction_1.png){width=100%}
</div>
<div>
![VAE Sampling, Epoch 1](img/vae_sample_1.png){width=70%}
</div>
</div>

## Example: Epoch 5

<div class="l-double">
<div>
![VAE Reconstruction, Epoch 5](img/vae_reconstruction_5.png){width=100%}
</div>
<div>
![VAE Sampling, Epoch 5](img/vae_sample_5.png){width=70%}
</div>
</div>

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

Finally, we can do something pretty cool: let's pick a set of continuous points
in the latent space, and generate samples from an evenly-spaced grid going from
one section to another and generating a continuous set of samples from the
space.

<p class="fragment">
This helps us to visualize how the network "thinks" the universe appears, based
on the training data!
</p>

## Example: Latent Walks

![Latent Walk](img/latent_space_30.png){width=45%}

## Example: Latent Walks

![Latent Walk](img/latent_space_500.png){width=45%}

## Example: Latent Walks

![Latent Walk](img/latent_space_1000.png){width=45%}

# 
## Parting Words

## Parting Words

These methods are very interesting, but we can make them even better by pitting
them against each other!

<p class="fragment">
In the next lecture we'll talk about **adversarial networks**, where two
classifiers "fight it out" so that each gets better at creating new samples.
</p>
