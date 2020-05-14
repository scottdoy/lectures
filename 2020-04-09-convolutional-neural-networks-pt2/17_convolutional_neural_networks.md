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
title: CONVOLUTIONAL NEURAL NETWORKS (Pt. 2)
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
institute: scottdoy@buffalo.edu
date: April 16, 2019
---

# Recap Last Lecture

## Recap: Rationale for Deep Networks

In theory, neural networks can replicate any function (decision surface), no
matter how complicated. "In theory".

In reality, this would require an unreasonable number of:

- ***Input nodes***, to describe increasingly large data types;
- ***Input samples***, to accurately describe a complex, varied class space;
- ***Hidden nodes***, to capture the nonlinear relationship between the inputs
  and desired outputs;
- ***Hours***, to fully train a network on all this data;
- ***Dollars***, to buy the hardware required for such a massive undertaking.

Surely, there must be a better way?

## Recap: Deep Learning

Deep learning, the stacking of several hidden layers together, is enabled by a
few insights:

- You can design the network such that it is ***differentiable end-to-end***; in
  other words, you can train each layer with gradient descent via
  backpropagation, and learn an optimal set of weights.
- You don't need to have ***fully connected layers*** -- you can replace those
  layers with other, sparser layers that look at a limited input space, or
  certain "aspects" of the data, and learn weights that tie those regions
  together.
- Nowadays, you have ***ridiculously large datasets*** from which you can pull
  training data. This will expose your network to sufficient amount of
  variability across a very complex class space while avoiding overfitting.

## Recap: Schematic of Fully-Connected Networks

![Fully-Connected Neural Network](../imgs/17_neural_net_schematic.jpeg){width=90%}

## Recap: Scalability of Traditional Neural Nets

As we increase pixel sizes, this number of inputs balloons very quickly: a
200x200 color image would result in 120,000 weights, and so on.

Moreover, each of these must be mapped to hidden units -- so our weight vector
is 120,000 multiplied by some (likely larger) number of hidden units.

And that's all for one layer, and for a dataset of just 200x200x3 images, which
are quite small.

## Recap: CNN Rationale

***Convolutional Neural Networks*** are specifically-designed to operate on
image data. This enables a few assumptions and tricks:

- Image inputs are ***geographically important***, meaning that image features
  from one part of the image don't have to be connected to those in another part
  of the image.
- In the case of color images (3D), hidden nodes can be arranged as a ***3D
  volume*** instead of a single "layer", allowing the same part of the image to
  be analyzed in different ways (i.e. using different filter parameters).
- As the data is processed, we can reduce the number of nodes at each layer by
  downsampling the image (pooling). We can still recognize the image content,
  even as it is reduced to a very small size.

## Recap: Schematic of Convolutional Nets

![Convolutional Neural Network](../imgs/17_cnn_schematic.jpeg){width=100%}

## Recap: CNN Layers

There are several types of layers in CNNs which process the input data:

- ***Convolutional Layers***: A kernel is convolved with the input. The height
  and width of the output is a function of the filter parameters (padding,
  kernel size, and stride), while the depth of the output is a hyperparameter of
  the layer.
- ***Rectified Linear Units***: A type of activation function, ReLU layers take
  the form of $\max(0,x)$. This prevents gradients from getting unmanagable
  (exploding or vanishing).
- ***Pooling Layers***: Downsamples the input volume in the height and width
  dimensions. This reduces the number of neurons / parameters that must be
  learned, without reducing the expressive power of the network.
- ***Fully-Connected Layers***: Same type of layer as in traditional nets, where
  each neuron is fully connected to each element of the input.

## Recap: Example of Convolutional Nets Operating on an Image

![](../imgs/17_convnet_overview.jpeg){width=100%}

# Layer Details

## CONV: Convolutional Layer

***Convolutional Layers*** consist of a set of learnable image filters, which
are convolved with the image to produce a map of responses at each local
neighborhood.

The size of the result of this layer depends on a number of factors:

- ***Receptive Field*** $\mathbf{F}$: equivalent to the filter's ***kernel size***
- ***Depth*** $\mathbf{K}$: the number of filters that "looks" at each region of the input
- ***Stride*** $\mathbf{S}$: how the filter moves across the image
- ***Padding*** $\mathbf{P}$: how we handle the edges of the spatial input volume

The ***Receptive Field***, ***Stride***, and ***Padding*** affect the "spatial"
output dimensions (height and width), while the ***Depth*** affects the, well,
"depth" output dimension.

## Example of a Convolutional Layer

![Illustration of the spatial connectivity of a neuron in the convolutional layer. These neurons are stacked 5-deep because there are 5 neurons "looking at" different aspects of the same input space.](../imgs/17_depthcol.jpeg){width=50%}

## CONV: Receptive Field Parameter

The spatial extent of a neuron's connectivity is called the ***receptive
field*** of the neuron, which is treated as a hyperparameter. This is equivalent
to the convolutional filter's kernel size.

Each neuron handles a limited spatial location but is connected through the full
depth of the input volume; i.e. the neurons corresponding to the color channels
of the same receptive field are all connected with one another.

So if we have an input size of [32x32x3], and we set our kernel size (receptive
field) of the [CONV] layer to 5, then each neuron in the [CONV] layer will have
weights connecting to a 5x5x3 region of the input volume, so each neuron would
have 75 weights and one bias term.

## CONV: Depth, Stride, and Padding Parameters

The following three parameters controls how the filter operates across the input
image:

- ***Depth*** controls how many neurons look at the same spatial region but
  different "aspects", like edges or colors. Increasing this has no effect on
  the height and width of the output volume, just the depth.
- ***Stride*** controls the overlap of neighboring receptive fields. A stride of
  1 means the filter moves one pixel at a time, 2 looks at a pixel and then
  leapfrogs the next, etc. More overlap means more neurons and potentially more
  redundancy, but also a more descriptive input space.
- ***Zero-padding*** refers to how we treat the borders of the input (same as
  with regular signal / image processing). We can use this to determine what
  happens at the border, to make input volumes "fit" the network, and to
  precisely control the size of the output of the CONV layer.

## CONV: Spatial Size of the Output

The size of the output (height and width) is a function of the input size in the
spatial dimension ($\mathbf{W}$), the receptive field size ($\mathbf{F}$), the
stride ($\mathbf{S}$), and the zero-padding ($\mathbf{P}$):

$$ \frac{\mathbf{W}-\mathbf{F}+2\mathbf{P}}{\mathbf{S}}+1 $$

So if we designed a CONV layer that operates on our [32x32x3] input
($\mathbf{W}=32$), and we used a filter size of 3 ($\mathbf{F}=3$), a padding of
2 ($\mathbf{P}=2$), and a stride of 3 ($\mathbf{S}=3$), then we would have an
output of size:

$$ \frac{32 - 3 + (2 \times 2)}{3} + 1 = 12 $$

Thus the output volume would have a spatial size of 12x12. Note that the actual
volume of the output would be 12x12x$\mathbf{K}$, which includes the depth (each
of the filter outputs is 12x12, so stacking up $\mathbf{K}$ filters gives you
the depth dimension.

## CONV: Crafting the Output

It's important to note that these parameters are ***not*** chosen at random;
they have to be selected such that they evenly and smoothly cover the input
space.

The use of zero-padding and some other tricks can help you "fit" the layers
together.

## CONV: Illustration of Varying Output Sizes

![Grey boxes are the input (1-dimensional) and the neurons have input sizes of $\mathbf{F}=3$. The input has been zero-padded once $\mathbf{P}=1$. ***Left:*** Stride of $\mathbf{S}=1$, yielding an output of size $(5-3+2)/1+1=5$. ***Right:*** Stride of $\mathbf{S}=2$, yielding output of $(5-3+2)/2+1=3$. We cannot use a stride of $\mathbf{S}=3$ because it wouldn't fit.](../imgs/17_stride.jpeg){width=100%}

## Parameter Sharing

Research team Krizhevsky, et al. used image sizes of $227\times 227\times 3$,
with $\mathbf{F}=11$, $\mathbf{S}=4$, $\mathbf{P}=0$, and depth $\mathbf{K}=96$,
yielding an output volume of $55\times 55\times96 = 290,400$ neurons!

With a field size of $11\times11\times3$, this yields $363$ weights (plus one
bias term) ***per neuron***, or $290,400 \times (363 + 1)=105,705,600$ parameters...
on just the first layer. ***Each and every one*** of these parameters has to be
estimated through gradient descent and backpropagation!

However, we can reduce this by making a simplifying assumption: Weights are not
likely to change dramatically from one region to the next!

Thus, even though the first layer has $55\times55\times96$ neurons, if we say
that all of the spatial $55\times55$ neurons can share the same parameter, then
we just have $96$ weights, multiplied over the region size:
$96\times11\times11\times3=34,848$.

Much better!

## Simplifying the Parameter Space

If all neurons in a depth slice use the same weight, then the neurons' weights
are ***convolved*** with the corresponding input volume.

The result is an activation map of the region (e.g. $55\times55$), and the
output volume is just this multiplied by the depth slices (e.g. $96$).

## Visualizing Learned Kernels

![96 filters learned by Krizhevsky. Each filter is $11\times11\times3$, corresponding to the size of the spatial location it considers (plus color).](../imgs/17_weights.jpeg){width=100%}

## Pooling Layer

The ***Pooling layer*** acts to reduce the spatial size of the representation,
which in turn reduces the parameters, computation, and overfitting.

Its primary operation is to take the maximum of a $2\times2$ filter with a
stride of $2$, which gets rid of 75\% of the least-important activations.

Since this is a single operation, it requires no new parameters! Only
hyperparameters $\mathbf{F}$ (spatial extent) and $\mathbf{S}$ (stride).

You can generalize the size of the filter and the stride, but typically
$\mathbf{F}=3,\mathbf{S}=2$ or $\mathbf{F}=2,\mathbf{S}=2$.

Note that there's no "real" reason the maximum operation is used; you could
average together (like you do with image resizing), perform a decimation, etc.,
but the max pooling approach seems to work better.

## Example of Maximum Pooling

![The $2\times2$ filter reduces the height and width of the input volume while preserving the depth.](../imgs/17_pool.jpeg){width=50%}

## Example of Maximum Pooling

![At each $2\times2$ block, you simply take the largest value and that becomes the representative at the end of the pooling operation.](../imgs/17_maxpool.jpeg){width=70%}

## Fully-Connected Layers

The ***fully-connected layers*** have are connected to all activations in the
previous layer and are essentially the same as regular neural network layers.

Since there's no modulation of the inputs, these can also be optimized / trained
the same way.

Since convolutional layers are simply modulated by their inputs (they function
the same way), a convolutional layer is a subset of the connections in a
fully-connected layer.

For example, setting a convolutional layer with stride equal to 1 and the depth
equal to the number of inputs is essentially the same as a fully-connected
layer.

# Practical Concerns and Design Hints

## Tips and Tricks

What follows are some tips from \texttt{cs231n}.

These are general rules of thumb to get you started using CNNs; they reflect the
author's own suggestions and are generally good advice.

## Tip: Fewer, Smaller CONV Layers

If you have three 3x3 [CONV] layers on each other, the third layer will have a
7x7 "view" of the input volume. Compare that with a single [CONV] layer with a
7x7 receptive field. ***The three 3x3 [CONV] layers are better.***

- The three [CONV] layers will have non-linearities sandwiched in-between them,
  which the 7x7 [CONV] layer won't have. This gives them more expressive power.
- If all volumes have $C$ channels, then the 7x7 [CONV] layer contains $(C
  \times (7 \times 7 \times C) = 49C^{2}$ parameters, but the three 3x3 layers
  will have $3 \times (C \times (3 \times 3 \times C)) = 27C^{2}$ parameters.

## Tip: When in Doubt, Steal!

Every month, people are testing their architectures against some benchmark
datasets like ImageNet.

***Unless you are interested specifically in deep learning architecture
design***, most of these small, incremental tweaks should not be of central
interest to you.

- If you're using SVMs to determine outcome in patient data, you wouldn't try to
  write a paper about a new type of SVM kernel function.
- If you are writing about a new type of genomic pathway, you don't also need to
  discuss a small change to your PCR protocol.
- Similarly, if you're trying to identify a tumor region in an image of tissue,
  you don't need to also invent a new type of nonlinearity or convolutional
  layer.

## Tip: When in Doubt, Steal!

So if that's the case, what architecture should you choose?

- Use whatever is available for your software stack;
- Use whatever has worked for other researchers looking at the same or similar
  problems;
- Use whatever is currently working the best on benchmark datasets.

While your application may not be able to use the same weights, at least the
architecture will be taken care of.

## Hyperparameter Adjustment

Okay, so we can pick a certain order of layer stacking. We have a pile of images
sitting on the hard drive, and we have labels associated with each image.

So now we need to format our data and set up our layer ***hyperparameters***:
that is, the sizes and operations at each layer, which are NOT learned by
gradient descent.

Here are some guidelines for adjustments.

## Input Layer

Your input layer -- i.e. the image dimensions -- should be divisible by 2.

You'll typically see 32, 64, 96, 224, 384, and 512.

***Why these sizes?*** That's a great question!

- Even numbers help to make the convolutional math work out.
- Smaller images require less memory, so older / simpler nets typically use
  smaller sizes.
- Larger images are used to hold more complex scenes / classes, within reason.
- More inputs require more nodes for processing, which increases dataset size
  requirements and training time limits.

## Convolutional Layers

Layers should use small filter sizes ($\mathbf{F}=3$, $\mathbf{F}=5$), with a
stride of $\mathbf{S}=1$, and padding should be used so that the convolutional
layer does not alter the spatial size of the output. Recall that for any $\mathbf{F}$:

$$ \mathbf{P} = \frac{(\mathbf{F} - 1)}{2}$$

will preserve input size.

Larger filters, e.g. $\mathbf{F}=7$, are only seen on the first convolutional
layer, if at all.


## Pooling Layers

By far the most common pooling layer performs ***max-pooling*** with receptive
field $\mathbf{F}=2$ and stride $\mathbf{S}=2$. Increasing these numbers means
that you are aggressively downsampling your data, which leads to too much
information loss.

## Sizing Concerns

This approach means that CONV layers always preserve input sizes, meaning that
they are just responsible for learning image features (and not image
downsizing).

The POOL layers are the only ones concerned with downsampling the image.

If you don't do this -- if you use $\mathbf{S}>1$ or $\mathbf{P}=0$ -- you have
to keep track of the volume as it changes size throughout the network. If you
bork this, your system won't be symmetric and will likely give you an error.

## Questions and Answers

***Stride of 1 in CONV***: Smaller strides tend to work better in practice, and
prevent CONV layers from downsampling.

***Padding***: Padding keeps the volumes from changing as a result of the
convolution operation, but also allows you to "retain" information at the
boundary instead of seeing it reduced at each convolutional pass.

***Memory***: GPUs are great for performing backpropagation calculations, but
have relatively little onboard memory. Filtering a $224\times224\times3$ image
with a typical architecture can lead to several millions of activations per
image. If you have to compromise, _do it at the beginning_ in the first CONV
layer by using larger filter sizes and strides.

## Calculating Memory Requirements

Memory in CNNs is taken up by three sources:

- ***Activations***: Intermediate volume sizes have a number of activations plus
  an equal number of gradients to keep track of.
- ***Parameters***: These numbers hold the network parameters, the gradients
  during backprop, and also a step size / learning rate.
- ***Miscellaneous***: Image batch sizes, augmented images / parameters, etc.

You can get a rough estimate of these values, multiply by 4 to get the number of
bytes needed, then divide by 1024 to get KB, MB, and GB. Then compare that to
your GPU's memory size (typically 3, 4, 6, or maybe 12 if you've got good
hardware).

To reduce memory, the first thing you should look at is reducing the batch size,
which in turn reduces your activations.

# Design of Convolutional Network Architectures

## Standard Layer Configurations

The order and number of layers in a CNN (CONV, POOL, RELU, FC) defines the
***architecture*** of the network. Some famous architectures have been given
names by the folks who designed them and proved their effectiveness over
competing architectures.

In general, you'll see architectures arranged as:

\centering\texttt{[INPUT] -> [[CONV -> RELU]*N -> POOL?]*M -> [FC -> RELU]*K ->
FC}

\raggedright
How does the architecture affect the processing of the network?

## Affect of Architecture on Processing

- \texttt{[INPUT] -> [FC]}

    Regular linear classifier (linear discriminant)

- \texttt{[INPUT] -> [CONV] -> [RELU] -> [FC]}

    Most basic kind of convolutional network

- \texttt{[INPUT] -> [[CONV] -> [RELU] -> [POOL]]*2 -> [FC] -> [RELU] -> [FC]}

    Here there is a convolutional layer paired with each pooling layer, followed
    by a fully-connected "head" section.

- \texttt{[INPUT] -> [[[CONV] -> [RELU]]*2 -> POOL]*3 -> [[FC] -> [RELU]]*2 ->
  [FC]}

    By allowing multiple convolutions to take place before the destructive
    pooling operation, we can ensure that deeper, more informative features get
    learned.

# Specific CNN Architectures

## LeNet

Developed in the 1990's, LeNet was used for handwriting digit analysis on the
MNIST dataset.

## LeNet: Diagram

![LeNet Schematic](../imgs/18_lenet.pdf){width=100%}

## AlexNet

Won the ImageNet ILSVRC 2012 challenge; larger and deeper than LeNet, with
multiple stacked CONV layers.

## AlexNet: Diagram

![AlexNet Schematic (Cut off in original paper)](../imgs/18_alexnet.pdf){width=100%}

## GoogLeNet

> Once Google gets in the game, everyone else is out of a job.

GoogLeNet won the ILSVRC 2014 challenge. The team developed an "inception"
module, which reduced the number of parameters in the network from 60M (in
AlexNet) to 4M.

They also replaced the FC layers with "average pooling" layers.

## GoogLeNet: Inception module

![Inception module](../imgs/18_googlenet_inception.pdf){width=100%}

## GoogLeNet: Diagram

![GoogLeNet Diagram](../imgs/18_googlenet.pdf){width=10%}

## VGGNet

The second-place team (first losing team?) in ILSVRC 2014 was the VGGNet, which
showed that the depth of the network is a critical component of performance.
This network uses a lot more parameters (140M), and therefore memory and
computation, but later iterations removed some of these parameters.

## VGGNet: Diagram

![VGG Configuration](../imgs/18_vgg_config.pdf){width=40%}

## ResNet

Deeper nets seem to work better, so won't we get increasing performance if we
just keep slamming layers together?

It turns out that each time a gradient is "propagated"
down through a very long set of layers, the value of the gradient decreases.
This is called the ***vanishing gradient*** problem.

To solve this, ResNet uses ***skip connections***: In a normal network, the
activation at a layer is $y=f(x)$, where $f(x)$ is our nonlinear function that
is differentiated during backpropagation. In a skip connection, this is
redefined as: $y=f(x) + x$, which allows the gradient to be preserved as it
travels back through the network.

## ResNet: Skip Connections

![Skip Connection Diagram](../imgs/18_resnet_skip.pdf){width=60%}

## ResNet: Diagram

![ResNet](../imgs/18_resnet.pdf){width=20%}


# What does a CNN "See"?

## So... What's it... Doing?

A huge elephant in the room when discussing CNNs is: What is actually going on
at each of these layers? How does it actually work?

To answer this, folks have tried to look at what's going on at different points
in the "process".

For example, Mahendran and Vedaldi (2014) showed that you can "invert" the
representations of the image at layers in the CNN by using a natural image as a
prior and then presenting random noise to the network. Then you look at the
outputs of the layers.

## Example Inverted Images

![Original Images](../imgs/18_inversion_orig.pdf){width=100%}

## Example Inverted Images

!["Inverted" Images](../imgs/18_inversion_reconstruction.pdf){width=100%}

## Intermediate Layer Representations

\begincols

\column{0.5\linewidth}

![Original Image](../imgs/18_orig_tissue.pdf){width=80%}

\column{0.5\linewidth}

![Tissue Map from Deep Learning](../imgs/18_DL_result.pdf){width=80%}

\stopcols

## AlexNet Architecture

![Modified AlexNet Setup](../imgs/18_alexnet_caffe.pdf){width=10%}

## Intermediate Layer Activations

\begincols
\centering

\column{0.5\linewidth}
\center
![Activation Layer 01](../imgs/18_activation01.pdf){width=50%}
![Activation Layer 03](../imgs/18_activation03.pdf){width=50%}

\column{0.5\linewidth}
\center
![Activation Layer 01](../imgs/18_activation02.pdf){width=50%}
![Activation Layer 03](../imgs/18_activation04.pdf){width=50%}

\stopcols

## Artistic Style vs. Content

So with that understanding, Gatys, et al. (2015) realized that you can separate
the "content" of the image (represented at deeper layers as objects and
locations) from the "pixel values" or "style" of the image (represented at
shallow layers, right after the first convolutional layers).

Essentially you can pull the textures from one image, the content from another,
and mix them!

## Artistic Style Framework

![Content Versus Style](../imgs/18_artistic_diagram.png){width=60%}

## Artistic Style Examples

![Neckarfront in T{\"u}bingen, Germany (Photo: Andreas Praefcke)](../imgs/18_artistic_style_orig.png){width=60%}

## Artistic Style Examples

![\textit{The Shipwreck of the Minotaur} by J.M.W. Turner, 1805](../imgs/18_artistic_style_01.png){width=60%}

## Artistic Style Examples

![\textit{The Starry Night} by Vincent van Gogh, 1889](../imgs/18_artistic_style_02.png){width=60%}

## Artistic Style Examples

![\textit{Der Schrei} by Edvard Munch, 1893](../imgs/18_artistic_style_03.png){width=60%}

## Artistic Style Examples

![\textit{Femme nue assise} by Pablo Picasso, 1910](../imgs/18_artistic_style_04.png){width=60%}

## Artistic Style Examples

![\textit{Composition VII} by Wassily Kandinsky, 1913](../imgs/18_artistic_style_05.png){width=60%}

## Artistic Style At Home!

One of the great things about recent research is the availability of open source
tools for implementing these techniques.

With just a bit of set-up, you can implement these techniques on your own
datasets. For example, check out
***[https://github.com/ebenolson/pydata2015](https://github.com/ebenolson/pydata15)***.

## Kyoshi!

![Kyoshi, A Good Girl](../imgs/18_kyoshi_sm.png){width=40%}

## Artistic Kyoshi!

![Application of Artistic Styles](../imgs/18_kyoshi_artstyle.png){width=70%}

## Deep Dreaming and Beyond

Finally, you can do some really crazy stuff when you start looking at these
networks in detail and examining / manipulating their outputs...

[Google's Deep Dream (Pt. 1)](http://googleresearch.blogspot.com/2015/06/inceptionism-going-deeper-into-neural.html)

[Google's Deep Dream (Pt. 2)](http://googleresearch.blogspot.com/2015/07/deepdream-code-example-for-visualizing.html)

# Next Class

## Keep Reading!

CNNs are extremely popular, and there are new architectures and tweaks coming
out all the time.

Keep an eye on developments in this field, but remember to ***keep it simple***!

Don't use a fancy new classifier just because you can, use it if you benefit
from it!

## Variational Autoencoders and Generative Networks

There are two other CNN architectures I'd like to cover:

- ***Variational Autoencoders:*** These learn image structure from unlabeled samples,
  providing a way to do dimensionality reduction and comparison in an "image
  space" defined by a CNN.
- ***Generative Adversarial Networks:*** These networks are similar to 
  DeepDream, where they are able to create images based on their "understanding"
  of the image space.
  
After this we will move on to non-image sequence datasets using ***Recurrent
Neural Networks (RNNs)***.

