---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade
slide-level: 2

title: Dimensionality Reduction
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
contact: scottdoy@buffalo.edu
date: 2020-03-31
---

# 
## Recap

## Recap: Principal Component Analysis

PCA is a method of projecting data by calculating a set of vectors that captures the variance or "spread" of the data.

<p class="fragment">Recall that eigenvectors and eigenvalues can represent the **direction** and **magnitude** (respectively) of something represented in a matrix.</p>

<p class="fragment">If our matrix is the covariance matrix \$\\boldsymbol{\\Sigma}\$, then the eigenvectors represent the direction of the data "spread", while the eigenvalues are the magnitude of that spread.</p>

<p class="fragment">Thus we can express the data as a lower-dimensional projection by choosing a set of eigenvectors corresponding to the largest eigenvalues.</p>

## Recap: Are Principal Components Always Orthogonal?

The covariance matrix \$\\boldsymbol{\\Sigma}\$ is always positive and symmetric, having dimension \$d\\times d\$ where \$d\$ is the number of dimensions.

<p class="fragment">We can prove that because of this, it has \$d\$ **distinct, positive eigenvalues**, each of which corresponds to an **orthonormal** eigenvector.</p>

# 
## Independent Component Analysis

## Motivation for ICA

PCA tries to represent data by an optimized projection of the data using the covariance of the samples.

<p class="fragment">ICA seeks out independent **generating components** of the data.</p>

<p class="fragment">Suppose you have \$d\$ independent, **noiseless** source signals \$x\_{i}(t)\$ for \$i=1,\\\cdots,d\$, where \$t\$ is our time component \$1\\leq t\\leq T\$.</p>

<p class="fragment">We denote by \$\\mathbf{x}(t)\$ the \$d\$ values of the mixed signal at time \$t\$, and assume that the mean of \$\\mathbf{x}\$ over time is zero.</p>

<p class="fragment">The multivariate density function is then written as:</p>

<p class="fragment">\$ p\\left[\\mathbf{x}(t)\\right]=\\prod\_{i=1}\^{d}p\\left[x\_{i}(t)\\right] \$</p>

## Illustration of Source Signals \$\\mathbf{x}(t)\$

<div class="l-double">
<div>
![](img/ica_signal_01.svg){width=100%}
</div>
<div>
![](img/ica_signal_02.svg){width=100%}
</div>
</div>

Example of two source signals, \$x\_{1}(t)\$ and \$x\_{2}(t)\$.

## Sources and Signals

The source signals are detected by a \$k\$-dimensional sensor:

<p class="fragment">\$ \\mathbf{s}(t)=\\mathbf{A}\\mathbf{x}(t) \$</p>

<p class="fragment">where \$\\mathbf{A}\$ is a \$k\\times d\$ matrix representing the individual modulation of the \$d\$ source signals with respect to the \$k\$ detectors.</p>

<p class="fragment">**Example**: If \$\\mathbf{x}\$ is a set of sound waves produced by \$d\$ instruments, and \$\\mathbf{s}\$ is an array of \$k\$ microphones that are recording the sound, then \$\\mathbf{A}\$ might represent the distance between each specific microphone and instrument.</p>

<p class="fragment">**Goal**: Extract the \$d\$ components in \$\\mathbf{s}\$ that are independent.</p>

<p class="fragment">Note that we're ignoring the effects of noise, time delay, and possible dependence of one signal on another.</p>

## Illustration of Detected Signals \$\\mathbf{s}(t)\$

<div class="l-multiple">
<div>
![](img/ica_sensed_signal_01.svg){width=100%}
</div>
<div>
![](img/ica_sensed_signal_02.svg){width=100%}
</div>
<div>
![](img/ica_sensed_signal_03.svg){width=100%}
</div>
</div>

The source signals are sensed by an array of \$k\$ detectors, each of which receives a different mixture of \$x\_{1}(t)\$ and \$x\_{2}(t)\$.

## Jacobian Matrix

The distribution in the output signals is related to the distribution:

<p class="fragment">\$ p\_\\mathbf{y}=\\frac{p\_{\\mathbf{s}}(\\mathbf{s})}{|\\mathbf{J}|} \$</p>

<p class="fragment">where \$\\mathbf{J}\$ is the Jacobian matrix:</p>

<p class="fragment">
\$ \\mathbf{J}=\\left(
\\begin{matrix}
\\frac{\\partial y\_{1}}{\\partial s\_{1}} \& \\cdots \& \\frac{\\partial y\_{d}}{\\partial s\_{1}} \\\\
\\vdots \& \\ddots \& \\vdots \\\\
\\frac{\\partial y\_{1}}{\\partial s\_{d}} \& \\cdots \& \\frac{\\partial y\_{d}}{\\partial s\_{d}}
\\end{matrix}\\right) \$
</p>

<p class="fragment">and</p>

<p class="fragment">\$ |\\mathbf{J}|=\\left| |\\mathbf{W}|\\prod\_{i=1}\^{d}\\frac{\\partial y\_{i}}{\\partial s\_{i}}\\right| \$</p>

## Reconstructed Output Signal

The final stage is modeled as a linear transform of the source signals, plus a static nonlinearity:

<p class="fragment">\$ \\mathbf{y}=f[\\mathbf{Ws}+\\mathbf{w}\_{0}] \$</p>

<p class="fragment">where \$\\mathbf{w}\_{0}\$ is a bias vector and \$f[\\cdot]\$ is some kind of function (e.g. a sigmoid).</p>

<p class="fragment">The goal in ICA is to find \$\\mathbf{W}\$ and \$\\mathbf{w}\_{0}\$ so as to make the outputs \$y\_{i}\$ as independent from one another as possible.</p>

<p class="fragment">This is motivated by the fact that **we know** (i.e. we assume) the original signals themselves were independent.</p>

## Finding \$\\mathbf{W}\$

So to find our matrix, we can calculate \$\\mathbf{W}\$ and \$\\mathbf{w}\_{0}\$ iteratively, by defining a cost function, finding the derivative, and setting that to zero.

<p class="fragment">The goal is to find the set of components which are **maximally independent**, so our "cost" function should be a measure of independence for signals that we can try to maximize.</p>

## Illustration of Recovered Source Signals

<div class="l-double">
<div>
![](img/ica_recovered_signal_01.svg){width=100%}
</div>
<div>
![](img/ica_recovered_signal_02.svg){width=100%}
</div>
</div>

The reconstructed source signals are found by transforming the detected signals
by a set of learned weights \$\\mathbf{W}\$.

## Compare Source and Recovered Signals

<div class="l-double">
<div>
![](img/ica_signal_01.svg){width=100%}
</div>
<div>
![](img/ica_recovered_signal_blank_01.svg){width=100%}
</div>
</div>

The reconstructed source signals are found by transforming the detected signals
by a set of learned weights \$\\mathbf{W}\$.

## Compare Source and Recovered Signals

<div class="l-double">
<div>
![](img/ica_signal_02.svg){width=100%}
</div>
<div>
![](img/ica_recovered_signal_blank_02.svg){width=100%}
</div>
</div>

The reconstructed source signals are found by transforming the detected signals
by a set of learned weights \$\\mathbf{W}\$.

## Finding \$\\mathbf{W}\$: Measuring Independence

We use **joint entropy** to measure independence:

<p class="fragment">
\\begin{align}
H(\\mathbf{y}) \&= -\\mathcal{E}[\\ln{p\_{\\mathbf{y}}(\\mathbf{y})}] \\\\
\&=\\mathcal{E}[\\ln{|\\mathbf{J}|}]-\\underbrace{\\mathcal{E}[\\ln{p\_{\\mathbf{s}}(\\mathbf{s})}]}_{\\textrm{independent of weights}} \\
\\end{align}
</p>

<p class="fragment">\$\\mathcal{E}\$ is the expected value across all \$t=1,\\ldots,T\$.</p>

<p class="fragment">Through gradient descent we find the learning rule for \$\\mathbf{W}\$:</p>

<p class="fragment">\$ \\Delta\\mathbf{W}\\propto\\frac{\\partial H(\\mathbf{y})}{\\partial\\mathbf{W}}=\\frac{\\partial}{\\partial\\mathbf{W}}\\ln{|\\mathbf{J}|}=\\frac{\\partial}{\\partial\\mathbf{W}}\\ln{|\\mathbf{W}|}+\\frac{\\partial}{\\partial\\mathbf{W}}\\ln{\\prod\_{i=1}\^{d}\\left|\\frac{\\partial y\_{i}}{\\partial s\_{i}}\\right|} \$</p>

## Finding \$\\mathbf{W}\$: Cofactors and Inverse Matrices

In component form we can write the first term as:

<p class="fragment">\$ \\frac{\\partial}{\\partial W\_{ij}}\\ln{|\\mathbf{W}|}=\\frac{\\textrm{cof}[W\_{ij}]}{|\\mathbf{W}|} \$</p>

<p class="fragment">
where \$\\textrm{cof}[W\_{ij}]\$ is the cofactor of \$W\_{ij}\$, or \$(-1)\^{i+j}\$ times
the determinant of the \$(d-1)-by-(k-1)\$-dimensional matrix gotten by deleting
the \$i\$th row and \$j\$th column of \$\\mathbf{W}\$.
</p>

<p class="fragment">This gives us:</p>

<p class="fragment">\$ \\frac{\\partial}{\\partial \\mathbf{W}}\\ln{|\\mathbf{W}|}=[\\mathbf{W}\^{T}]\^{-1} \$</p>

<p class="fragment">Which, in turn, gives the weight update rule for \$\\mathbf{W}\$:</p>

<p class="fragment">\$ \\Delta\\mathbf{W}\\propto[\\mathbf{W}\^{T}]\^{-1}+(\\mathbf{1}-2\\mathbf{y})\\mathbf{s}\^{T}\_{g} \$</p>

## Bias \$\\mathbf{w}\_{0}\$ Learning Rule

It can be shown that with the same sets of assumptions, the learning rule for
the bias weights is:

<p class="fragment">\$ \\Delta\\mathbf{w}\_{0}\\propto\\mathbf{1}-2\\mathbf{y} \$</p>

<p class="fragment">
It's difficult to know how many components we should try to reconstruct; if the
number is too high, ICA may be sensitive to numerical simulation and may be
unreliable.
</p>

<p class="fragment">This is a potentially useful alternative to PCA, if we suspect that our classes are elongated in parallel.</p>

## Full Illustration of ICA

![Full Illustration of ICA](img/ica_full_system.svg){width=70%}

# 
## Curse of Dimensionality

## Problems with High Dimensional Visualization

Using "similarity" instead of "distance" loses some intuitive interpretation
about our data's structure.

<p class="fragment">
When this data is in very high dimensions, it's impossible for us to visualize,
even if the mathematics works perfectly.
</p>

<p class="fragment">
We'd like to figure out a way to represent data in a low number (1-3) of
dimensions while preserving the similarity between points.
</p>

<ul>
<li class="fragment">We can see the data in a way that makes sense to us (i.e. using distance as a reliable surrogate of similarity).</li>
<li class="fragment">We can see the data at all (in three or fewer spatial dimensions).</li>
</ul>

## Motivation for Low-Dimensional Representation

First: Why do we NEED to see things in low dimensions?

<p class="fragment">
Will a classifier work the same on the low-dimensional representation as it does
on the high-dimensional one?
</p>

<p class="fragment">Are these methods only to help us humans visualize the data?</p>

<p class="fragment">**No!** We are always constrained by the curse of dimensionality.</p>

<p class="fragment">We've discussed it before, but now let's examine it in detail.</p>

## Accuracy in High Dimensions

The curse of dimensionality seems paradoxical at first: if features are
statistically independent, and the class means are different, shouldn't we
**always** do better?

<p class="fragment">
Consider the two class case where \$p(\\mathbf{x}|\\omega\_{j})\\sim
N(\\boldsymbol{\\mu}\_{j},\\boldsymbol{\\Sigma})\$ for \$j=1,2\$.
</p>

<p class="fragment">**Assuming equal priors** (just to make it simple), the Bayes error rate is:</p>

<p class="fragment">\$ P(e)=\\frac{1}{\\sqrt{2\pi}}\\int\_{r\/2}\^{\\infty}e\^{-\\frac{r\^{2}}{2}}du \$</p>

<p class="fragment">where \$r\^2\$ is the Mahalanobis distance between the class means:</p>

<p class="fragment">\$ r\^{2}=(\\boldsymbol{\\mu}\_{1}-\\boldsymbol{\\mu}\_{2})\^{T}\\boldsymbol{\\Sigma}\^{-1}(\\boldsymbol{\\mu}\_{1}-\\boldsymbol{\\mu}\_{2}) \$</p>

## Accuracy in High Dimensions

\$ P(e)=\\frac{1}{\\sqrt{2\pi}}\\int\_{\\frac{r}{2}}\^{\\infty}e\^{-\\frac{r\^{2}}{2}}du \$

<p class="fragment">
How does our probability of error change as the distance between the class means
\$r\$ increases?
</p>

<p class="fragment">
Assuming conditional indpenedence,
\$\\boldsymbol{\\Sigma}=diag(\\sigma\_{1}\^{2},\\ldots,\\sigma\_{d}\^{2})\$ and:
</p>

<p class="fragment">\$ r\^{2}=\\sum\_{i=1}\^{d}\\left(\\frac{\\mu\_{i1}-\\mu\_{i2}}{\\sigma\_{i}}\\right)\^{2} \$</p>

<p class="fragment">How does this change as \$d\$ gets larger?</p>

<p class="fragment">**We are adding more components, and thus (potentially) increasing \$r\$**.</p>

## Illustration of Increasing Dimensionality

<div class="l-double">
<div>
![](img/cod_illustration.svg){width=80%}
</div>
<div>
In the one-dimensional space, there is some significant overlap (i.e. Bayes
error) between the features.

When we add dimensions, we see a reduction in this overlap; in the third
dimensions, the class spaces are completely separate, and the Bayes error is
zero.
</div>
</div>

## Theory vs. Reality

**In theory**, the worst feature will have identical class means, so that
\$\\mu\_{i1}-\\mu\_{i2}=0\$, so \$r\$ will not increase at that point in the sum.

<p class="fragment">
If \$r\$ is increased without limit, then our error should theoretically approach
zero! (Which obviously doesn't happen.)
</p>

<p class="fragment">
**So what's wrong?** The reason not all classifiers are perfect comes down to
one of these:
</p>

<ul>
<li class="fragment">Our assumption that the features are independent is wrong.</li>
<li class="fragment">Our underlying model for our distributions is wrong.</li>
<li class="fragment">Our training samples are finite, so we cannot accurately estimate the distributions.</li>
</ul>

<p class="fragment">
**Dimensionality reduction** methods seek to address these issues by
projecting the data into a low-dimensional space, combining dependent variables
and ignoring non-informative ones.
</p>

# 
## Multidimensional Scaling

## Simple Case Example

Let \$\\mathbf{y}\_{i}\$ be a projection of a sample \$\\mathbf{x}\_{i}\$.

<p class="fragment">
\$\\delta\_{ij}\$ (delta) is the distance between \$\\mathbf{x}\_{i}\$ and
\$\\mathbf{x}\_{j}\$, and \$d\_{ij}\$ (lowercase "d")is the distance between
\$\\mathbf{y}\_{i}\$ and \$\\mathbf{y}\_{j}\$.
</p>

<p class="fragment">
Thus we want to find how to arrange \$\\mathbf{y}\_{1},\\ldots,\\mathbf{y}\_{n}\$ such
that the distances \$d\_{ij}\$ are as close as possible to \$\\delta\_{ij}\$.
</p>

## Simple Case Illustration

![3D to 2D mapping, with Euclidean distances for \$\\delta\_{ij}\$ and
\$d\_{ij}\$.](img/mds_illustration.svg){ width=80%}

## Criterion Functions for MDS

We can set up a few criterion functions:

<table>
<tr>
<td>**Equation**</td>
<td>**Characteristic**</td>
</tr>
<tr>
<td>\$J\_{ee}=\\frac{\\sum\_{i<j}(d\_{ij}-\\delta\_{ij})\^{2}}{\\sum\_{i<j}\\delta\_{ij}\^{2}}\$</td>
<td>Emphasizes large error, regardless of original distance</td>
</tr>
<tr>
<td>\$J\_{ff}=\\sum\_{i<j}\\left(\\frac{d\_{ij}-\\delta\_{ij}}{\\delta\_{ij}}\\right)\^{2}\$</td>
<td>Emphasizes proportional error, regardless of actual error</td>
</tr>
<tr>
<td>\$J\_{ef}=\\frac{1}{\\sum\_{i<j}\\delta\_{ij}}\\sum\_{i<j}\\frac{(d\_{ij}-\\delta\_{ij})\^{2}}{\\delta\_{ij}}\$</td>
<td>Compromise between the two</td>
</tr>
</table>

All are invariant to rigid transforms and are normalized.

## Gradients for Criterion Functions

Gradients are easy to compute: The gradient of \$d\_{ij}\$ with respect to
\$\\mathbf{y}\_{i}\$ is a unit vector in the direction of
\$\\mathbf{y}\_{i}-\\mathbf{y}\_{j}\$:

<p class="fragment">\$ \\boldsymbol{\\nabla}\_{\\mathbf{y}\_{k}}J\_{ef}=\\frac{2}{\\sum\_{i<j}\\delta\_{ij}}\\sum\_{j\\neq k}\\frac{d\_{kj}-\\delta\_{kj}}{\\delta\_{kj}}\\frac{\\mathbf{y}\_{k}-\\mathbf{y}\_{j}}{d\_{kj}} \$</p>

<p class="fragment">Example: 30 points spaced at unit intervals along a spiral, which circles around the \$x\_{3}\$ axis:</p>

<p class="fragment">
\\begin{align}
x\_{1}(k) \&= \\cos{\\left(\\frac{k}{\\sqrt{2}}\\right)} \\\\
x\_{2}(k) \&= \\sin{\\left(\\frac{k}{\\sqrt{2}}\\right)} \\\\
x\_{3}(k) \&= \\frac{k}{\\sqrt{2}} \\\\
\\end{align}
</p>

## Illustration of Spiral MDS

![3D to 2D embedding with \$J\_{ef}\$.](img/mds_spiral.svg){ width=80% }

## Wrapup of Linear Methods

One thing to keep in mind is that these methods are **linear** -- they cannot
encode nonlinear relationships between datapoints.

<p class="fragment">
A nonlinear dataset is one in which linear distances (e.g. Euclidean distance)
between points is **not** a reliable measure of similarity.
</p>

<p class="fragment">What are some examples?</p>

## Examples of Nonlinear Datasets

![](img/swissroll_unlabeled.svg){width=80%}

## How to Know if Datasets are Nonlinear?

As we've seen, unsupervised methods rely on data structure to "tell a story" --
therefore, if your linear method is applied to a nonlinear dataset, and you just
get a big blob, how do you know what's wrong?

<p class="fragment">
You need a **large number** of samples before you can conclude that your
dataset is nonlinear (and for nonlinear methods to work at all).
</p>

<p class="fragment">
The swiss roll is nonlinear in 3 dimensions, but if your data is nonlinear in a
million dimensions, it'll be tough to know ahead of time.
</p>

<p class="fragment">
Therefore, validation of these methods will require **some** amount of
labeling or ground truth.
</p>

# 
## Locally Linear Methods

## Various Manifold Methods

If you have a lot of data, you can assume it lies on a well-defined manifold --
that is, the points form a "sheet" in high-dimensional space.

<p class="fragment">
The only distances that are valid are those between a point and its close
neighbors; everything else is invalid (i.e. the distance is "infinite").
</p>

<p class="fragment">
Think of houses on switchback streets: if you're in a car, and not driving over
people's lawns, you have to travel the road to get to your neighbor's house.
</p>

## Common Themes

There are a lot of these methods, but they all make some pretty basic
assumptions:

<ul>
<li class="fragment">Points in high-dimensional space lie on a manifold;</li>
<li class="fragment">There are enough points in the datasest to "define" that manifold;</li>
<li class="fragment">You can set a neighborhood parameter to define how far apart points can be without "hopping" onto another part of the manifold;</li>
<li class="fragment">It's possible to devise a mapping such that \$\\delta\_{ij}\$ and \$d\_{ij}\$ are similar for pairs of points \$\\mathbf{x}\_{i}, \\mathbf{x}\_{j}\$ and their projections \$\\mathbf{y}\_{i}, \\mathbf{y}\_{j}\$</li>
</ul>

## Do It Yourself!

There are a lot of nonlinear methods. 

<ul>
<li class="fragment">**Isometric mapping** (ISOMAPS)</li>
<li class="fragment">**Locally Linear Embedding** (LLE)</li>
<li class="fragment">**DBSCAN**</li>
<li class="fragment">**t-Distributed Stochastic Neighbor Embedding** (t-SNE)</li>
<li class="fragment">**UMAP**</li>
</ul>

<p class="fragment">
We will cover the last two, as they are what most people gravitate towards these
days, and it's implemented in MATLAB.
</p>

# 
## t-Distributed Stochastic Neighbor Embedding

## Formulation of t-SNE

This is a relatively new method developed in 2008 by:

<ul>
<li class="fragment">Geoffrey Hinton, who has done great foundational work in neural networks</li>
<li class="fragment">Laurens van der Maaten, who has written extensively on dimensionality reduction</li>
</ul>

<p class="fragment">
It is a **nonlinear, probabilistic** technique which is also solved through
numerical optimization (i.e., gradient descent).
</p>

<p class="fragment">
This means that running t-SNE multiple times may give you **different
embeddings**.
</p>

## How Does t-SNE Work?

Let's assume we have our \$d\$-dimensional dataset, \$\\{\\mathbf{x}\_{1},
\\mathbf{x}\_{2}, \\ldots, \\mathbf{x}\_{N}\\}\$.

<p class="fragment">
We assume that \$d\$ is fairly large, and that all features are relevant to
describing the dataset.
</p>

<p class="fragment">
The goal is to find a mapping where **similarity between points is
preserved**, which in t-SNE is done by modeling between-point similarity as a
probability distribution (and a Gaussian, no less). 
</p>

## Similarity as a Probability

If we assume the points are distributed as a Gaussian in high-dimensional space,
then the probability, \$p\_{j|i}\$, is understood as the likelihood that point
\$\\mathbf{x}\_{i}\$ would have \$\\mathbf{x}\_{j}\$ as its "neighbor" (in other words,
that the points would be close together). 

<p class="fragment">\$ p\_{j|i} = \\frac{\\exp\\left[-|\\mathbf{x}\_{i} - \\mathbf{x}\_{j}|\^{2}/ 2\\sigma\^{2}\_{i}\\right]}{\\sum\_{k\\neq i}\\exp\\left[-|\\mathbf{x}\_{i} - \\mathbf{x}\_{k}|\^{2} / 2\\sigma\^{2}_{i}\\right]} \$</p>

<p class="fragment">
If you look closely, this is just a Gaussian where one of the points serves as
the "mean". The value of \$\\sigma\$ is a parameter that can be derived from the
"complexity" of the data, and is a tunable parameter.
</p>

## Similarity as a Probability

By normalizing by the number of all points, we get a measure of point similarity:

<p class="fragment">\$ p\_{ij} = \\frac{p\_{i|j} + p\_{j|i}}{2N} \$</p>

## Similarity in the Low Dimensional Space

Next, we define a similar point-wise probability for the points in **low**
dimensional space as well. Here, high-dimensional point \$\\mathbf{x}\_{i}\$ is
referred to in low-dimensional space as \$\\mathbf{y}\_{i}\$.

<p class="fragment">Low-dimensional probability is defined as:</p>

<p class="fragment">\$ q\_{ij} = \\frac{\\left(1 + |\\mathbf{y}\_{i} - \\mathbf{y}\_{j}|\^{2}\\right)\^{-1} }{\\sum\_{k\\neq i}\\left(1 + |\\mathbf{y}\_{i} - \\mathbf{y}\_{k}|\^{2}\\right)\^{-1}} \$</p>

<p class="fragment">
This is NOT a Gaussian modeling: this is a Student-t distribution (hence,
"t-distributed SNE").
</p>

## Why Use Different Distributions?

The paper goes into detail for each of their choices, but the use of two
different distributions addresses some issues where outlier points have
incorrect amounts of influence on the resulting mapping.

<p class="fragment">
The Student-t distribution works like an inverse square law for large distances,
meaning that the scale of the mapping doesn't affect the result.
</p>

## Putting it Together

So we have a probability distribution on the data in high dimensions, \$p\_{ij}\$,
which we can calculate. We also have a form for the low-dimensional
distribution, \$q\_{ij}\$, but we can't calculate that because we don't know what
\$\\mathbf{y}\$ should be.

<p class="fragment">
To find the location of the points \$\\mathbf{y}\$, we minimize the
Kullback-Leibler Divergence, which is defined as:
</p>

<p class="fragment">\$ KL(P||Q) = \\sum\_{i\\neq j}p\_{ij}\\log{\\frac{p\_{ij}}{q\_{ij}}} \$</p>

<p class="fragment">
The KL Divergence is basically how you calculate the difference between two
probability distributions.
</p>

<p class="fragment">How do we minimize? Gradient descent!</p>

## Parameters, Parameters Everywhere

As with any numerical optimization approach, we have to think about:

<ul>
<li class="fragment">Our **learning rate**</li>
<li class="fragment">Our **search time**</li>
<li class="fragment">Our **initial conditions** (i.e. local vs. global minima)</li>
</ul>

<p class="fragment">
Because of this, t-SNE can give you different results if run multiple times
(because of local minima), and selection of parameter values is **critically
important**.
</p>

<p class="fragment">
So interpreting your results should be done with a fair amount of caution. Make
sure you test robustness by running the algorithm multiple times with different
parameter sets on data you know is clean.
</p>

## Still More Parameters 

In addition, t-SNE has its own tunable parameters:

<ul>
<li class="fragment">"Perplexity", which is related to the bandwidth of the Gaussians used to model probability densities in high dimensions</li>
<li class="fragment">Similarity metric, which in the original formulation is Euclidean</li>
</ul>

<p class="fragment">
Again: you need to check that what you're getting makes sense for your own data
and assumptions.
</p>

<p class="fragment">
Interpreting dimensionality reduction methods should be done with extreme
caution.
</p>

# 
## Parting Words

## Tip of the Iceberg

Unsupervised methods, clustering, and DR are obviously a HUGE topic.

<p class="fragment">They are typically the first thing you can do when you start collecting data.</p>

<p class="fragment">
Cheap, (somewhat) fast, and give you an idea of how well your calculated
features are doing.
</p>

<p class="fragment">
There are a ton of variations of what we've discussed, but if you are
interested, this is a good starting point.
</p>

## Next Topic

Neural networks are an extension of linear machines and will serve as the basis for deep learning.

<p class="fragment">Next class, we will begin building the foundations we'll need for understanding these complex classifiers.</p>
