---
# YAML Preamble
fontsize: 10pt
documentclass: beamer
classoption:
- xcolor=table
- aspectratio=169
theme: metropolis
slide-level: 2

title: NONPARAMETRIC METHODS
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
institute: scottdoy@buffalo.edu
date: March 14, 2019
---

# Recap

## Recap: Parameter Estimation Foundations

Core assertion in model-building and parameter estimation: Values of a random variable are observed according to a known probability law.

"Randomness" refers to the ***unpredictable variations*** in an observed value.

These variations can occur with a specific, pre-determined likelihood -- the ***probability law***.

## Recap: Parameter Estimation Foundations

The probability law (i.e. distribution) is ***known*** or assumed. 

This distribution governs the general pattern of observed variations.

We can estimate the parameters of the distribution (denoted $\boldsymbol{\theta}$).

## Recap: Relating Data to a Model

\begincols


\column{0.4\linewidth}

![Relating Data to a Model](../imgs/01_scatter_nonlinear.pdf){width=100%}

\column{0.6\linewidth}

- Collect samples of the random variable.
- Decide on a model (typically Gaussian).
- Identify the parameter set (if Gaussian, $\boldsymbol{\theta}=(\mu, \Sigma)$).
- Finally, we assume that samples in $\mathcal{D}_{i}$ give no information about $\boldsymbol{\theta}_{j}$ if $i\neq j$.
- Thus our task is to estimate a total of $c$ parameter sets $\boldsymbol{\theta}_{i}$, $i\in\{1,\ldots,c\}$.

\stopcols

## Recap: Two Approaches to Parameter Estimation


\begincols

\column{0.5\linewidth}

***Maximum Likelihood Estimation (MLE)***

- $\boldsymbol{\theta}_{i}$ is a set of fixed, unknown quantities that we are trying to discover.
- The estimated values are those that maximize the probability of observing our training data.

\column{0.5\linewidth}

***Bayesian Estimation (BAY)***

- $\boldsymbol{\theta}_{i}$ is a set of random variables, each with a known prior distribution.
- Training data observations turn these priors into posterior densities.
- More training "sharpens" the density near the true values of the parameters
-- this is known as ***Bayesian Learning***

\stopcols

## Recap: MLE vs. BAY

***Computational Efficiency***: MLE is simpler

***Interpretability***: MLE yields single values, BAY yields a distribution

***Model Confidence***: MLE relies on a good model, BAY explicitly accounts for uncertainty

***Bias-Variance***: MLE can correct for bias-variance, BAY handles the tradeoff through  uncertainty

***Overall***: MLE is simpler, BAY uses more information

## Recap: BAY Estimation Tied to Sample Size

Our sample-size-dependent estimation of parameters:

\begin{align*}
\mu_{n} &= \left( \frac{n\sigma_{0}^{2}}{n\sigma_{0}^{2} + \sigma^{2}} \right) \widehat{\mu}_{n} + \frac{ \sigma^{2} }{ n\sigma_{0}^{2} + \sigma^{2} } \mu_{0} \\
\sigma_{n}^{2} &= \frac{ \sigma_{0}^{2} \sigma^{2} }{ n\sigma_{0}^{2} +
\sigma^{2} }
\end{align*}

If $\sigma_{n}^{2}$ is our "uncertainty", as $n \rightarrow \infty$, our
uncertainty goes towards zero: More samples = less uncertainty.

If $\mu_{n}$ is our "best guess", as $n \rightarrow \infty$, it is a linear
combination of $\widehat{\mu}_{n}$ (the sample mean) and $\mu_{0}$ (our best
prior guess for $\mu$).

As long as $\sigma_{0}^{2} \neq 0$, then $\mu_{n}$ approaches the sample mean as
$n \rightarrow \infty$.

# Nonparametric Introduction

## Check Our Assumptions

The core assumption for parametric techniques relies on a known parameterized probability law.

What if we have ***no idea*** what model governs our sample values, or if our model is a poor match for our samples?

In other words, we either have ***no model*** or a ***poorly fit model***.

We will examine a few nonparametric techniques, where we assume essentially "arbitrary distributions" (those without a form) or unknown distributions.

## Histogram Methods

\begincols

\column{0.5\linewidth}

![](../imgs/10_histograms.png){width=100%}

\column{0.5\linewidth}

Recall our training: $\mathcal{D} = \{x_{1}, \ldots, x_{N}\}$.

Our goal is to model $p(x)$ from $\mathcal{D}$.

In a nonparametric setting, we can choose a bin size, $\Delta_{i}$, and count
the number of points falling into bin $i$.

To convert into a normalized density, divide by the total number of observations
$N$ and the width $\Delta_{i}$:

$$p_{i} = \frac{ n_{i} }{ N\Delta_{i} } $$

Hence, the model for the density is constant over each bin (typically all
$\Delta_{i}$ are equal).

\stopcols

## Histograms By Width

\begincols

\column{0.5\linewidth}

![](../imgs/10_histogram_binsize.png){width=100%}

\column{0.5\linewidth}

- Green: True density (Gaussians)
- Purple: Randomly-observed values

Choosing a bin width that is too small (top) or too large (bottom) give
inaccurate representations.

***It's hard to select $\Delta$ properly.***

Remember: $N$ is often small, and we don't know the real distribution.

\stopcols

## Histogram Methods

We can continuously estimate $p(x)$ as we collect training.

Binning introduces discontinuities at the edges of the bin, and scales poorly as the dimensionality increases.

Thus we should do the following:

- Break the feature domain into "regions", and  consider sample observations within some "neighborhood".
- Choose a "bin width" or smoothness parameter to accurately represent the feature density.

The process of calculating the density across a region is known as ***density estimation***.

# Density Estimation

## General Density Estimation

Just as we did with the histograms, we assume that our sample values fall within
some region $\mathcal{R}$; the probability that a point falls within
$\mathcal{R}$ is:

$$ P = \int_{\mathcal{R}} p(\mathbf{x}^{\prime}) d\mathbf{x}^{\prime} $$

If we observe $n$ samples, then the probability that $k$ fall within
$\mathcal{R}$ is given by the ***binomial distribution***:

$$ P_{k} = {n \choose k} P^{k} (1 - P)^{n - k}, {n \choose k} = \frac{ n! }{ k!(n - k)!} $$

And the expected value of $k$ is thus:

$$ \mathcal{E}[k] = nP $$

## General Density Estimation

\begincols

\column{0.5\linewidth}

![Binomial Distribution](../imgs/10_binomial_distribution.png){width=100%}

\column{0.5\linewidth}

$P_{k}$ peaks sharply around the mean, so we expect that $k/n$ is a pretty good
estimate of $P$.

This estimate gets more and more accurate as $n$ increases (i.e. as we observe
more samples).

\stopcols

## General Density Estimation

\begincols

\column{0.5\linewidth}

![Relative Probability](../imgs/10_relative_probability.pdf){width=100%}

\column{0.5\linewidth}

If we assume that $p(\mathbf{x})$ is continuous and $\mathcal{R}$ is really small so $p(\mathbf{x})$ does not vary appreciably within it, then we have:

$$ \int_{\mathcal{R}} p(\mathbf{x}^{\prime}) d\mathbf{x}^{\prime} \simeq p(\mathbf{x})V $$

$V$ is the volume enclosed by $\mathcal{R}$.

\stopcols

## Pratical Concerns

We combine the previous equations to get:

$$ p(\mathbf{x}) \simeq \frac{ k/n }{ V } $$

However, this leaves some problems.

We said that $\mathcal{R}$ is small, so $p(\mathbf{x})$ doesn’t vary, but it
should be large enough that the number of points falling inside it yields a
sharply peaked binomial.

We can fix the volume $V$ and increase the number of training samples, but this
will only yield an estimate of the space-averaged density $P/V$ (we want the
actual $P$).

Also, since $V$ is the space enclosed by $\mathcal{R}$, if it is TOO small, then
no points will fall inside it and we’ll end up with $p(\mathbf{x}) \simeq 0$.

## Getting Around our Limitations

So let's start by assuming we have an infinite number of samples.

To estimate the density at $\mathbf{x}$, first form a sequence of regions
$\mathcal{R}_{1}, \mathcal{R}_{2}, \ldots$ where $\mathcal{R}_{1}$ contains 1
sample, $\mathcal{R}_{2}$ contains two, and so on.

We can write $V_{n}$ as the volume of $\mathcal{R}_{n}$, and $k_{n}$ is the
number of samples falling in $\mathcal{R}_{n}$:

$$ p_{n}(\mathbf{x}) = \frac{ k_{n} / n }{ V_{n} } = \frac{ k_{n}  }{ nV_{n} } $$

As $n \rightarrow \infty$, we want $p_{n}(\mathbf{x})$ (the density dependent on
our sample size) to converge to $p(\mathbf{x})$ (the true density around
$\mathbf{x}$).

For this to happen, we must satisfy a few conditions as $n \rightarrow \infty$.

## Convergence Requirements

$$ p_{n}(\mathbf{x}) = \frac{ k_{n} }{ nV_{n} } $$

Here are our conditions:

- $\lim_{n\rightarrow\infty} V_{n} = 0$ ensures the space-averaged $P/V$ will
  converge to $p(\mathbf{x})$.
- $\lim_{n\rightarrow\infty} k_{n} = \infty$ ensures that the frequency ratio
  will converge to $P$, e.g. that the binomial distribution will be sufficiently
  peaked.
- $\lim_{n\rightarrow\infty} k_{n} / n = 0$ is required for $p_{n}(\mathbf{x})$
  to converge at all; by specifying that as $n\rightarrow\infty$, the region
  $\mathcal{R}_{n}$ will get a large amount of samples, but they form a very
  small fraction of the overall number of samples.

## Two Methods for Convergence

So how do we build a region that specifies these conditions?

***Parzen Windows***: Shrink a region by specifying volume $V_{n}$ as a function
of $n$, such as $V_{n} = 1/\sqrt{n}$. Then, we'll show that $p_{n}(\mathbf{x})$
converges to $p(\mathbf{x})$.

***K-Nearest Neighbor***: Specify $k_{n}$ as a function of $n$, such as $k_{n} =
\sqrt{n}$. Then, the volume $V_{n}$ is grown until it encloses $k_{n}$ neighbors
of $\mathbf{x}$.

Both of these methods converge (of course, as $n\rightarrow\infty$).

## Two Methods for Convergence

![Convergence Methods](../imgs/10_convergence_both.pdf){width=100%}

# Parzen Windows

## Parzen Windows

Let's start by assuming the region $\mathcal{R}$ is a $d$-dimensional hypercube,
with $h_{n}$ being the length of an edge. Thus the volume is given by:

$$ V_{n} = h_{n}^{d} $$

We can obtain an analytic expression for $k_{n}$, the number of samples falling
in the hypercube, by defining a ***window function***:

$$ \varphi(\mathbf{u}) =
\begin{cases}
1 & \quad \abs{u_{j}} \leq 1/2 \quad \forall j \\
0 & \quad \text{otherwise}
\end{cases} $$

This function defines a unit hypercube centered at the origin.

## Window Function

$$ \varphi(\mathbf{u}) =
\begin{cases}
1 & \quad \abs{u_{j}} \leq 1/2 \quad \forall j \\
0 & \quad \text{otherwise}
\end{cases} $$

We can see that if $\mathbf{u} = (\mathbf{x} - \mathbf{x}_{i})/h_{n}$, then
$\varphi(\mathbf{u})$ is equal to 1 if $\mathbf{x}_{i}$ falls within the
hypercube of volume $V_{n}$ centered at $\mathbf{x}$, and is zero otherwise.

The number of samples in the hypercube is given by:

$$ k_{n} = \sum_{i=1}^{n} \varphi\left(\frac{\mathbf{x} - \mathbf{x}_{i}}{ h_{n} } \right) $$

Recall that $p_{n}(\mathbf{x}) = \frac{ k_{n} }{ nV_{n} }$, so we get:

$$ p_{n}(\mathbf{x}) = \frac{ 1 }{ n } \sum_{i=1}^{n} \frac{ 1 }{ V_{n} } \varphi \left( \frac{ \mathbf{x} - \mathbf{x}_{i} }{ h_{n} } \right) $$

## Probability Estimate

$$ p_{n}(\mathbf{x}) = \frac{ 1 }{ n } \sum_{i=1}^{n} \frac{ 1 }{ V_{n} } \varphi \left( \frac{ \mathbf{x} - \mathbf{x}_{i} }{ h_{n} } \right) $$

We can make $\varphi$ -- the ***Parzen window*** function -- to be any function
of $\mathbf{x}$, controlling how to weight samples in $\mathcal{D}$ to determine
$p(\mathbf{x})$ at a specific $\mathbf{x}$.

The equation for $p_{n}(\mathbf{x})$ is an ***average of functions*** of
$\mathbf{x}$ and $\mathbf{x}_{i}$.

Thus the ***window function*** is being used for ***interpolation***.

We can ensure the density is "regular" (non-negative, sums to 1) by the
conditions:

$$ \varphi(\mathbf{x}) \geq 0 $$

$$ \int \varphi(\mathbf{u}) d\mathbf{u} = 1 $$

## Effect of Window Width

As before, we must consider the effect of the window width $h_{n}$ on
$p_{n}(\mathbf{x})$.

Define the function:

$$ \delta_{n}(\mathbf{x}) = \frac{ 1 }{ V_{n} } \varphi\left(\frac{ \mathbf{x} }{ h_{n} } \right) $$

Substituting this into our equation for the density, we get:

$$ p_{n}(\mathbf{x}) = \frac{ 1 }{ n } \sum_{i=1}^{n} \delta_{n}( \mathbf{x} - \mathbf{x}_{i} ) $$

Since $V_{n} = h_{n}^{d}$, the width $h_{n}$ clearly affects both the amplitude
and the width of $\delta_{n}(\mathbf{x})$.

## Effect of Window Width

![Window Width Visualization](../imgs/10_histwidth.pdf){width=100%}

## Effect of Window Width

![Bin size on Points](../imgs/10_histwidth_pts.pdf){width=100%}

## Effect of Window Width

\begincols

\column{0.5\linewidth}

![Wide Histogram](../imgs/10_hist_wide.pdf){width=70%}

\column{0.5\linewidth}

The relationship between $h_{n}$ and the amplitude of $\delta_{n}$ is inverse.

If $h_{n}$ is large, then the amplitude of $\delta_{n}$ is small, and
$\mathbf{x}$ must be far from $\mathbf{x}_{i}$ before $\delta_{n}( \mathbf{x} -
\mathbf{x}_{i})$ deviates from $\delta_{n}(\mathbf{0})$.

If $h_{n}$ is small, then the peak value of $\delta_{n}(\mathbf{x} -
\mathbf{x}_{i})$ is large and occurs close to $\mathbf{x} = \mathbf{x}_{i}$
(i.e. $\delta_{n}(\mathbf{0})$).

\stopcols

## Effect of Window Width

\begincols

\column{0.5\linewidth}

![Thin Histogram](../imgs/10_hist_thin.pdf){width=70%}

\column{0.5\linewidth}

The relationship between $h_{n}$ and the amplitude of $\delta_{n}$ is inverse.

If $h_{n}$ is large, then the amplitude of $\delta_{n}$ is small, and
$\mathbf{x}$ must be far from $\mathbf{x}_{i}$ before $\delta_{n}( \mathbf{x} -
\mathbf{x}_{i})$ deviates from $\delta_{n}(\mathbf{0})$.

If $h_{n}$ is small, then the peak value of $\delta_{n}(\mathbf{x} -
\mathbf{x}_{i})$ is large and occurs close to $\mathbf{x} = \mathbf{x}_{i}$
(i.e. $\delta_{n}(\mathbf{0})$).

\stopcols

## Effect of Window Width

For any value of $h_{n}$, the distribution is normalized:

$$ \int \delta_{n}(\mathbf{x} - \mathbf{x}_{i}) d\mathbf{x} = \int \frac{ 1 }{ V_{n} } \varphi\left( \frac{ \mathbf{x} - \mathbf{x}_{i} }{ h_{n} } \right) d\mathbf{x} = \int \varphi(\mathbf{u})d\mathbf{u} = 1 $$

This means that as $h_{n}$ gets smaller and smaller, $\delta_{n}(\mathbf{x} -
\mathbf{x}_{i})$ approaches a Dirac delta function centered at $\mathbf{x}_{i}$,
and $p_{n}(\mathbf{x})$ approaches a superposition of deltas centered at the
samples.

When that happens, you'll need more and more samples to approximate the true
distribution of $p(\mathbf{x})$!

## Revisiting the Gaussian

![Gaussian Windows](../imgs/10_gaussian_windows.pdf){width=50%}

## Multimodal Distribution Estimation

![Gaussian Windows](../imgs/10_shape_windows.pdf){width=50%}

## Classification Using Parzen Windows

\begincols

\column{0.5\linewidth}

![Parzen Windows: Small Bin](../imgs/10_parzen_smallbin.pdf){width=80%}

\column{0.5\linewidth}

- We can estimate the densities for each class and classify a test point by
  assigning it to the class with the maximum posterior.
- The Parzen window-based classifier depends heavily on the form of the kernel
  function.
- The error can be made arbitrarily low by selecting a tiny $h_{n}$, but what does
  this lead to?

\stopcols

## Classification Using Parzen Windows

\begincols

\column{0.5\linewidth}

![Parzen Windows: Large Bin](../imgs/10_parzen_largebin.pdf){width=80%}

\column{0.5\linewidth}

- We can estimate the densities for each class and classify a test point by
  assigning it to the class with the maximum posterior.
- The Parzen window-based classifier depends heavily on the form of the kernel
  function.
- The error can be made arbitrarily low by selecting a tiny $h_{n}$, but what
  does this lead to?

\stopcols

## Summary of Parzen Windows

If you get confused about the densities, think in terms of building a histogram.

You want to know how many samples fall within each range of x-values.

You have to choose the bin width parameter to properly reveal the underlying
distribution.

Having more samples will lead you to a better estimation of the density; an
infinite number of samples will lead to a perfect estimation (convergence).

This is a simple method, but without much else to go on, it may give you a good
way of looking at your data.

# K-Nearest Neighbors

## Second Method for Convergence

![](../imgs/10_convergence.pdf){width=100%}

***K-Nearest Neighbor***: Specify $k_{n}$ as a function of $n$, such as $k_{n} =
\sqrt{n}$. Then, the volume $V_{n}$ is grown until it encloses $k_{n}$ neighbors
of $\mathbf{x}$.

## Limitation of Parzen Windows

Selecting the window size is typically done \textit{ad hoc}.

Instead of picking a window size and then seeing how the data fits, why not use the training data to dictate the size of the window?

K-NN allows the window size to be a function of observed samples:

- Center the window on $\mathbf{x}$
- Let the window grow until it captures $k_{n}$ nearby samples, where $k_{n}$ is a function of $n$.
- The samples within the window are the $k_{n}$ "nearest neighbors" of $\mathbf{x}$.

If density is high, then the window is small -- high resolution.

If density is low, then the window is large (but stops at an appropriate size).

## K-Nearest Neighbors

$$ p_{n}(\mathbf{x}) = \frac{ k_{n} }{ nV_{n} } $$

Recall our convergence discussion from last time...

We want $\lim_{n\rightarrow\infty}{k_{n}} = \infty$, assuring that $k_{n}/n$ will estimate $p_{n}(\mathbf{x})$.

We also want $\lim_{n\rightarrow\infty}{k_{n}/n} = 0$, ensuring $k_{n}$ grows more slowly than $n$.

These conditions ensure that $p_{n}(\mathbf{x})$ converges to $p(\mathbf{x})$.

## Examples of K-NN Densities

![K-NN Histogram](../imgs/10_knn_hist.png){width=50%}

## Examples of K-NN Densities

![K-NN Histogram](../imgs/10_knn_hist3d.png){width=40%}

## K-NN Estimation from a Single Sample

If $n=1$ and $k_{n} = \sqrt{n} = 1$, our estimate becomes:

$$ p_{n}(\mathbf{x}) = \frac{ 1 }{ 2 \abs{x - x_{1}} } $$

This diverges (as opposed to converging) to infinity -- so it's a poor estimate of $p(\mathbf{x})$.

However, the density never reaches 0 in the finite-sample case, because instead of defining the density by some arbitrary window, we define it by the nearest possible values of the random variable (which is always nonzero).

Since $n$ never actually reaches infinity, this is an okay tradeoff in most scenarios.

## K-Nearest Neighbor Estimates

![K-NN Estimates](../imgs/10_knn_estimates.png){width=45%}

## Limitations of K-NN

We saw in Parzen windows we had to choose the width carefully so that we generalized well without overfitting.

Here, we select $k_{n}$, e.g. the number of neighbors we use to grow the region to reach.

When training is limited, $k_{n}$ can drastically alter the form of the density.

As with everything else, the choice of $k_{n}$ is done based on what gives the best results (which can be evaluated in terms of classifier accuracy).

# Classification using K-NN

## Estimation of A Posteriori Probabilities

We can directly estimate the posterior probabilities $P(\omega_{i} | \mathbf{x})$ from a set of $n$ labeled samples by using them to estimate the densities.

Suppose we place a window of volume $V$ around $\mathbf{x}$ and capture $k$ samples, $k_{i}$ of which are labeled $\omega_{i}$.

Then we estimate the joint probability $p(\mathbf{x}, \omega_{i})$ as:

$$ p_{n}(\mathbf{x}, \omega_{i}) = \frac{ k_{i}/n }{ V } = \frac{ k_{i} }{ nV } $$

And so we use Bayes law to get the estimate for $P(\omega_{i} | \mathbf{x} )$:

$$ P_{n}( \omega_{i} | \mathbf{x} ) = \frac{ p_{n}(\mathbf{x}, \omega_{i}) }{ \sum_{j=1}^{c} p_{n}(\mathbf{x}, \omega_{j} ) } = \frac{ k_{i} }{ k } $$

## Classification Using K-NN

\begincols

\column{0.5\linewidth}

![K-NN Classifier](../imgs/10_knn_classifier_01.png){width=70%}

\column{0.5\linewidth}

$$ P_{n}(\omega_{i} | \mathbf{x} ) = \frac{ p_{n}(\mathbf{x}, \omega_{i}) }{ \sum_{j=1}^{c} p_{n}(\mathbf{x}, \omega_{j}) } = \frac{ k_{i} }{ k } $$

Thus, the probability that we observe $\omega_{i}$ given $\mathbf{x}$ is simply
the fraction of randomly-observed samples within the neighborhood of
$\mathbf{x}$ that are labeled $\omega_{i}$.

\stopcols

## Classification Using K-NN

![K-NN Voronoi](../imgs/10_knn_classifier_voronoi.png){width=45%}

## Classification Using K-NN

![K-NN Voronoi in 3D](../imgs/10_knn_classifier_3d.pdf){width=35%}

# Parting Words

## K-NN For Density Estimation

k-NN is a conceptually simple method: it says that samples are likely to belong
to the class of other samples that are nearby.

While technically it is a density estimation method, k-NN is often used to skip
straight to performing classification.

I'm sparing you some details about convergence and proof of the error bounds,
because they typically follow from the discussion about convergence with Parzen
Windows.

The book has the details if you're interested.

# Next Class

## Supervised vs. Unsupervised Classification

So far, we have assumed that we knew the labels associated with our samples:
$\mathcal{D}_{1}$ contains samples from $\omega_{1}$, and so on.

Obviously, this is not always (or even usually) the case.

If we just have a cloud of points, how do we decide how to best cluster and
classify points?

The next class, we will discuss clustering, expectation maximization, and some
  methods for unsupervised classification.
