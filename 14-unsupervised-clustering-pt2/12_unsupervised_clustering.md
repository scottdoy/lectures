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
	| UNSUPERVISED  
	| CLUSTERING (Pt. 1)
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
institute: scottdoy@buffalo.edu
date: March 26, 2019
---

# Recap Last Lectures

## Recap: Nature of Training Samples

A training set $\mathcal{D}$ is a ***limited random sampling*** of the feature space.

***Supervised Learning***: $\mathcal{D} = \{\mathbf{x}_{i}, y_{i}\}$, where $y_{i} \in \{\omega_{1}, \omega_{2}, \ldots, \omega_{c}\}$ is the $i$-th class label.

***Unsupervised Learning***: $\mathcal{D} = \{\mathbf{x}_{i}\}$, where we just have the values for $\mathbf{x}$, but no class label.

## Recap: Supervised Classification

We've seen several approaches to classification using class labels:

- ***Bayes***: Use $\mathcal{D}$ to predict $p(\omega_{i}|\mathbf{x})$
- ***Decision Tree***: Split $\mathcal{D}$ using optimal feature thresholds
- ***Linear Discriminants / SVMs***: Build a set of functions $g_{i}(\mathbf{x})$ such that the predicted label is $\hat{y} = \omega_{j}$, where $j=\argmax_{i}g_{i}(\mathbf{x})$
- ***$k$-Nearest Neighbor***: Predicted label $\hat{y}$ is the most-common label of the $k$-nearest neighbors of $\mathbf{x}$ according to the feature space

## Recap: Parametric Methods

$\mathcal{D}$ may come from an underlying ***parameterized*** distribution:

*Normal:*
: $\mathbf{\theta} = (\mu, \Sigma)$ (mean, covariance / variance)

*Log-normal:*
: $\mathbf{\theta} = (\sigma, \mu)$ (shape, log-scale)

*Binomial:*
: $\mathbf{\theta} = (n,p)$ (trials, success probability)

*Gamma:*
: $\mathbf{\theta} = (k,\theta)$ or $(\alpha, \beta)$ or $(k,\mu)$ (shape, scale/rate/mean)

*Weibull:*
: $\mathbf{\theta} = (k,\lambda,\theta)$ (shape, scale, location)

*Poisson:*
: $\mathbf{\theta} = (\lambda)$ (mean / variance)

## Recap: Parametric Methods

![](../imgs/03_probability_density_function.pdf)

## Recap: Choosing your Distribution

Each of these has its own form!

Choose the one that:

- Describes your data
- Has the fewest parameters
- Makes intuitive sense, given the source of the feature
- (Is the Normal Distribution)

## Recap: Nonparametric methods

What if:

- ... we don't know what our parametric form should be?
- ... our samples don't come from a single distribution?
- ... we have way too few samples to even estimate our parameters?

In these cases, we need non-parametric methods.

## Recap: Two Methods for Finding Densities

![Parzen Windows (top) and $k$-Nearest Neighbors (bottom) for estimating density at a point as $N\rightarrow\infty$.](../imgs/10_convergence_both.pdf){width=100%}

## Recap: Parzen Windows (Effect of Width on the Window)

![Effect of $h$ (width) on the "window" $\delta_{n}(\mathbf{x})$.](../imgs/10_histwidth.pdf){width=100%}

## Recap: Parzen Windows (Effect of Width on the Density)

![Effect of $h$ (width) on the density estimates $p_{n}(\mathbf{x})$.](../imgs/10_histwidth_pts.pdf)

## Recap: Classification With Parzen Windows

\begincols

\column{0.5\linewidth}

![Small $h$](../imgs/10_parzen_smallbin.pdf){width=60%}

\column{0.5\linewidth}

![Large $h$](../imgs/10_parzen_largebin.pdf){width=60%}

\stopcols

Classification using Parzen Windows. If $h$ is small, we risk over-training. If
$h$ is large, we will provide greater generalizability at the possible expense
of accuracy.

## Recap: Classification with $k$NN

\begincols

\column{0.5\linewidth}

![](../imgs/10_knn_classifier_01.png){width=80%}

\column{0.5\linewidth}

Classification with $k$NN is so simple, it essentially "skips" the consideration
of densities and just labels samples by counting the neighbors that belong to
each class.

\stopcols

## Recap: Classification with $k$NN

\begincols

\column{0.5\linewidth}

![](../imgs/10_knn_classifier_voronoi.pdf){width=80%}

\column{0.5\linewidth}

![](../imgs/10_knn_classifier_3d.pdf){width=80%}

\stopcols

# Clustering: Mixture Densities

## Clustering Introduction

In almost all situations, gathering class labels is ***hard***.

We are often presented with datasets having many unlabeled samples:

- Uploaded photos (Facebook, Google+, the NSA)
- Recorded telephone audio (Call centers, speech therapy, the NSA again)
- Recorded video segments (Security firms, video game cameras, also the NSA)

For some kinds of problems, you can ***crowd-source*** labeling.

## Cost of Acquiring Labels

\begincols

\column{0.5\linewidth}

![](../imgs/11_cat02.jpg){height=80%}

\column{0.5\linewidth}

![](../imgs/11_dog.jpg){height=80%}

\stopcols

## Cost of Acquiring Labels

\begincols

\column{0.5\linewidth}

![](../imgs/02_fna_91_5691_malignant.gif){width=100%}

\column{0.5\linewidth}

![](../imgs/02_fna_92_5311_benign.gif){width=100%}

\stopcols

## Why Use Unlabeled Data?

Without labels, can we learn about categories?

- You can cluster, and then perform "hands-on" labeling.
- You can get a "free" look at the data before doing labeling, to get an "unbiased" look at the structure of the feature space. 
- You can perform "weak" supervised classification, a.k.a. ***semi-supervised*** learning

## Semi-Supervised Learning

From Zhou, Z. "A Brief Introduction to Weakly Supervised Learning":

*Incomplete Supervision:*
: Only a small subset of the available data is labeled

*Inexact Supervision:*
: Only coarse-level annotations are provided

*Inaccurate Supervision:*
: Labels may be incorrectly assigned

## Known Knowns, Known Unknowns

As ever, we start with some basic assumptions:

- We know the number of classes, $c$.
- We know the priors, $P(\omega_{j})$, for $j=1,\cdots,c$.
- We know the form of $p(\mathbf{x}|\omega_{j},\mathbf{\boldsymbol{\theta}}_{j})$ (e.g. Gaussian).
- We do NOT know the parameter vectors $\boldsymbol{\theta_{1}}, \cdots, \boldsymbol{\theta_{c}}$.
- We do NOT know the actual category labels for the samples.

## Mixture Densities

We can start by modeling the observed density, $p(\mathbf{x}|\boldsymbol{\theta})$, as a mixture of $c$ different class densities:

$$ p(\mathbf{x}|\boldsymbol{\theta})=\sum_{j=1}^{c}p(\mathbf{x}|\omega_{j},\boldsymbol{\theta}_{j})P(\omega_{j}) $$

... where $\boldsymbol{\theta}=\left(\boldsymbol{\theta}_{1}, \cdots, \boldsymbol{\theta}_{c}\right)^{t}$.

In this form, $p(\mathbf{x}|\boldsymbol{\theta})$ is known as a ***mixture density***.

Conditional densities $p(\mathbf{x}|\omega_{j},\boldsymbol{\theta}_{j})$ are the ***component densities***.

Priors $P(\omega_{j})$ are the ***mixing parameters***.

## Component Densities and Mixing Parameters

![Observed Sample Distribution](../imgs/11_gaussian_mixture_fused.pdf)

## Component Densities and Mixing Parameters

![Underlying Component Distributions](../imgs/11_gaussian_mixture_1d.pdf)


## Identifiability of a Density

$$ p(\mathbf{x}|\boldsymbol{\theta})=\sum_{j=1}^{c}p(\mathbf{x}|\omega_{j},\boldsymbol{\theta_{j}})P(\omega_{j}) $$

Our unknown is our set of parameters $\boldsymbol{\theta}$, so that's what we
want to estimate.

Once we have our parameter sets, we can "un-mix" the density, figure out what
classes are the largest contributors at each point of $\mathbf{x}$, and then
classify new points accordingly.

One valid question: if we have an infinite number of samples, and we know the
underlying form of $p(\mathbf{x}|\boldsymbol{\theta})$, then is
$\boldsymbol{\theta}$ unique?

A density $p(\mathbf{x}|\boldsymbol{\theta})$ is ***identifiable*** if
$\boldsymbol{\theta}\neq\boldsymbol{\theta}^{\prime}$ implies that there exists
an $\mathbf{x}$ such that $p(\mathbf{x}|\boldsymbol{\theta})\neq
p(\mathbf{x}|\boldsymbol{\theta}^{\prime})$.

# Maximum Likelihood Estimation

## Maximum Likelihood Estimates

Suppose $p(\mathbf{x}|\boldsymbol{\theta})$ gives us a set
$\mathcal{D}=\{\mathbf{x}_{1},\cdots,\mathbf{x}_{n}\}$

The likelihood of observing a specific $\mathcal{D}$ is the joint density:

$$ p(\mathcal{D}|\boldsymbol{\theta})=\prod_{k=1}^{n}p(\mathbf{x}_{k}|\boldsymbol{\theta}) $$

## Maximizing our Dataset Probability

If we're looking to find $\hat{\boldsymbol{\theta}}$ which maximizes
$p(\mathcal{D}|\boldsymbol{\theta})$, we have to do the whole log-likelihood /
gradient thing:

$$ l=\sum_{k=1}^{n}\ln{p(\mathbf{x}_{k}|\boldsymbol{\theta})} $$

$$\nabla_{\boldsymbol{\theta}_{i}}l=\sum_{k=1}^{n}\frac{1}{p(\mathbf{x}_{k}|\boldsymbol{\theta})}\nabla_{\boldsymbol{\theta}_{i}}\left[\sum_{j=1}^{c}p(\mathbf{x}_{k}|\omega_{j},\boldsymbol{\theta}_{j})P(\omega_{j})\right] $$

## Maximum Likelihood Estimates

If we introduce the posterior probability, we can write in terms of the
component densities:

$$ P(\omega_{i}|\mathbf{x}_{k},\boldsymbol{\theta})=\frac{p(\mathbf{x}_{k}|\omega_{i},\boldsymbol{\theta}_{i})P(\omega_{i})}{p(\mathbf{x}_{k}|\boldsymbol{\theta})} $$

Then we can rewrite the previous derivative as:

$$\nabla_{\boldsymbol{\theta}_{i}}l	= \sum_{k=1}^{n}P(\omega_{i}|\mathbf{x}_{k},\boldsymbol{\theta}) \nabla_{\boldsymbol{\theta}_{i}}\ln{p(\mathbf{x}_{k}|\omega_{i},\boldsymbol{\theta}_{i})} $$

As always, we set this to zero and solve for the class-specific parameters
$\boldsymbol{\theta}_{i}$.

# Normal Mixtures: Estimating $\boldsymbol{\mu}$

## Normal Mixtures and Additional Assumptions

We've already assumed we know the form of the density (namely, that it's
Gaussian).

There are four parameters that we may not know:

- $\boldsymbol{\mu}_{i}$, the multivariate mean;
- $\boldsymbol{\Sigma}_{i}$, the covariance matrix;
- $P(\omega_{i})$, the prior probability; and
- $c$, the total number of classes.

Just like linear discriminants, for simplicity we start by assuming that we know three of
these: $\boldsymbol{\Sigma}_{i}$, $P(\omega_{i})$, and $c$.

## Case 1: Unknown Mean Vectors

Once again, we take the log-likelihood of the Gaussian for simplicity:

$$ \ln{p(\mathbf{x}|\omega_{i},\boldsymbol{\mu}_{i})}=-\ln{\left[(2\pi)^{\frac{d}{2}} |\boldsymbol{\Sigma}_{i}|^{\frac{1}{2}} \right]} -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_{i})^{t}\boldsymbol{\Sigma}_{i}^{-1}(\mathbf{x}-\boldsymbol{\mu}_{i}) $$

... take the derivative...

$$ \nabla_{\boldsymbol{\mu}_{i}}\ln{p(\mathbf{x}|\omega_{i},\boldsymbol{\mu}_{i})}=\boldsymbol{\Sigma}_{i}^{-1}(\mathbf{x}-\boldsymbol{\mu}_{i}) $$

... and drop it into the old MLE equation for finding
$\hat{\boldsymbol{\theta}}$, which, to refresh your memory, is:

$$ \sum_{k=1}^{n}P(\omega_{i}|\mathbf{x}_{k},\boldsymbol{\theta})\nabla_{\boldsymbol{\theta}_{i}}\ln{p(\mathbf{x}_{k}|\omega_{i},\boldsymbol{\theta}_{i})}=0 $$

## Case 1: Unknown Mean Vectors

Thus we have:

$$ \sum_{k=1}^{n}P(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\mu}})\boldsymbol{\Sigma}_{i}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}_{i}})=0 $$

If we multiply by $\boldsymbol{\Sigma}_{i}$ and moving around terms, we are left
with:

$$ \hat{\boldsymbol{\mu}}_{i} = \frac{\sum_{k=1}^{n}P(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\mu}})\mathbf{x}_{k}}{\sum_{k=1}^{n}P(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\mu}})} $$

***In other words***, $\hat{\boldsymbol{\mu}}_{i}$ is a weighted average of the
samples.

The weight for the $k$th sample is an estimate of how likely it is that
$\mathbf{x}_{k}$ belongs to the $i$th class.

Does this equation give us $\hat{\boldsymbol{\mu}}_{i}$?

## Explicit Solution or Iterative Estimates?

Unfortunately, we can't solve $\hat{\boldsymbol{\mu}}_{i}$ explicitly.

What we CAN do is perform an iterative search: Select an initial value for
$\hat{\boldsymbol{\mu}}_{i}(0)$, then solve:

$$ \hat{\boldsymbol{\mu}}_{i}(j+1)=\frac{\sum_{k=1}^{n}P(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\mu}}(j))\mathbf{x}_{k}}{\sum_{k=1}^{n}P(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\mu}}(j))} $$

This is essentially a ***gradient ascent*** method, meaning that eventually we
will converge to a point where
$\hat{\boldsymbol{\mu}}_{i}(j+1)\approx\hat{\boldsymbol{\mu}}_{i}(j)$

As with all iterative approaches, once the gradient is zero, we are only ensured
that we've reached a ***local*** maximum.

## Example Solution for Finding $\boldsymbol{\mu}_{i}$

Let's say we have the following two-component, one-dimensional normal mixture:

$$ p(x|\mu_{1},\mu_{2})=\underbrace{\frac{1}{3\sqrt{2\pi}}\exp\left[-\frac{1}{2}(x-\mu_{1})^{2}\right]}_{\omega_{1}} +\underbrace{\frac{2}{3\sqrt{2\pi}}\exp\left[-\frac{1}{2}(x-\mu_{2})^{2}\right]}_{\omega_{2}} $$

We pull a set of $k=25$ samples sequentially from this mixture.

These samples are used to calculate the log-likelihood function:

$$ l(\mu_{1},\mu_{2})=\sum_{k=1}^{n}\ln{p(x_{k}|\mu_{1},\mu_{2})} $$

## Example Solution for Finding $\boldsymbol{\mu}_{i}$: Illustration

![](../imgs/11_sample_means_example_observed.pdf)

## Example Solution for Finding $\boldsymbol{\mu}_{i}$: Illustration

![](../imgs/11_sample_means_example.pdf)

## Example Solution for Finding $\boldsymbol{\mu}_{i}$: Illustration

![2D Mixture Density](../imgs/11_2d_gaussian_mle_mixture.pdf)

## Unlabeled Densities: Importance of Sample Size

As ever, it's important to have a significant number of samples to estimate your
parameters.

What would it look like if we had insufficient samples for this problem?

## Unlabeled Densities: Few Samples

![Low Number of Observations](../imgs/11_cluster_examples_few.pdf)

## Unlabeled Densities: More Samples

![Larger Number of Observations](../imgs/11_cluster_examples_mid.pdf)

## Unlabeled Densities: Lots of Samples

![Lots of Observations](../imgs/11_cluster_examples_many.pdf)

# Normal Mixtures: Estimating $\boldsymbol{\Sigma}$

## Case 2: Unconstrained Covariance

If no constraints are placed on $\boldsymbol{\Sigma}_{i}$, then we have a
problem...

Let's say our two-component mixture is given like so:

$$ p(x\textbar{}\mu,\sigma^{2}) = \underbrace{\frac{1}{2\sqrt{2\pi}\sigma}\exp\left[-\frac{1}{2}\left(\frac{x-\mu}{\sigma}\right)^{2}\right]}_{\text{First Component}} + \underbrace{\frac{1}{2\sqrt{2\pi}}\exp\left[-\frac{1}{2}x^{2}\right]}_{\text{Second Component}} $$

In other words, the second component has $\mu=0$, $\sigma=1$.

Now let's set the mean of the first component to $\mu=x_{1}$. For
$p(x_{1}|\mu,\sigma^{2})$, the first term's exponential goes to 0, and so we
have:

$$ p(x_{1} | \mu, \sigma^{2}) = \frac{1}{2\sqrt{2\pi}\sigma} + \frac{1}{2\sqrt{2\pi}}\exp\left[-\frac{1}{2}x_{1}^{2}\right] $$

## Case 2: Unconstrained Covariance

For $x_{k}\neq\mu$, the first term's exponential remains. Thus:

$$ p(x_{k} | \mu,\sigma^{2}) \geq \frac{1}{2\sqrt{2\pi}}\exp\left[-\frac{1}{2}x_{k}^{2}\right] $$

The joint probability for $x_{1},\cdots,x_{n}$ is the product of all of the
above, so with some rearranging we have:

$$ p(x_{1},\cdots,x_{n} | \mu,\sigma^{2}) \geq \left[\frac{1}{\sigma}+\exp\left[-\frac{1}{2}x_{1}^{2}\right]\right]\frac{1}{(2\sqrt{2\pi})^{n}}\exp\left[-\frac{1}{2}\sum_{k=2}^{n}x_{k}^{2}\right] $$

## All Parameters Unknown: Exploding Equations with Small Variance

$$ p(x_{1},\cdots,x_{n} | \mu,\sigma^{2}) \geq \left[\frac{1}{\sigma}+\exp\left[-\frac{1}{2}x_{1}^{2}\right]\right]\frac{1}{(2\sqrt{2\pi})^{n}}\exp\left[-\frac{1}{2}\sum_{k=2}^{n}x_{k}^{2}\right] $$

What's wrong with this equation? Specifically, look at $\sigma$...

If $\sigma$ is small, the equation explodes. That is to say, the MLE solution is
***singular***.

## Fixing Exploding Equations

So how do we get around this? Put some constraints on $\boldsymbol{\Sigma}_{i}$!

Recall the likelihood equation when finding $\boldsymbol{\mu}_{i}$:

$$ \ln{p(\mathbf{x}|\omega_{i},\boldsymbol{\mu}_{i})}=-\ln{\left[(2\pi)^{\frac{d}{2}} |\boldsymbol{\Sigma}_{i}|^{\frac{1}{2}} \right]}  -\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu}_{i})^{t}\boldsymbol{\Sigma}_{i}^{-1}(\mathbf{x}-\boldsymbol{\mu}_{i}) $$

Just like before, we want to differentiate and set to zero, but this time we
have to differentiate with respect to both $\boldsymbol{\mu}_{i}$ ***and***
$\boldsymbol{\Sigma}_{i}$.

## Solving for Everything

Remember the relation we had previously:

$$ \sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})\nabla_{\boldsymbol{\theta}_{i}}\ln{p(\mathbf{x}_{k}|\omega_{i},\hat{\boldsymbol{\theta}}_{i}) = 0} $$

I am skipping a lot of the math, but we end up with the following:

\begin{align*}
\hat{P}(\omega_{i}) &= \frac{1}{n}\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}}) \\
\hat{\boldsymbol{\mu}_{i}} &= \frac{\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})\mathbf{x}_{k}}{\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})} \\
\hat{\boldsymbol{\Sigma}_{i}} &= \frac{\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})^{t}}{\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})}
\end{align*}

## Explanations for Everything

Okay, so what do those mean?

\begin{columns}<1->[t,onlytextwidth]
	\column{0.5\textwidth}
		$\hat{P}(\omega_{i}) = \frac{1}{n}\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})$
	\column{0.5\textwidth}
		The likelihood of $\omega_{i}$ is the fraction of samples from $\omega_{i}$
\end{columns}

\begin{columns}<1->[t,onlytextwidth]
	\column{0.5\textwidth}
		$\hat{\boldsymbol{\mu}_{i}} = \frac{\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})\mathbf{x}_{k}}{\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})}$
	\column{0.5\textwidth}
		$\hat{\boldsymbol{\mu}}_{i}$ is the mean of those samples in $\omega_{i}$
\end{columns}

\begin{columns}<1->[t,onlytextwidth]
	\column{0.5\textwidth}
		$\hat{\boldsymbol{\Sigma}_{i}} = \frac{\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})^{t}}{\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})}$
	\column{0.5\textwidth}
		$\hat{\boldsymbol{\Sigma}}_{i}$ is the corresponding covariance matrix
\end{columns}

If $\hat{P}(\omega_{i} | \mathbf{x}_{k}, \hat{\boldsymbol{\theta}})$ is between
$0.0$ and $1.0$, then all samples play a role in the estimates (not just those
belonging to $\omega_{i}$).

## One More Equation

One final point: Above we've written
$\sum_{k=1}^{n}\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})$ a
lot.

For each of the above three equations, we can substitute Bayes equation here:

\begin{align*}
	\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}}) &= \frac{ p(\mathbf{x}_{k}|\omega_{i},\hat{\boldsymbol{\theta}}_{i})\hat{P}(\omega_{i}) }{ \sum_{j=1}^{c}p(\mathbf{x}_{k}|\omega_{j},\hat{\boldsymbol{\theta}}_{j})\hat{P}(\omega_{j}) }\\
	 &= \frac{ |\boldsymbol{\hat{\Sigma}}_{i}|^{-\frac{1}{2}} \exp\left[-\frac{1}{2}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})^{t}\hat{\boldsymbol{\Sigma}}_{i}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})\right]\hat{P}(\omega_{i})}{ \sum_{j=1}^{c}|\hat{\boldsymbol{\Sigma}}_{j}|^{-\frac{1}{2}}\exp\left[-\frac{1}{2}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{j})^{t}\hat{\boldsymbol{\Sigma}}_{j}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{j})\right] \hat{P}(\omega_{j})}
\end{align*}

## Explicit Description of Our Parameter Set

$$\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}}) = \frac{|\boldsymbol{\hat{\Sigma}}_{i}|^{-\frac{1}{2}}\exp\left[-\frac{1}{2}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})^{t}\hat{\boldsymbol{\Sigma}}_{i}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})\right]\hat{P}(\omega_{i})}{\sum_{j=1}^{c}|\hat{\boldsymbol{\Sigma}}_{j}|^{-\frac{1}{2}}\exp\left[-\frac{1}{2}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{j})^{t}\hat{\boldsymbol{\Sigma}}_{j}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{j})\right]\hat{P}(\omega_{j})}$$

Much simpler, right?

This is just to illustrate that now, we DO have an explicit way to calculate each of our unknown parameters given our sample set.

## Okay, So Now What?

Once again, we can take an iterative approach to solving for our parameter sets:

$$ \hat{\boldsymbol{\mu}}_{i}(j+1)=\frac{\sum_{k=1}^{n}P(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\mu}}(j))\mathbf{x}_{k}}{\sum_{k=1}^{n}P(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\mu}}(j))} $$

# $k$-Means Clustering

## Mahalanobis Distance

\begin{align*}
\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}}) &= \frac{|\boldsymbol{\hat{\Sigma}}_{i}|^{-\frac{1}{2}}\exp\left[-\frac{1}{2}
\tikz[baseline]{
        \node[fill=blue!20,anchor=base] (t1)
            {$(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})^{t}\hat{\boldsymbol{\Sigma}}_{i}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})$};
        }
\right]\hat{P}(\omega_{i})}{\sum_{j=1}^{c}|\hat{\boldsymbol{\Sigma}}_{j}|^{-\frac{1}{2}}\exp\left[-\frac{1}{2}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{j})^{t}\hat{\boldsymbol{\Sigma}}_{j}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{j})\right]\hat{P}(\omega_{j})}\\
\end{align*}

The value of $\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})$ is high if the ***squared Mahalanobis distance*** \tikz[baseline]{
	\node[fill=blue!20,anchor=base] (t1)
	{$(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})^{t}\hat{\boldsymbol{\Sigma}}_{i}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})$};
 } is small.

Intuitively, this just means that if a sample is close to the centroid of a class cluster, then the likelihood of it belonging to the associated class is high.

## Calculating Probabilities Via Distance to Centroids

If we replace the Mahalanobis with the squared Euclidean distance $\|\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i}\|^{2}$, we can find the mean $\hat{\boldsymbol{\mu}}_{m}$ nearest to $\mathbf{x}_{k}$.

Thus we can approximate $\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})$ as:

$$ \hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}}) \simeq \left\{
\begin{array}{l l}
	1 & \quad \textrm{if $i=m$} \\
	0 & \quad \textrm{otherwise.} \\
\end{array} \right. $$

Then we can plug this into the equation we got before and solve for $\hat{\boldsymbol{\mu}}_{1},\cdots,\hat{\boldsymbol{\mu}}_{c}$.

We can ``initialize'' by selecting $c$ class centroids at random from the unlabeled data, and then iterating.

## Algorithm for $k$-Means Clustering

\begin{algorithm}[H]
	\caption{$k$-Means Clustering}
	\emph{begin initialize $n,c,\boldsymbol{\mu}_{1},\cdots,\boldsymbol{\mu}_{c}$}\;
	\Repeat{no change in $\boldsymbol{\mu}_{i}$}{
		classify $n$ samples according to nearest $\boldsymbol{\mu}_{i}$\;
		recompute $\boldsymbol{\mu}_{i}$\;
	}
	\KwRet{$\boldsymbol{\mu}_{1},\cdots,\boldsymbol{\mu}_{c}$}
\end{algorithm}

## $k$-Means Example: Hill-climbing

![Stoachastic hill-climbing in $k$-means.](../imgs/11_kmeans_hill_climb.pdf)

## Comparing $k$-Means and MLE

\begincols

\column{0.5\linewidth}

![MLE Example](../imgs/11_2d_gaussian_mle_mixture.pdf){width=80%}

\column{0.5\linewidth}

![$k$-Means Example](../imgs/11_kmeans_hill_climb.pdf){width=80%}

\stopcols

Comparison of MLE and $k$-Means. Since the overlap between the components is
relatively small, we arrive at basically the same answers for $\mu_{1}$ and
$\mu_{2}$.

## $k$-Means In Sample Space

![$k$-Means Sample Space](../imgs/11_kmeans_sample_space.pdf)

## $k$-Means Summary

$k$-Means is a staple of unsupervised clustering methods.

It is simple and relatively fast, operating in $O(ndcT)$ time where $n$ is the
number of patterns, $d$ is the number of features, $c$ is the number of clusters
and $T$ is the number of iterations.

It typically finishes in a small number of iterations.

## $k$-Means Caveats

When does it fail?

- If we are wrong about our number of classes, we will converge on two means
  that don't mean anything.
- If the clusters are too close to one another, there may not be enough samples
  to properly ``ascend'' to the true value of $\mu$.

Remember: ***everything*** is dependent on your features!

# $k$-Means Example (Matlab)
# Evaluating Clusters

## How Well Did We Cluster?

So far, we've been talking about ***clusters*** and ***classes*** as being
synonymous, but that's clearly not the case.

An underlying density with multiple modes could represent the same class, while
a unimodal density may represent multiple classes (i.e. it's a bad feature for
discrimination).

We make assumptions about the form of the data, but clearly we could estimate
perfectly valid parameter sets and completely miss the underlying structure of
the data.

## Examples of Misleading Parameters

![Distributions with equal $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$.](../imgs/11_misleading_clusters.pdf)

## Similarity Measures

If we are interested in finding ***subclasses*** in our unlabeled data, we have
to define what we mean.

Finding "clusters" seems like a good place to start, but how do we define a
cluster?

We need a way to say that samples in one cluster are more similar to each other
than they are to samples in another cluster.

We also need a way to evaluate a given partitioning of the dataset, to say that
Clustering Result A is better or worse than Clustering Result B.

We can even start by using a distance metric to define our clusters in the first
place!

## Examples of Distance Thresholding: High Threshold

![High Threshold Cluster](../imgs/11_cluster_high_thresh.pdf)

Clustering with a high threshold means that almost everything is put into the
same cluster. In graph theory, this would be called a ***fully connected***
graph and is akin to ***under-training***.

## Examples of Distance Thresholding: Medium Threshold

![Medium Threshold Cluster](../imgs/11_cluster_mid_thresh.pdf)

Clustering with an intermediate threshold can reveal structure in your data, but
it is up to the designer to say whether this is the ***correct*** structure.

## Examples of Distance Thresholding: Low Threshold

![Low Threshold Cluster](../imgs/11_cluster_low_thresh.pdf)

Clustering with a low threshold forces only the closest points to be associated
with the same cluster. This is akin to ***over-training***.

## Choosing a Similarity Metric: Invariance to Transforms

Generally speaking, we assume the correct distance metric is Euclidean (the
shortest distance between two points is a straight line).

However, this is only justified if the space is ***isotropic*** (distances in
one dimension are equivalent to those in another) and the data is evenly spread.

Euclidean distance is robust to ***rigid*** transformations of the feature space
(rotation and translation).

It is NOT robust to arbitrary linear transformations which distort distance
relationships.

## Rotation's Effect on Cluster Groupings

![Cluster Scaling](../imgs/11_cluster_scaling.pdf){width=40%}

## Choosing a Similarity Metric: Normalization

We can try to counteract the effects of non-isotropy in our feature space
through ***normalization***, where we translate and scale the data to have a
zero mean and unit variance.

We can also rotate the coordinate space so that the axes coincide with the
eigenvectors of the covariance matrix; this is part of ***principal
component analysis (PCA)*** which we will cover shortly.

However, normalization has its own problems...

## Normalization Sometimes Ruins Our Clusters

![Normalization](../imgs/11_cluster_normalization.pdf)

Normalization can take a well-separated dataset with two processes (left) and
smoosh it together so we end up with a single cloud (right).

# Parting Words

## Benefits of Unsupervised Methods

Question: Will supervised methods always out-perform unsupervised methods?

As always, ***it depends!*** Having more information (like labels) is better,
but over-fitting is always a danger.

Over-fitting isn't just fine-tuning your decision hyperplanes, it is also when
you choose $c$ to be too high, or if you pick too many mixtures for each of your
densities.

Sometimes, clustering can give you insight above and beyond what you
***thought*** you knew!

- If you think you have two classes, and you find three clusters, what does that
  mean?
- If you have five classes and you find two clusters, what does THAT mean?

# Next Class

## Similarity Metric Selection

In most situations, the choice of similarity metric / distance function will be
arbitrary.

In the literature, Euclidean distance metric is usually assumed, unless you have
reason to believe otherwise.

With non-linear datasets, this may NOT true... but that's for next class.

We will also look at evaluating our clusters by calculating ***criterion
functions***: sum-of-squared-error, minimum variance, scatter matrices, trace /
determinant criterions, and invariant criterions.

(We may not look at all of those, but enough to give you the idea.)
