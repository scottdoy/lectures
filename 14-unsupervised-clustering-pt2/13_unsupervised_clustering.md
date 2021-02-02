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
    | CLUSTERING (Pt. 2)
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
institute: scottdoy@buffalo.edu
date: March 28, 2019
---

# Recap

## Recap: Why Use Unlabeled Data?

- Samples are cheap to collect, costly to label.
- Clustering gives you a free(ish) look at the structure of your data.
- Clustering does not preclude performing hands-on labeling later.
- Unsupervised methods adapt to new trends in the data over time.
- Some methods learn features as well as class labels.

## Recap: Component Densities and Mixing Parameters

$$p(\mathbf{x}|\boldsymbol{\theta})=\sum_{j=1}^{c}p(\mathbf{x}|\omega_{j},\boldsymbol{\theta}_{j})P(\omega_{j})$$

In this form, $p(\mathbf{x}|\boldsymbol{\theta})$ is known as a ***mixture
density***.

Conditional densities $p(\mathbf{x}|\omega_{j},\boldsymbol{\theta_{j}})$ are the
***component densities***.

Priors $P(\omega_{j})$ are the ***mixing parameters***.

## Recap: Component Densities and Mixing Parameters

![Observed Sample Distribution](../imgs/11_gaussian_mixture_fused.pdf)

## Recap: Component Densities and Mixing Parameters

![Underlying Component Distributions](../imgs/11_gaussian_mixture_1d.pdf)

## Recap: Normal Mixtures and Additional Assumptions

We've already assumed we know the form of each mixture density (namely, they
are Gaussian).

There are four parameters that we may not know:

- $\boldsymbol{\mu}_{i}$, the multivariate mean;
- $\boldsymbol{\Sigma}_{i}$, the covariance matrix;
- $P(\omega_{i})$, the prior probability; and
- $c$, the total number of classes.

We CAN evaluate the system if we don't know anything.

## Recap: Estimating our Parameter Sets for Clustering

Our strategy for finding $c$ and $P(\omega_{i})$ is simply to estimate them from
the domain (similar to how we estimate $P(\omega_{i})$ in the Bayesian case).

$c$ can be optimized by looking at the clustering criteria (later in this
lecture).

Typically (as in Bayes), we select non-informative $P(\omega_{i})$ -- what
happens if we select unequal priors?

## Recap: Unequal Prior Values

![](../imgs/11_gaussian_mixture_1d_unequal_fused.pdf)

## Recap: Unequal Prior Values

![](../imgs/11_gaussian_mixture_1d_unequal.pdf)

## Recap: MLE: Estimating $\boldsymbol{\theta}_{i}$

For $\boldsymbol{\mu}_{i}$ and $\boldsymbol{\Sigma}_{i}$, we find the derivative
of our mixture likelihoods with respect to the parameters, set equal to zero,
and iteratively find the ***most likely*** parameter set that would give us
our training data.

If our mixture density is this:

$$p(\mathbf{x}|\boldsymbol{\theta})=\sum_{j=1}^{c}p(\mathbf{x}|\omega_{j},\boldsymbol{\theta}_{j})P(\omega_{j})$$

And we assume that each $p(\mathbf{x}|\omega_{j},\boldsymbol{\theta}_{j})$ is
Gaussian, then we can differentiate the natural logarithm with respect to each
parameter in turn and calculate the maximum likelihood estimate.

## Recap: MLE: Solving for $\boldsymbol{\Sigma}$

There's a lot of tricky math and derivations, but at the end of the day we can get an estimate for $\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})$:

\begin{align*}
\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}}) &= \frac{ |\boldsymbol{\hat{\Sigma}}_{i}|^{-\frac{1}{2}}\exp\left[-\frac{1}{2}
\tikz[baseline]{
        \node[fill=blue!20,anchor=base] (t1)
            {$(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})^{T}\hat{\boldsymbol{\Sigma}}_{i}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})$};
}
\right]\hat{P}(\omega_{i})}{\sum_{j=1}^{c}|\hat{\boldsymbol{\Sigma}}_{j}|^{-\frac{1}{2}}\exp\left[-\frac{1}{2}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{j})^{T}\hat{\boldsymbol{\Sigma}}_{j}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{j})\right]\hat{P}(\omega_{j})}\\
\end{align*}

With this long, ugly thing, we can estimate the likelihood that a point $\mathbf{x}_{k}$ belongs to $\omega_{i}$.

Simple explanation: if the squared Mahalanobis distance,
\tikz[baseline]{
    \node[fill=blue!20,anchor=base] (t1)
    {$(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})^{T}\hat{\boldsymbol{\Sigma}}_{i}^{-1}(\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i})$};
}, is small, then $\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})$ is large.

## Recap: $k$-Means Clustering

If we replace the Mahalanobis with the squared Euclidean distance
$\|\mathbf{x}_{k}-\hat{\boldsymbol{\mu}}_{i}\|^{2}$, we can find the mean
$\hat{\boldsymbol{\mu}}_{m}$ nearest to $\mathbf{x}_{k}$.

Thus we can approximate
$\hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}})$ as:

$$ \hat{P}(\omega_{i}|\mathbf{x}_{k},\hat{\boldsymbol{\theta}}) \simeq \left\{
			\begin{array}{l l}
			1 & \quad \textrm{if $i=m$} \\
			0 & \quad \textrm{otherwise.} \\
			\end{array} \right. $$

Then we can plug this into the equation we got before and solve for
$\hat{\boldsymbol{\mu}}_{1},\cdots,\hat{\boldsymbol{\mu}}_{c}$.

We can "initialize" by selecting $c$ class centroids at random from the
unlabeled data, and then iterating.

## Recap: $k$-Means In Sample Space

![$k$-Means Sample Space](../imgs/11_kmeans_iter1.pdf)

## Recap: $k$-Means In Sample Space

![$k$-Means Sample Space](../imgs/11_kmeans_iter2.pdf)

## Recap: $k$-Means In Sample Space

![$k$-Means Sample Space](../imgs/11_kmeans_iter3.pdf)

## Recap: $k$-Means Hill-climbing

![Stoachastic hill-climbing in $k$-means.](../imgs/11_kmeans_hill_climb.pdf)

## Recap: Comparing MLE and $k$-Means

\begincols

\column{0.5\linewidth}

![MLE Example](../imgs/11_2d_gaussian_mle_mixture.pdf){width=80%}

\column{0.5\linewidth}

![$k$-Means Example](../imgs/11_kmeans_hill_climb.pdf){width=80%}

\stopcols

Comparison of MLE and $k$-Means. Since the overlap between the components is
relatively small, we arrive at basically the same answers for $\mu_{1}$ and
$\mu_{2}$.

## Recap: $k$-Means Summary

$k$-Means is a staple of unsupervised clustering methods.

It is simple and fast.

When does it fail?

- If we are wrong about the number of clusters, we will converge on parameters
that don't "mean" anything.

- If the clusters are too close to one another, there may not be enough samples to
properly "ascend" to the true value of $\mu$.

Remember: ***everything*** is dependent on your features!

# Data Description and Clustering

## How Well Did We Cluster?

$k$-Means finds the parameters $\boldsymbol{\theta}$ of the underlying processes that control our samples.

***Clusters*** and ***Classes*** are NOT synonymous!

- Some classes are multi-modal: "Atypical" nuclei can be too small OR too large.
- Some features are bad: If $\boldsymbol{\theta}_{1} \approx \boldsymbol{\theta}_{2}$, the feature cannot distinguish $\omega_{1}$ and $\omega_{2}$.

## Examples of Misleading Parameters

![Distributions with equal $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$.](../imgs/11_misleading_clusters.pdf)

## Identifying the "True" Clusters

![How many clusters are in this data?](../imgs/11_unclear_clustering.pdf){width=40%}

## Clustering vs. Classification

\begincols

\column{0.5\linewidth}

![](../imgs/07_polynomial_disc_func.pdf){width=100%}

\column{0.5\linewidth}

- A class is a label that ***you*** decide on, and is not necessarily synonymous with clusters.
- If we don't have class labels, we ***cannot*** say that $\mathcal{R}_{1}$ on the left is the same decision region as $\mathcal{R}_{1}$ on the right.

\stopcols

## Illustration of Clustering Methods

![](../imgs/12_sklearn_clustering.png){width=80%}

## The First Major Issue: Defining Similarity

Unlabeled samples rely ***completely*** on their descriptive features.

We need to quantitatively say that samples in one cluster are "more similar" to each other than they are to samples in another cluster.

This is a ***similarity metric***.

## Distance as Inverse Similarity

***Distance*** may represent the inverse of ***similarity***: High distance = low similarity

In Euclidean space, $\mathbf{x}_{a}$ is ***more similar*** to $\mathbf{x}_{b}$ than $\mathbf{x}_{c}$ if $\|\mathbf{x}_{a}-\mathbf{x}_{b}\|^{2}<\|\mathbf{x}_{a}-\mathbf{x}_{c}\|^{2}$.

We can cluster points by setting a threshold $d_{0}$, where two points $\mathbf{x}_{a}$ and $\mathbf{x}_{b}$ belong to the same cluster if $\|\mathbf{x}_{a}-\mathbf{x}_{b}\|^{2} < d_{0}$.

## Examples of Distance Thresholding: High Threshold

\begincols

\column{0.3\linewidth}

![High Threshold Cluster](../imgs/11_cluster_high_thresh.pdf){width=100%}

\column{0.3\linewidth}

![Medium Threshold Cluster](../imgs/11_cluster_mid_thresh.pdf){width=100%}

\column{0.3\linewidth}

![Low Threshold Cluster](../imgs/11_cluster_low_thresh.pdf){width=100%}

\stopcols

## Choosing a Similarity Metric

***Discrete Metric:***

$$ d(\mathbf{x,y}) = \begin{cases}
1 & \text{if } \mathbf{x\neq y}\\
0 & \text{if } \mathbf{x=y}
\end{cases}$$

***Euclidean Metric:***

$$ d(\mathbf{x,y}) = \sqrt{(x_{1}-y_{1})^{2}+\cdots+(x_{d}-y_{d})^{2}} $$

***Taxicab Metric:***

$$ d(\mathbf{x,y}) = \sum_{i=1}^{d}\abs{x_{i}-y_{i}} $$

## Illustration of Distances

![](../imgs/05_distancemetrics.png){width=45%}

## Choosing a Similarity Metric: Invariance to Transforms


Euclidean distance is a good first choice:

$$ D(\mathbf{x},\mathbf{x}^{\prime}) = \sqrt{\sum_{k=1}^{d}(x_{k} - x_{k}^{\prime})^{2}} $$

- ***Isotropy***: distances in all dimensions must be equivalent.
- ***Smoothness***: distances in all feature ranges are equivalent.
- ***Linearity***: data should be observed throughout the feature space.

Euclidean distance is robust to rigid transformations of the feature space
(rotation and translation), but are NOT robust to ***arbitrary linear
transformations*** which distort distance relationships.

## Rotation's Effect on Cluster Groupings

![Cluster Scaling](../imgs/11_cluster_scaling.pdf){width=40%}

## Alternative Distance Metrics

We generalize from the Euclidean to the ***Minkowski*** metric:

$$ D(\mathbf{x},\mathbf{x}^{\prime})=\left( \sum_{k=1}^{d}|x_{k}-x_{k}^{\prime}|^{q}\right)^{\frac{1}{q}} $$

If $q=2$, then this is Euclidean; if $q=1$, this is the ***city block*** metric.

You can also look at Mahalanobis distance,
$(\mathbf{x}-\mathbf{x}^{\prime})^{T}\Sigma^{-1}(\mathbf{x}-\mathbf{x}^{\prime})$,
which depends on the data itself to define the distance.

We can even abandon distance and define an arbitrary symmetric function
$s(\mathbf{x},\mathbf{x}^{\prime})$ as some measurement of "similarity", e.g.
the angle between two vectors:

$$ s(\mathbf{x},\mathbf{x}^{\prime})=\frac{\mathbf{x}^{t}\mathbf{x}^{\prime}}{\|\mathbf{x}\|\|\mathbf{x}^{\prime}\|} $$

# Criterion Functions

## The Second Major Issue: Evaluation

Suppose we've got our unlabeled training set
$\mathcal{D}=\{\mathbf{x}_{1},\cdots,\mathbf{x}_{n}\}$.

We want to divide this into exactly $c$ disjoint subsets
$\mathcal{D}_{1},\cdots,\mathcal{D}_{c}$, which represent cluster memberships.

All samples in $\mathcal{D}_{a}$ should be more alike to each other than 
to those in $\mathcal{D}_{b}$, $a\neq b$.

Let's define ourselves a ***criterion function*** that is simply used to
evaluate how good a given partition is; our task will then be to find a
partition that "extremizes" our function.

## Sum-of-Squared-Error Criterion

The ***sum-of-squared-error*** criterion is defined as the difference between
the samples and the mean of the assigned cluster:

$$ J_{e}=\sum_{i=1}^{c}\sum_{\mathbf{x}\in\mathcal{D}_{i}}\|\mathbf{x}-\mathbf{m}_{i}\|^{2}$$

where:

$$ \mathbf{m}_{i}=\frac{1}{n_{i}}\sum_{\mathbf{x}\in\mathcal{D}_{i}}\mathbf{x} $$

The optimal partition is the one that minimizes $J_{e}$.

The idea is that it measures the error incurred if all of the samples in
$\mathcal{D}_{i}$ were represented by the cluster center $\mathbf{m}_{i}$.

## Sum-of-Squared-Error Criterion

This works very well if we have clusters that are: 

- Equally evenly spread
- Far apart from each other

If one cluster is spread out and one is small, this criterion may end up trying
to "even out" the differences by selecting a non-optimal partition.

## Sum-of-Squared-Error Criterion

![Sum-of-Squared Error Failure](../imgs/11_SSE_clustering.pdf){width=35%}

## Related Minimum Variance Criterion

We can wrangle the mean vectors out of the expression for Sum-of-Squared-Error
and get the equivalent expression:

$$ J_{e}=\frac{1}{2}\sum_{i=1}^{c}n_{i}\bar{s}_{i}$$

Where $\bar{s}_{i}$ can be formed into whatever kind of similarity function we
want:

\begin{align*}
\bar{s}_{i} &= \frac{1}{n^{2}_{i}}\sum_{\mathbf{x}\in\mathcal{D}_{i}}\sum_{\mathbf{x}^{\prime}\in\mathcal{D}_{i}}\|\mathbf{x}-\mathbf{x}^{\prime}\|^{2}\\
\bar{s}_{i} &= \frac{1}{n_{i}^{2}}\sum_{\mathbf{x}\in\mathcal{D}_{i}}\sum_{\mathbf{x}^{\prime}\in\mathcal{D}_{i}}s(\mathbf{x},\mathbf{x}^{\prime})\\
\bar{s}_{i} &= \min_{\mathbf{x},\mathbf{x}^{\prime}\in\mathcal{D}_{i}}s(\mathbf{x},\mathbf{x}^{\prime}) \\
\end{align*}

## Scatter Criteria

We can calculate a bunch of values that relate to the "scatter" of the data in
each cluster.

We'll come back to these when we discuss PCA, but here is a list of six
quantities that are useful to know.

## Mean Vectors and Scatter Matrices

\begin{table}
    \begin{tabular}{lr}
    \toprule
    \textbf{\alert{Name}} & \textbf{\alert{Equation}} \\
    \toprule
    Mean vector for the $i$th cluster & $\mathbf{m}_{i}=\frac{1}{n_{i}}\sum_{\mathbf{x}\in\mathcal{D}_{i}}\mathbf{x}$ \\
    \midrule
    Total mean vector & $\mathbf{m}=\frac{1}{n}\sum_{\mathcal{D}}\mathbf{x}=\frac{1}{n}\sum_{i=1}^{c}n_{i}\mathbf{m}_{i}$ \\
    \midrule
    Scatter matrix for the $i$th cluster & $\mathbf{S}_{i}=\sum_{\mathbf{x}\in\mathcal{D}_{i}}(\mathbf{x}-\mathbf{m}_{i})(\mathbf{x}-\mathbf{m}_{i})^{T}$ \\
    \midrule
    Within-cluster scatter matrix & $\mathbf{S}_{W}=\sum_{i=1}^{c}\mathbf{S}_{i}$ \\
    \midrule
    Between-cluster scatter matrix & $\mathbf{S}_{B}=\sum_{i=1}^{c}n_{i}(\mathbf{m}_{i}-\mathbf{m})(\mathbf{m}_{i}-\mathbf{m})^{T}$ \\
    \midrule
    Total scatter matrix & $\mathbf{S}_{T}=\sum_{\mathbf{x}\in\mathcal{D}}(\mathbf{x}-\mathbf{m})(\mathbf{x}-\mathbf{m})^{T}$ \\
        & $\mathbf{S}_{T} = \mathbf{S}_{W} + \mathbf{S}_{B}$ \\
    \bottomrule
    \end{tabular}
\end{table}

## Trace Criterion

We need a scalar measure of the "size" of the scatter matrix, so we know if
one set of points is more or less scattered than another.

One measure is the ***trace*** of the within-cluster scatter matrix, which is
the sum of its diagonal elements.

Minimizing this turns out to be the sum-of-squared-error criterion:

$$ \Tr{\mathbf{S}_{W}} = \sum_{i=1}^{c}\Tr{\mathbf{S}_{i}}=\sum_{i=1}^{c}\sum_{\mathbf{x}\in\mathcal{D}_{i}}\|\mathbf{x}-\mathbf{m}_{i}\|^{2}=J_{e} $$

Note that $\Tr{\mathbf{S}_{T}}$ is independent of the partitioning (i.e. it
doesn't change), and this is equal to $\Tr{\mathbf{S}_{W}} +
\Tr{\mathbf{S}_{B}}$.

Therefore, minimizing $\Tr{\mathbf{S}_{W}}$ also maximizes
$\Tr{\mathbf{S}_{B}}$.

## Determinant Criterion

The determinant measures the square of the scattering volume.

$\mathbf{S}_{B}$ is singular if the number of clusters is less than or equal to
the number of dimensions, so we don't want to use it for our criterion function.

If we assume that $\mathbf{S}_{W}$ is nonsingular, we can have:

$$ J_{d} = \abs{\mathbf{S}_{W}} = \abs{\sum_{i=1}^{c}\mathbf{S}_{i}} $$

The trace and determinant criteria do not need to be the same, although they
often are.

## Invariant Criterion

We may want to look at the ratio of between-cluster and within-cluster matrices,
$\mathbf{S}_{W}^{-1}\mathbf{S}_{B}$, as an optimal partition would ideally have
small clusters that are also spread far apart.

This can be done by finding partitions that result in the eigenvalues of
$\mathbf{S}_{W}^{-1}\mathbf{S}_{B}$ are large.

Since the trace of a matrix is the sum of its eigenvalues, we can maximize a
criterion function as:

$$ \Tr{\mathbf{S}_{W}^{-1}\mathbf{S}_{B}} = \sum_{i=1}^{d}\lambda_{i} $$

## Invariant Criterion

By using the relation $\mathbf{S}_{T} = \mathbf{S}_{W} + \mathbf{S}_{B}$, we can
derive the following invariant criterion functions:

$$ J_{f}=\Tr{\mathbf{S}_{T}^{-1}\mathbf{S}_{W}}=\sum_{i=1}^{d}\frac{1}{1+\lambda_{i}}$$

$$ \frac{|\mathbf{S}_{W}|}{|\mathbf{S}_{T}|}=\prod_{i=1}^{d}\frac{1}{1+\lambda_{i}} $$

## Comparing Criterion Examples

![Unknown Number of Clusters](../imgs/11_unknown_clusters.png){width=35%}

## Comparing Criterion Examples: Two Clusters

\begincols

\column{0.3\linewidth}

![$J_{e}$, $c=2$](../imgs/11_unknown_clusters_je_c2.png){width=100%}

\column{0.3\linewidth}

![$J_{d}$, $c=2$](../imgs/11_unknown_clusters_jd_c2.png){width=100%}

\column{0.3\linewidth}

![$J_{f}$, $c=2$](../imgs/11_unknown_clusters_jf_c2.png){width=100%}

\stopcols

## Comparing Criterion Examples: Three Clusters

\begincols

\column{0.3\linewidth}

![$J_{e}$, $c=3$](../imgs/11_unknown_clusters_je_c3.png){width=100%}

\column{0.3\linewidth}

![$J_{d}$, $c=3$](../imgs/11_unknown_clusters_jd_c3.png){width=100%}

\column{0.3\linewidth}

![$J_{f}$, $c=3$](../imgs/11_unknown_clusters_jf_c3.png){width=100%}

\stopcols

# Clustering Example (Matlab)

# Hierarchical Clustering

## Introduction to Hierarchical Clustering

So far, we've been interested in ***disjoint subsets***; that is,
$\mathcal{D}_{a}\cup\mathcal{D}_{b}=\emptyset$.

However, remember that our classes are things that ***we*** define,
and some objects can be associated with more than one class.

- kingdom = animal
- phylum = Chordata
- ...
- family = Salmonidae
- genus = Oncorhynchus
- species = Oncorhynchus kisutch

## Hierarchical Definitions

Data description is ***flat*** in disjoint partitioning.

In hierarchical clustering, we have a ***sequence*** of partitions. The approach is:

1. Partition the data into $n$ clusters (each data point is its own cluster).
2. Merge clusters one at a time based on some criteria until we only have $1$ cluster.

Level $k$ in the sequence is when $c=n-k+1$.

We select a minimum number of clusters, $c$, to stop the algorithm. If we set $c=1$, we get a dendrogram.

In this setup, every two samples $\mathbf{x}$ and $\mathbf{x}^{\prime}$ will be grouped eventually.

## Dendrogram Tree

\begincols

\column{0.5\linewidth}

![Dendrogram Tree](../imgs/12_dendrogram_tree.png){width=100%}

\column{0.5\linewidth}

Cluster similarity is visualized as the distance from one level to the next.

Large distances imply here is a natural clustering at that level.

We can visualize this setup as text brackets:

$\{\{\mathbf{x}_{1},\{\mathbf{x}_{2},\mathbf{x}_{3}\}\},\{\{\{\mathbf{x}_{4},\mathbf{x}_{5}\},\{\mathbf{x}_{6},\mathbf{x}_{7}\}\},\mathbf{x}_{8}\}\}$.

\stopcols

## Alternative Representation: Venn Diagrams

\begincols

\column{0.5\linewidth}

![Venn Diagram](../imgs/12_venn_diagram.png){width=60%}

\column{0.5\linewidth}

Venn diagrams can also be used to see these relationships.

This is less quantitative, but represents how the different samples are grouped in the feature space.

\stopcols

## Two Approches to Hierarchical Clustering

There are two related approaches to hierarchical clustering:

- ***Agglomerative***: bottom-up, starting with $n$ singleton clusters and grouping up.
- ***Divisive***: top-down, start with 1 all-encompassing cluster and then splitting.

We will focus on the former, which typically has a simpler (but oft-repeated) calculation at each hierarchical level.

## Agglomerative Clustering

\begin{algorithm}[H]
    \caption{Agglomerative Hierarchical Clustering}
    \emph{begin initialize $c,\hat{c}\leftarrow n,\mathcal{D}_{i}\leftarrow \{\mathbf{x}_{i}\},i=1,\cdots,n$}\;
    \Repeat{$c=\hat{c}$}{
        $\hat{c}\leftarrow\hat{c}-1$\;
        find nearest clusters, $\mathcal{D}_{i}$ and $\mathcal{D}_{j}$\;
        merge $\mathcal{D}_{i}$ and $\mathcal{D}_{j}$\;
    }
    \KwRet{$c$ clusters}
\end{algorithm}


## Agglomerative Procedure

Define measurements of cluster relationships:

\begin{align*}
    d_{min}(\mathcal{D}_{i},\mathcal{D}_{j})& =\min_{\substack{\mathbf{x}\in\mathcal{D}_{i} \\ \mathbf{x}^{\prime}\in\mathcal{D}_{j}}}\|\mathbf{x}-\mathbf{x}^{\prime}\| \\
    d_{max}(\mathcal{D}_{i},\mathcal{D}_{j})& =\max_{\substack{\mathbf{x}\in\mathcal{D}_{i} \\ \mathbf{x}^{\prime}\in\mathcal{D}_{j}}}\|\mathbf{x}-\mathbf{x}^{\prime}\| \\
    d_{avg}(\mathcal{D}_{i},\mathcal{D}_{j})& =\frac{1}{n_{i}n_{j}}\sum_{\mathbf{x}\in\mathcal{D}_{i}}\sum_{\mathbf{x}^{\prime}\in\mathcal{D}_{j}}\|\mathbf{x}-\mathbf{x}^{\prime}\| \\
    d_{mean}(\mathcal{D}_{i},\mathcal{D}_{j})& =\|\mathbf{m}_{i}-\mathbf{m}_{j}\| \\
\end{align*}

## Computational Complexity of Agglomerative Approaches

Suppose we have $n$ patterns in $d$-dimensional space, and we want $c$ clusters with $d_{min}$.

We'll need to calculate $n(n-1)$ distances, each of which is an $O(d)$ calculation.

For the first step, the complexity is $O(n(n-1)(d+1))=O(n^{2}d)$.

For subsequent steps we are merging points, so the number of calculations goes down. 

For an arbitrary step, we just need to calculate $n(n-1)-\hat{c}$ ``unused'' distances in the list.

The full time complexity is thus $O(cn^{2}d)$.

## Nearest-Neighbor Algorithm

When the distance between clusters is $d_{min}(\cdot,\cdot)$, it's called the
***nearest-neighbor*** clustering algorithm.

Imagine each $\mathbf{x}$ is a node in a graph, with edges connecting nodes in
the same cluster, $\mathcal{D}_{i}$. The merging of $\mathcal{D}_{i}$ and
$\mathcal{D}_{j}$ corresponds to adding an edge between the nearest pair of
nodes in $\mathcal{D}_{i}$ and $\mathcal{D}_{j}$.

Because of the way we add edges, we won't have any closed loops or circuits.
Therefore it generates a tree -- this is a ***spanning tree*** is when all the
subsets are linked.

Moreover, this algorithm will create a ***minimum spanning tree***: the sum of
edge lengths is lower than for all spanning trees.

## Example of Chaining Effect

\begincols

\column{0.5\linewidth}

![Nearest-Neighbor Clustering](../imgs/12_chaining_before.png){width=50%}

\column{0.5\linewidth}

![Single Sample Added](../imgs/12_chaining_after.png){width=50%}

\stopcols

## Farthest-Neighbor Algorithm

In contrast, using $d_{max}(\cdot,\cdot)$ results in the ***farthest-neighbor*** algorithm.

This discourages the growth of elongated clusters; the distance between different clusters is determined by the most distant nodes in those clusters.

The diameter of a partition is the largest diameter for clusters in the partition, so each iteration increases the diameter of the partition by a minimal amount.

This works when clusters are far apart and fairly compact, but if the clusters are elongated in size then the groupings can be meaningless.

## Example of Farthest-Neighbor Clustering

\begincols

\column{0.5\linewidth}

![Large Farthest Distance](../imgs/12_farthest_neighbor_large.png){width=50%}

\column{0.5\linewidth}

![Small Farthest Distance](../imgs/12_farthest_neighbor_small.png){width=50%}

\stopcols

## Stepwise-Optimal Clustering

Previously we looked at the "nearest" clusters and merged them.

As we saw in the discussion about clustering evaluation, we need a way to define the "nearest" (i.e. most-similar) samples.

To do this, we can replace "nearest" with a criterion function which changes the least with each merger.

This gives us a new algorithm, and a new "distance" whose minimum value indicates the optimal merger:

$$d_{e}(\mathcal{D}_{i},\mathcal{D}_{j})=\sqrt{\frac{n_{i}n_{j}}{n_{i}+n_{j}}}\|\mathbf{m}_{i}-\mathbf{m}_{j}\| $$

## Stepwise-Optimal Clustering

\begin{algorithm}[H]
    \caption{Stepwise Optimal Hierarchical Clustering}
    \emph{begin initialize $c,\hat{c}\leftarrow n,\mathcal{D}_{i}\leftarrow \{\mathbf{x}_{i}\},i=1,\cdots,n$}\;
    \Repeat{$c=\hat{c}$}{
        $\hat{c}\leftarrow\hat{c}-1$\;
        find clusters $\mathcal{D}_{i}$ and $\mathcal{D}_{j}$ whose merger changes the criterion the least\;
        merge $\mathcal{D}_{i}$ and $\mathcal{D}_{j}$\;
    }
    \KwRet{$c$ clusters}
\end{algorithm}

# Principal Component Analysis

## Dimensionality Reduction

***Component analysis*** is used both for dimensionality reduction and clustering.

We will start with linear techniques, and these will lend naturally to nonlinear approaches later.

There are two common linear approaches to component analysis: Principal Component Analysis and Multiple Discriminant Analysis.

PCA seeks a projection to ***represent*** the data, while MDA seeks a projection to ***separate*** the data.

## Principal Component Analysis: Basic Approach

We have a $d$-dimensional mean $\boldsymbol{\mu}$ and a $d\times d$ covariance $\boldsymbol{\Sigma}$ for our data set $\mathcal{D}$.

Compute eigenvectors and eigenvalues of $\boldsymbol{\Sigma}$ and sort columns by decreasing eigenvalues.

Choose the $k$ largest eigenvalues, and form a $d\times k$ matrix $\mathbf{A}$ of the $k$ associated eigenvectors.

The data is then projected onto the $k$-dimensional subspace according to:

$$ \mathbf{x}^{\prime}=\mathbf{F}_{1}(\mathbf{x})=\mathbf{A}^{T}(\mathbf{x}-\boldsymbol{\mu}) $$

## PCA in Detail

Imagine we want to represent all samples in $\mathcal{D}$ with a single vector $\mathbf{x}_{0}$.

$\mathbf{x}_{0}$ should minimize the sum of the squared distances between itself and each sample:

$$ J_{0}(\mathbf{x}_{0})=\sum_{k=1}^{n}\|\mathbf{x}_{0}-\mathbf{x}_{k}\|^{2} $$

We can assume that this criterion function will be minimized by the sample mean:

$$ \mathbf{x}_{0}=\mathbf{m}=\frac{1}{n}\sum_{k=1}^{n}\mathbf{x}_{k} $$

## Dimensionality of Our Representation

The sample mean is a "zero-dimensional" representation of the data, and can correspond to many different sample distributions, as we've seen.

## Similar Means Can Represent Different Distributions

![Distributions with equal $\boldsymbol{\mu}$ and $\boldsymbol{\Sigma}$.](../imgs/11_misleading_clusters.pdf)

## Dimensionality of Our Representation

Let's say we want to "project" the data onto a ***one***-dimensional representation (a line) running through the sample mean.

If $\mathbf{e}$ is a unit vector in the direction of the desired line, each sample can be written as:

$$ \mathbf{x}_{k}=\mathbf{m}+a_{k}\mathbf{e} $$

... where $a_{k}$ represents the distance of sample point $\mathbf{x}_{k}$ from the mean $\mathbf{m}$.

## Obtaining the Optimal Set of Coefficients

We can get optimal coefficients $a_{1},\cdots,a_{n}$ by minimizing the criterion:

\begin{align*}
J_{1}(a_{1},\cdots,a_{n},\mathbf{e})&=\sum_{k=1}^{n}\|(\mathbf{m}+a_{k}\mathbf{e})-\mathbf{x}_{k}\|^{2}\\
&=\sum_{k=1}^{n}\|a_{k}\mathbf{e}-(\mathbf{x}_{k}-\mathbf{m})\|^{2}\\
&=\sum_{k=1}^{n}a_{k}^{2}\|\mathbf{e}\|^{2}-2\sum_{k=1}^{n}a_{k}\mathbf{e}^{T}(\mathbf{x}_{k}-\mathbf{m})+\sum_{k=1}^{n}\|\mathbf{x}_{k}-\mathbf{m}\|^{2} \\
\end{align*}

Get the derivative with respect to $a_{k}$, set it to zero, and solve:

$$ a_{k}=\mathbf{e}^{T}(\mathbf{x}_{k}-\mathbf{m}) $$

## Return to the Scatter Matrix

We have the distances to the line, but what direction is it in?

It passes through the mean in infinitely many directions.

Recall our scatter matrices that we've been using:

$$ \mathbf{S}=\sum_{k=1}^{n}(\mathbf{x}_{k}-\mathbf{m})(\mathbf{x}_{k}-\mathbf{m})^{T} $$

If we plug our equation for $a_{k}$ into the criterion function $J_{1}$, we get:

$$ J_{1}(\mathbf{e})=-\mathbf{e}^{T}\mathbf{S}\mathbf{e}+\sum_{k=1}^{n}\|\mathbf{x}_{k}-\mathbf{m}\|^{2} $$

## Solving for the Direction of $\mathbf{e}$

$$ J_{1}(\mathbf{e})=-\mathbf{e}^{T}\mathbf{S}\mathbf{e}+\sum_{k=1}^{n}\|\mathbf{x}_{k}-\mathbf{m}\|^{2} $$

We can solve for the optimal $\mathbf{e}$ using the method of Lagrange multipliers, subject to $\|\mathbf{e}\|=1$:

$$ \underbrace{u}_{L(\mathbf{e},\lambda)}=\underbrace{\mathbf{e}^{T}\mathbf{S}\mathbf{e}}_{f(\mathbf{e})}-\lambda\underbrace{(\mathbf{e}^{T}\mathbf{e}-1)}_{g(\mathbf{e})} $$

Differentiate with respect to $\mathbf{e}$, set to zero, and we get $\mathbf{S}\mathbf{e}=\lambda\mathbf{e}$.

Since $\mathbf{e}^{T}\mathbf{S}\mathbf{e}=\lambda\mathbf{e}^{T}\mathbf{e}=\lambda$, maximizing $\mathbf{e}^{T}\mathbf{S}\mathbf{e}$ involves selecting the eigenvector corresponding to the largest eigenvalue of $\mathbf{S}$.

## Extending to Multiple Dimensions

One-dimensional representations are kind of boring.

If we want to obtain a $d^{\prime}$-dimensional representation, where
$d^{\prime}\leq d$, we can estimate $\mathbf{x}$ as:

$$ \mathbf{x}=\mathbf{m}+\sum_{i=1}^{d^{\prime}}a_{i}\mathbf{e}_{i} $$

This makes our criterion function:

$$ J_{d^{\prime}}=\sum_{k=1}^{n}\left\|\left(\mathbf{m}+\sum_{i=1}^{d^{\prime}}a_{ki}\mathbf{e}_{i}\right)-\mathbf{x}_{k}\right\|^{2} $$

## Why It's Called PCA

That criterion function is minimized when
$\mathbf{e}_{1},\cdots,\mathbf{e}_{d^{\prime}}$ are the $d^{\prime}$
eigenvectors of the scatter matrix with the largest eigenvalues.

These are also orthogonal, since $\mathbf{S}$ is real and symmetric.

In linear algebra, they form a ***basis*** for representing
$\mathbf{x}$.

The coefficients $a_{i}$ are the components of $\mathbf{x}$ in that basis, and
are called the ***principal components***.

# PCA Example (Matlab)

# Parting Words

## Drawbacks to PCA

PCA is a fairly simple approach, but it does the job in linear spaces.

PCA seeks an ***optimal projection***: that is, it tries to represent the variation in the data.

If our subspace is non-linear, we need additional approaches to properly separate out our data (as we saw in the criterion functions).

If we know that our data is coming from multiple sources, then we may want to seek an ***independent component*** set that splits apart those signals.

## Next Class

We will wrap up our discussion of clustering with ***dimensionality reduction***, which is a way to visualize and handle data that exists in too high of a dimension.

We will also cover ***nonlinear*** methods of reduction, which do not rely on a Euclidean space distribution of the samples.

After that, we will begin a discussion of deep learning and artificial intelligence with our first lecture on neural networks.

