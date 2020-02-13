---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade
slide-level: 2

author: Scott Doyle
contact: scottdoy@buffalo.edu
title: Nonmetric Methods<br/>Decision Trees
subtitle: Machine Learning for Biomedical Data
date: 2020-02-13
---

# 
## Definitions of Metrics

## What is a Metric?

Let's say we have a set of points called \$\\mathcal{X}\$.

<p class="fragment">We define a function called a ***distance function***:</p>

<p class="fragment">\$\$ d: \\mathcal{X} \\times \\mathcal{X} \\mapsto [0,\\infty) \$\$</p>

<p class="fragment">
This function says that you can take two points in \$\\mathcal{X}\$ and map them to
a point on the number line from 0 to an arbitrarily large number.
</p>

## Metric Conditions

We call \$d\$ a ***metric*** if, for all points \$\\mathbf{x,y,z}\\in\\mathcal{X}\$,
ALL of the following conditions are satisified:

<div class="fragment">**Non-negativity**: Distances cannot be less than
zero.

\$\$d(\\mathbf{x,y}) \\geq 0\$\$</div>

<div class="fragment">**Identity of Indiscernibles**: If two points have a distance of 0, then they are (effectively) the same point.

\$\$d(\\mathbf{x,y})=0\\iff\\mathbf{x}=\\mathbf{y}\$\$</div>

## Metric Conditions

**Symmetry**: The distance from \$\\mathbf{x}\$ to \$\\mathbf{y}\$ should be
the same as the distance from \$\\mathbf{y}\$ to \$\\mathbf{x}\$

\$\$d(\\mathbf{x,y}) = d(\\mathbf{y,x})\$\$

<div class="fragment">
**Triangle Inequality**: A straight path between two points \$\\mathbf{x}\$
and \$\\mathbf{z}\$ should be the shortest path, compared to a "detour" through
point \$\\mathbf{y}\$)

\$\$d(\\mathbf{x,z}) \\leq d(\\mathbf{x,y}) + d(\\mathbf{y,z})\$\$</div>



## Discrete Metric

\$\$ d(\\mathbf{x,y}) = \\begin{cases}
1 & \\text{if } \\mathbf{x\\neq y}\\\\
0 & \\text{if } \\mathbf{x=y}
\\end{cases}\$\$

## Euclidean Metric

\$ d(\\mathbf{x,y}) = \\sqrt{(x\_{1}-y\_{1})\^{2}+\\cdots+(x\_{d}-y\_{d})\^{2}} \$

## Taxicab Metric:

\$\$ d(\\mathbf{x,y}) = \\sum\_{i=1}\^{d}|x\_{i}-y\_{i}| \$\$

## Illustration of Distances

![](img/distancemetrics.png){width=45%}

## Metric vs. Nonmetric Features

**Metric features** are those which can be:


<ul>
<li class="fragment">Ordered -- One is numerically higher, bigger, or greater than another</li>
<li class="fragment">Distanced -- You can calculate a distance metric between two objects</li>
</ul>

<p class="fragment">
**Nonmetric** or **nominal** features cannot be ordered, and there is no
"distance" between them.
</p>

## Metric vs. Nonmetric Features

<div class="l-double">
<div>
![Walrus](img/walrus.jpg){width=80%}
</div>
<div>
![Seal](img/seal.jpg){width=80%}
</div>
</div>

<p class="fragment">
Example: Animal teeth can be small and fine, they can exist in one line or
multiple rows, they can be tusks, and some animals have no teeth.
</p>

<p class="fragment">There is no sense of distance between "tusks" and "beaks".</p>

## Metric vs. Nonmetric Features

Objects in nonmetric features can be represented by **property d-tuples**. 

<p class="fragment">
For example, an animal might
be a \$d\$-tuple characterizing the animal's teeth: (size, type, use). 
</p>

<div class="fragment txt-left">
Thus, to
describe two different animals:

- walrus = (large, tusk, dominance)
- seal = (small, cusped, chewing)
</div>

<p class="fragment">How do we classify a set of objects represented by tuples?</p>

# 
## Decision Trees

## Elaborate if/than/else Questions

For each object, ask a series of questions to try and identify it.

<p class="fragment">These can be "yes/no", "true/false", or "value(property)\$\\in\${values}" questions.</p>

<ul>
<li class="fragment">How do we ***choose*** these questions?</li>
<li class="fragment">What is the ***smallest number*** of questions we can ask to reach a conclusion?</li>
<li class="fragment">When can we ***stop*** asking questions; i.e. when have we reached the true label?</li>
<li class="fragment">What is the correct ***order*** in which to ask these questions?</li>
</ul>

## Decision Tree Example

![](img/decisiontree.svg){width=70%}

## Decision Tree Terminology

Links or **branches** correspond to specific values from the node.

<p class="fragment">
Starting from the **root node**, the link is followed to the **descendent
node** based on its value.
</p>

<p class="fragment">Links must be **distinct** and **exhaustive**:</p>

<ul>
<li class="fragment">Only one link can be followed.</li>
<li class="fragment">There must be a link for each possible value</li>
</ul>

<p class="fragment">
When a **leaf** node is reached, there are no more questions and the object is
classified.
</p>

## Decision Tree Example and Terminology

<div class="l-double">
<div>
![](img/decisiontree.svg){width=100%}
</div>
<div>

Objects can be represented by their property-value sets:

\$\\mathbf{x}=\${"sweet", "yellow", "thin", "medium"}


<p class="fragment">**Banana:** is (yellow AND thin), which is all we need to find a leaf for \$\\mathbf{x}\$</p>

</div>
</div>

## Decision Tree Example and Terminology

<div class="l-double">

<div>


![](img/decisiontree.svg){width=100%}

</div>
<div>


Categories are defined by node paths:

<ul>
<li class="fragment">Apples are: (green AND medium) OR (red AND medium)</li>
<li class="fragment">Apples are also: (medium AND NOT yellow)</li>
</ul>

<p class="fragment">
Easily incorporates knowledge from human experts (assuming the problem is simple
and training set is small).
</p>

</div>
</div>

# 
## Training: Classification and Regression Trees (CART)

## CART Training: Preliminaries

Assume we have training data \$\\mathcal{D}\$ and a pre-determined set of **nodes** or features.

<p class="fragment">Each node splits the training set into smaller subsets: \$\\mathcal{D} = \\{\\mathcal{D}\_{1}, \\mathcal{D}\_{2}\\}\$.</p>

<p class="fragment">If all the subsets have the same label, then we can stop splitting because the node is **pure**; otherwise, the node is **impure**.</p>

## CART

This is part of the CART approach (Classification And Regression Trees).

<p class="fragment">At any node, you can either:</p>

<ul>
<li class="fragment">Declare this node a leaf and stop splitting (Leads to an imperfect result if  the node is impure); or</li>
<li class="fragment">Continue with a different property and grow the tree further.</li>
</ul>

## CART

Six questions arise from this approach:

<ol>
<li class="fragment">How many branches do we extend from a node?</li>
<li class="fragment">Which property is tested at a node?</li>
<li class="fragment">When do we declare a node to a leaf?</li>
<li class="fragment">Can we “prune” a tree to make it smaller?</li>
<li class="fragment">If a node is declared a leaf, and the leaf is impure, what category label should be assigned?</li>
<li class="fragment">How can we handle missing data?</li>
</ol>

# 
## Number of Branches

## Number of Splits at Each Node

The number of branches descending from a node is its **branching factor**, \$B\$.

<p class="fragment">\$B\$ can be selected by the designer, but any node with \$B>2\$ can be replaced with a series of nodes with \$B=2\$ to make it a **binary tree**.</p>

<p class="fragment">We will focus on binary trees, which are easier to train and understand. However, \$B>2\$ may be more computationally efficient in practice.</p>

## Non-Binary Splits (\$B\\geq 2\$)

![Non-Binary Decision Tree](img/decisiontree_nonbin.svg){width=60%}

## Binary Splits (\$B=2\$)

![Binary Decision Tree](img/decisiontree_bin.svg){width=60%}

# 
## Query Selection

## Query Selection and Node Impurity

We call a split a **query**, denoted \$T\$. We want to select our queries such that:

<ul>
<li class="fragment">The overall tree is simple and compact (minimize the overall number of nodes); and</li>
<li class="fragment">The data reaching the immediate descendent nodes should be as ***pure*** as possible.</li>
</ul>

<p class="fragment">First, ***simplicity***: We should use the fewest number of possible queries.</p>

## Complex Tree (Not Simple or Compact)

![Complex Decision Tree](img/decisiontree_complex.svg){width=60%}

## Simple Tree (Simple and Compact)

![Simple Decision Tree](img/decisiontree_simple.svg){width=60%}

## Node Impurity Definition

When deciding whether to split a node, we should quantify node "impurity".

<p class="fragment">Define a function \$i(N)\$, which returns some measure of impurity at node \$N\$:</p>

<ul>
<li class="fragment">\$i(N)=0\$: all samples that reach \$N\$ have the same label</li>
<li class="fragment">\$i(N)>>0\$: all classes are equally mixed</li>
</ul>

## Measures of Node Impurity: Entropy

The most common impurity is **Entropy / Information Impurity:**

<p class="fragment">\$\$ i(N) = - \\sum\_{j}\\widehat{P}(\\omega\_{j})\\log\_{2}{\\widehat{P}(\\omega\_{j})} \$\$</p>

<p class="fragment">Here, $\\widehat{P}(\\omega\_{j})\$ refers to the fraction of training patterns in category \$\\omega\_{j}\$ at node \$N\$.</p>

## Max and Min of Entropy Impurity

\$\$ i(N) = - \\sum\_{j}\\widehat{P}(\\omega\_{j})\\log\_{2}{\\widehat{P}(\\omega\_{j})} \$\$

<p class="fragment">Given this, when is \$i(N)\$ at a maximum, and when is it at a minimum?</p>

<ul>
<li class="fragment">Entropy is a measure of ***randomness*** in a sequence.</li>
<li class="fragment">Coin flips are entirely random; the Fibonacci sequence is not random.</li>
</ul>

<p class="fragment">
What are the values of \$\\widehat{P}(\\omega\_{1})\$ and
\$\\widehat{P}(\\omega\_{2})\$, and when are they **most informative** or
**least informative**?
</p>

## When is Entropy Impurity Maximum?

If there are two classes, and node \$N\$ contains **equal numbers** of points
from each class, then \$\\widehat{P}(\\omega\_{1}) = \\widehat{P}(\\omega\_{2}) = 0.5\$

<p class="fragment">
\\begin{align}
i(N) &= - \\sum\_{j}\\widehat{P}(\\omega\_{j})\\log\_{2}{\\widehat{P}(\\omega\_{j})}\\\\
&= -\\left[ \\widehat{P}(\\omega\_{1})\\log\_{2}{\\widehat{P}(\\omega\_{1})} + \\widehat{P}(\\omega\_{2})\\log\_{2}{\\widehat{P}(\\omega\_{2})}\\right]\\\\
&= -\\left[ 0.5\\log\_{2}{0.5} + 0.5\\log\_{2}{0.5} \\right]\\\\
&= -\\left[ 0.5(-1) + 0.5(-1) \\right]\\\\
&= 1
\\end{align}
</p>

## When is Entropy Impurity Minimum?

If there are two classes, and node \$N\$ contains **only points from one
class**, \$\\omega\_{1}\$, then \$\\widehat{P}(\\omega\_{1}) = 1.0,
\\widehat{P}(\\omega\_{2}) = 0.0\$

<p class="fragment">
\\begin{align}
i(N) &= - \\sum\_{j}\\widehat{P}(\\omega\_{j})\\log\_{2}{\\widehat{P}(\\omega\_{j})}\\\\
&= -\\left[ \\widehat{P}(\\omega\_{1})\\log\_{2}{\\widehat{P}(\\omega\_{1})} + \\widehat{P}(\\omega\_{2})\\log\_{2}{\\widehat{P}(\\omega\_{2})}\\right]\\\\
&= -\\left[ 1.0\\log\_{2}{1.0} \\right]\\\\
&= -\\left[ 1.0(0) \\right]\\\\
&= 0
\\end{align}
</p>

## Measures of Node Impurity: Variance (Two-Class)

**Variance Impurity:**

\$\$ i(N) = \\widehat{P}(\\omega\_{1})\\widehat{P}(\\omega\_{2}) \$\$

<p class="fragment">This is most useful in the two-category case; if the node is pure, then either \$\\widehat{P}(\\omega\_{1})\$ or \$\\widehat{P}(\\omega\_{2})\$ is \$0\$, and therefore \$i(N)=0\$.</p>

## Measures of Node Impurity: Gini (Multi-Class)

***Gini Impurity:***

\$\$ i(N) = \\sum\_{i\\neq j}\\widehat{P}(\\omega\_{i})\\widehat{P}(\\omega\_{j}) = \frac{1}{2}\\left[1-\\sum\_{j}\\widehat{P}\^{2}(\\omega\_{j})\\right]\$\$

<p class="fragment">This is the extension of Variance Impurity to the multi-class case.</p>

<p class="fragment">
We simply sum over the product of priors of every unique pair of \$\\omega\_{i}\$
and \$\\omega\_{j}\$.
</p>

## Measures of Node Impurity: Misclassification

***Misclassification Impurity:***

\$\$ i(N) = 1-\\max\_{j}{\\widehat{P}(\\omega\_{j})} \$\$

<p class="fragment">
This is the minimum probability that a training pattern would be misclassified
at \$N\$.
</p>

## Node Impurity as a Function of Classes

![Node Impurity Function](img/impurity_function.svg){width=80%}

## Selecting a Specific Query to Test

So we have two criteria: a ***simple*** tree that leads to nodes with ***minimum impurity***.

<p class="fragment">Therefore, given node \$N\$, what split \$s\$ should we use for query \$T\$?</p>

## Selecting a Specific Query to Test

Since we're trying to minimize impurity (i.e. maximize purity) at the descendent nodes, we can calculate the ***impurity gradient***:

<p class="fragment">\$\$\\Delta i(s) = i(N)-\\widehat{P}_{L}i(N\_{L})-(1 - \\widehat{P}\_{L})i(N\_{R})\$\$</p>

<p class="fragment">
\$N\_{L}, N\_{R}\$ are the left and right descendent nodes and \$i(N\_{L}), i(N\_{R})\$
are their impurities.
</p>

<p class="fragment">
\$\\widehat{P}\_{L}\$ is the fraction of patterns / data at node \$N\$ that go to
\$N\_{L}\$ when query \$T\$ is used.
</p>

<p class="fragment">Thus, we seek the split \$s\^{\\star}=\\max\_{s}{\\Delta i(s)}\$.</p>

<p class="fragment">
If we are using entropy impurity, this is equivalent to ***maximizing
information gain***.
</p>

## Local vs. Global Query Selection

\$\$ \\Delta i(s) = i(N) - \\widehat{P}\_{L}i(N\_{L}) - (1 - \\widehat{P}\_{L})i(N\_{R}) \$\$

For binary trees, this is a one-dimensional optimization problem.

<p class="fragment">This is a ***local, greedy*** strategy: we select the optimal \$s\$ at each \$N\$ separately.</p>

<p class="fragment">This is NOT guaranteed to be ***globally optimal***, and will not necessarily yield a small tree.</p>

<p class="fragment">However, it is a straightforward method that reaches a solution in a reasonable amount of time.</p>

## Differences in Impurity Functions

The selection of impurity function can affect the selection of a split.

<p class="fragment">Imagine that at node \$N\$ there are 100 points, 90 of which are from \$\\omega\_{1}\$ and 10 of which are in \$\\omega\_{2}\$.</p>

<p class="fragment">
Now you have to evaluate a proposed split that will create two new nodes (binary
tree), where the left node contains 30 points (20 from \$\\omega\_{1}\$, and 10 from
\$\\omega\_{2}\$), and the right contains 70 points (70 from \$\\omega\_{1}\$, and 0
from \$\\omega\_{2}\$).
</p>

<p class="fragment">What are the ***Misclassification*** and ***Gini*** impurities?</p>

## Differences in Impurity Functions

***Misclassification Impurity***

\\begin{align}
i\_{MI}(N) &= 1 - \\max\_{j}\\widehat{P}(\\omega\_{j})\\\\
 &= 1 - 0.9 = 0.1 \\\\
i\_{MI}(N\_{R}) &= 1 - 1 = 0 \\\\
i\_{MI}(N\_{L}) &= 1 - 0.66 = 0.33 \\\\
\\Delta i\_{MI}(N) &= i(N) - \\widehat{P}\_{L} i(N\_{L}) - (1 - \\widehat{P}\_{L}) i(N\_{R})\\\\
&= 0.1 - 0.3 * 0.3 - (1-0.3) * 0\\\\
&= \\mathbf{0}
\\end{align}

## Differences in Impurity Functions

***Gini Impurity***

\\begin{align}
i\_{GI}(N) &= \frac{1}{2}\\left[1-\\sum\_{j}\\widehat{P}\^{2}(\\omega\_{j})\\right]\\\\
 &= \frac{1}{2}\\left[1 - (0.9\^{2} + 0.1\^{2})\\right] = 0.09 \\\\
i\_{GI}(N\_{R}) &= 0.5 \\left[1 - (1.0\^{2} + 0\^{2})\\right] = 0 \\\\
i\_{GI}(N\_{L}) &= 0.5 \\left[1 - (0.66\^{2} + 0.33\^{2})\\right] = 0.22 \\\\
\\Delta i\_{GI}(N) &= i(N) - \\widehat{P}\_{L} i(N\_{L}) - (1 - \\widehat{P}\_{L}) i(N\_{R})\\\\
&= 0.09 - 0.33 * 0.22 - (1-0.3) * 0\\\\
&= \\mathbf{0.0234}
\\end{align}

## Twoing Criterion for Multiple Classes

The goal in the multiclass case is to identify ***groups*** of the \$c\$ classes.

<p class="fragment">
If our set of categories is
\$\\mathcal{C}=\\{\\omega\_{1},\\omega\_{2},\ldots,\\omega\_{c}\\}\$, we want a split that
creates ***supercategories***
\$\\mathcal{C}\_{1}=\\{\\omega\_{1i},\\omega\_{2i},\ldots,\\omega\_{ik}\\}\$ and
\$\\mathcal{C}\_{2} = \\mathcal{C}-\\mathcal{C}\_{1}\$.
</p>

<p class="fragment">
Our impurity criterion is now the ***change*** in impurity: \$\\Delta
i(s,\\mathcal{C}\_{1})\$. Thus, we find the split \$s\^{\ast}(\\mathcal{C}\_{1})\$ that
maximizes the impurity gradient, given \$\\mathcal{C}\_{1}\$.
</p>

<p class="fragment">
Then we find the supercategory \$\\mathcal{C}\_{1}\^{\\ast}\$ that maximizes \$\\Delta
i(s\^{\\ast}(\\mathcal{C}\_{1}), \\mathcal{C}\_{1})\$, meaning that we are searching
for both the optimal split AND the optimal supercategories.
</p>

<p class="fragment">***This has many applications in biomedical data!***</p>

## Supercategories in Prostate Cancer

<div class="l-multiple" style="grid-template-columns: auto auto auto auto;">
<div style="grid-row: 1;">
![Gleason Grade 3](img/gleason_g3.png){width=80%}
</div>
<div style="grid-row: 1;">
![Gleason Grade 4](img/gleason_g4.png){width=80%}
</div>
<div style="grid-row: 1;">
![Gleason Grade 5](img/gleason_g5.png){width=80%}
</div>
<div style="grid-row: 1 / span 2; vertical-align: middle;">
![Neoplasia (PIN)](img/gleason_pin.png){width=80%}
</div>
<div style="grid-row: 2;">
![Benign Epithelium](img/gleason_epi.png){width=80%}
</div>
<div style="grid-row: 2;">
![Benign Stroma](img/gleason_str.png){width=80%}
</div>
<div style="grid-row: 2;">
![Atrophy](img/gleason_atr.png){width=80%}
</div>
</div>

## Supercategories in Prostate Cancer

![How to Select the Best Supercategories?](img/tissue_cascade.png){width=90%}

# 
## Stop Splitting

## When to Stop?

What happens if we keep splitting until each leaf only contains one data point?

<ul>
<li class="fragment">We've ***over-trained*** our tree!</li>
<li class="fragment">This is a classifier with ***high complexity*** and ***low generalizability***.</li>
</ul>

## Under-Training Scenario

What happens if we stop splitting our training set too early?

<ul>
<li class="fragment">We've ***under-trained*** our tree!</li>
<li class="fragment">This is a classifier with ***low complexity*** and ***low accuracy***.</li>
</ul>

## When to Stop Splitting

So we need to know how to stop splitting - How should we do this?

<ol>
<li class="fragment">Cross-validation</li>
<li class="fragment">Threshold the impurity gradient</li>
<li class="fragment">Set a global criterion function that measures complexity vs. accuracy</li>
</ol>

## Cross-Validation Criterion

We separate \$\\mathcal{D}\$ into ***training***, \$\\mathcal{D}\_{1}\$, and ***validation***, \$\\mathcal{D}\_{2}=\\mathcal{D}-\\mathcal{D}\_{1}\$.

<p class="fragment">Use \$\\mathcal{D}\_{1}\$ to build a tree, then evaluate performance with \$\\mathcal{D}\_{2}\$.</p>

<p class="fragment">Continue splitting the tree until performance peaks on $\\mathcal{D}\_{2}\$.</p>

<p class="fragment">Consider this method if you have a LOT of training data, and you are reasonably sure that $\\mathcal{D}\_{2}\$ generalizes well to the entire feature space.</p>

## Threshold Impurity

Keep splitting until the best splits do not change the impurity:

<p class="fragment">\$\$ \\max\_{s}{\\Delta i(s)}\\leq \\beta \$\$</p>

<p class="fragment">This allows us to use all of \$\\mathcal{D}\$ to train the tree (without splitting off a validation set).</p>

<p class="fragment">Leaf nodes can exist at different levels in the tree, allowing for varieties of class complexity.</p>

<p class="fragment">Need to select \$\\beta\$ appropriately?</p>

## Global Criterion Function

To balance accuracy and complexity, we can define an overall ***complexity function*** to minimize:

<p class="fragment">\$\$ \\alpha \\ast \\left[\\textrm{size}\\right] + \\sum\_{\\textrm{nodes}}i(N) \$\$</p>

<p class="fragment">where \$\\left[\\textrm{size}\\right]\$ is the number of nodes or links, and \$\\alpha\$ is a positive constant.</p>

<p class="fragment">We keep splitting the tree until this function is minimized.</p>

<p class="fragment">If we are using entropy impurity, this is related to the ***minimum description length (MDL)***, which we'll talk about later when we discuss classifier evaluation.</p>

<p class="fragment">The first term penalizes having too many nodes or links, while the second term penalizes leaf nodes with a high degree of impurity.</p>

<p class="fragment">Again, we have a ***parameter*** in \$\\alpha\$ that we need to set.</p>

# 
## Pruning Trees

## The Horizon Effect

Building a tree from the ground up requires us to decide whether to split ***at the current node***.

<p class="fragment">We cannot incorporate information about descendent nodes – the so-called ***horizon effect***.</p>

<p class="fragment">In ***pruning***, we build an "exhaustive" tree, and try to merge pairs of neighboring leaves together – the inverse of splitting.</p>

<p class="fragment">Pairs are merged if they yield an acceptable (small) increase in impurity, and the common antecedent node is declared a leaf.</p>

## Pruning Uses All Features

Pruning uses ***all training***, and avoids the horizon effect by working backwards.

<p class="fragment">For large training sets and complex problems, this can be computationally prohibitive.</p>

<p class="fragment">For small training sets, pruning is generally preferred, but is ***sensitive to training data***!</p>

## Pruning Sensitivity to Training

![Grey Area Indicates Lower-Left Region](img/pruning01.png){width=90%}

## Example of Pruning and Volatility

![Slight Change in Training Leads to New Splits](img/pruning02.png){width=90%}

# 
## Classifying a Leaf

## Leaf Class Assignment

This rule is easy:

<ul>
<li class="fragment">A ***pure*** leaf is assigned to the class of all its data points.</li>
<li class="fragment">An ***impure*** leaf is assigned to the ***dominant class***, i.e.  the class that has the most samples at that node.</li>
</ul>

# 
## Missing Attributes

## Missing Attributes

We sometimes encounter ***deficient patterns***, or data points with some during training or classification.

<p class="fragment">If we throw out the deficient patterns at the start, we end up wasting a lot of valuable data!</p>

## Calculating Splits with Missing Data

For training, we can just ignore the missing attributes.

<p class="fragment">To handle classification, we can also calculate “surrogate” splits that approximate the primary.</p>

<p class="fragment">If a pattern is deficient, we use the best split that does not include the missing attributes.</p>

## Missing Attributes

![](img/missing_attrs.png){width=70%}

# 
## Handling Multiple Features

## Combinations of Features

Sometimes it can be more efficient to use linear combinations of features to define splits.

<p class="fragment">This can help create trees with ***simple decision planes***, which are probably more realistic. It's easy to see this in the two-variable case.</p>

## Combinations of Features: Multivariate Trees

![Linear Bounds](img/multivariate_trees_threshold.png){width=80%}

## Combinations of Features: Multivariate Trees

![Combinations of Features](img/multivariate_trees_lincomb.png){width=80%}

# 
## Other Training Methods

## Interactive Dichotomizer 3 (ID3)

ID3: used with nominal / unordered inputs.

<p class="fragment">Every split has a branching factor \$B\_{j}\$, which is the number of attribute bins of variable \$j\$ used for splitting -- not often binary.</p>

<p class="fragment">The number of levels is equal to the number of input variables.</p>

<p class="fragment">The algorithm continues until all nodes are pure, and there are no more variables.</p>

<p class="fragment">This can be followed up by pruning.</p>

## C4.5

C4.5, a successor to ID3, is a popular method for decision trees.

<p class="fragment">Improvements to ID3:</p>

<ul>
<li class="fragment">Handles real-valued inputs (like CART)</li>
<li class="fragment">Ignores missing attributes for calculating gain and entropy, and attributes with different costs</li>
<li class="fragment">Employs multi-way splits (like ID3)</li>
<li class="fragment">Uses pruning based on the statistical significance method</li>
</ul>

## Differences Between C4.5 and CART

C4.5 does not pre-compute “surrogate” splits, as in CART.

<li class="fragment">If a node \$N\$ with branching factor \$B\$ requires the missing data during classification, C4.5 will follow EACH of the links down to the \$B\$ leaf nodes.</li>

<li class="fragment">The classification label is based on the labels of the \$B\$ leaf nodes, weighted by the decision probabilities at \$N\$ of the training data.</li>

<li class="fragment">During pruning, C4.5 “collapses” redundant rules from the root node to a leaf.</li>

## Pruning in C4.5

<div class="l-double">
<div>
The leftmost leaf in this tree has an associated rule:

**IF:**
\$\$\\begin{bmatrix}
(0.04x\_{1} + 0.16x\_{2} < 0.11)\\\\
\\textrm{AND }(0.27x\_{1}-0.44x\_{2} < -0.02)\\\\
\\textrm{AND }(0.96x\_{1}-1.77x\_{2} < -0.45)\\\\
\\textrm{AND }(5.43x\_{1}-13.33x\_{2} < -6.03)
\\end{bmatrix}\$\$

**THEN:**
\$\$\\mathbf{x}\\in\\omega\_{1}\$\$

</div>
<div>


![Pruning Figure](img/pruningc45.png){height=80%}

</div>
</div>

## Pruning in C4.5

<div class="l-double">

<div>


This rule can be simplified as:

**IF:**
\$\$\\begin{bmatrix}
(0.04x\_{1} + 0.16x\_{2} < 0.11)\\\\
\\textrm{AND }(5.43x\_{1}-13.33x\_{2} < -6.03)
\\end{bmatrix}\$\$

**THEN:**
\$\$\\mathbf{x}\\in\\omega\_{1}\$\$

Note that unlike the leaf merging strategy of CART, the rule-based method can
collapse nodes near the root.

</div>
<div>


![Pruning Figure](img/pruningc45.png){height=80%}

</div>
</div>

# 
## Parting Words

## Advantages of Decision Trees

<ul>
<li class="fragment">Simple to understand and to interpret, and can be visualised</li>
<li class="fragment">Requires little preparation (normalization, cleaning, etc.)</li>
<li class="fragment">Cost of predicting is low</li>
<li class="fragment">Handles both numerical and categorical data</li>
<li class="fragment">Handles multiple classes</li>
</ul>

## Disadvantages of Decision Trees

<ul>
<li class="fragment">Trees can be overly complex and may not generalise well (overfitting)</li>
<li class="fragment">Can be dependent on small variations in the data (brittle)</li>
<li class="fragment">Creating an ***optimal*** tree is computationally difficult</li>
<li class="fragment">Trees can be biased if some classes are very rare</li>
</ul>

## Which Decision Tree is Best?

The general components of decision tree design are:

<ul>
<li class="fragment">Feature processing</li>
<li class="fragment">Impurity measure for deciding on optimal splits</li>
<li class="fragment">Stopping criterion for growing the tree</li>
<li class="fragment">Pruning method</li>
</ul>

<p class="fragment">
Each algorithm has attributes that may be
well-suited to your dataset.
</p>

<p class="fragment">
As with everything in ML, you should experiment, see
what works and what doesn't.
</p>

<p class="fragment">Remember that there is no such thing as a
"best" classifier for all situations.
</p>

# 
## Next Class

<ul>
<li class="fragment">Extensions of Decision Trees: Randomized Decision Trees</li>
<li class="fragment">Random Forests Algorithm, an early classifier ensemble algorithm</li>
</ul>

# 

## Thank You!
