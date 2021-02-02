---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade

title: LINEAR DISCRIMINANTS (Pt. 2)
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
contact: scottdoy@buffalo.edu
date: 2020-03-05
---

# 

## Recap

## Recap: Linear Machines

General form of a linear machine: 

\$ g\_{i}(\\mathbf{x}) = \\mathbf{w}\_{i}\^{T}\\mathbf{x} + w\_{i0} \$

## Geometric Interpretation of Linear Machine

![Discriminant in a Flat Plane](img/discplane.png){width=40%}

## Higher-dimensional Representations

We can generalize this to a higher-dimensional space by a change of variables:

\\begin{align}
g(\\mathbf{x}) \&= \\sum\_{i=1}\^{\\widehat{d}} a\_{i}y\_{i}(\\mathbf{x}) \\\\
g(\\mathbf{x}) \&= \\mathbf{a}\^{T}\\mathbf{y} \\\\
\\end{align}

where \$\\mathbf{a}\^{T}\$ is a \$\\widehat{d}\$-dimensional weight and \$y\_{i}\$ are
arbitrary functions of \$\\mathbf{x}\$.

## Sidebar: Where Did the Bias Term Go?

We can include the bias term in the discriminant function by setting \$x\_{0} = 1\$:

\$ g(\\mathbf{x}) = w\_{0} + \\sum\_{i=1}\^{d}w\_{i}x\_{i} = \\sum\_{i=0}\^{d}w\_{i}x\_{i} \$

Basically we "absorb" the bias term \$w\_{0}\$ into the weight vector, and then add a dimension to \$\\mathbf{x}\$, so we start the summation from \$0\$ instead of \$1\$.

## Sidebar: Augmented Vectors

This gives us the following mappings, which we call **augmented vectors**:

\$ \\mathbf{y}=
\\begin{bmatrix}
1\\\\
x\_{1}\\\\
\\vdots\\\\
x\_{d}
\\end{bmatrix} =
\\begin{bmatrix}
1\\\\
\\mathbf{x}
\\end{bmatrix}
\\qquad
\\mathbf{a}=
\\begin{bmatrix}
w\_{0}\\\\
w\_{1}\\\\
\\vdots\\\\
w\_{d}
\\end{bmatrix} =
\\begin{bmatrix}
w\_{0}\\\\
\\mathbf{w}
\\end{bmatrix}\$

We reduced the problem of finding a weight vector \$\\mathbf{w}\$ AND a bias weight \$w\_{0}\$ to finding just a single weight vector \$\\mathbf{a}\$.

## Recap: Calculation of a Polynomial Discriminant Function

<div class="l-double">
<div>
![Polynomial Discriminant Function](img/polynomial_disc_func.svg){width=100%}
</div>
<div>
Our discriminant function has the form:

\$g(x) = a\_{1} + a\_{2}x + a\_{3}x\^{2}\$

The discriminant function is characterized by:

\$\\mathbf{y} = (1, x, x\^{2})\^{T}\$

which projects the 1D data from \$x\$ onto the 3D curve in
\$\\mathbf{y}\$-space.

</div>
</div>

## Recap: How Should We Find \$\\mathbf{a}\$?

We need to find a solution to the set of linear inequalities \$\\mathbf{a}\^{T}\\mathbf{y}\_{i} > 0\$ (or \$\\mathbf{a}\^{T}\\mathbf{y}\_{i} \\geq b\$).

We can define a criterion function \$J(\\mathbf{a})\$ that spits out a number which is minimized when \$\\mathbf{a}\$ is an optimal solution vector.

This is a **scalar function optimization** or **numerical optimization** problem.

# 
## Linearly Separable Cases

## How Do We Find \$\\mathbf{a}\$?

To set up numerical optimization, we need to define our criterion function.

<p class="fragment">To define our criterion function, let's consider an ideal dataset of **linearly separable** training samples.</p>

## Linearly Separable Cases

Linear discriminant function: \$g(\\mathbf{x}) = \\mathbf{a}\^{T}\\mathbf{y}\$.

<p class="fragment">Set of \$n\$ training samples: \$\\mathbf{y}, \\ldots, \\mathbf{y}\_{n}\$ labeled either as \$\\omega\_{1}\$ or \$\\omega\_{2}\$.</p> 

<p class="fragment">In the binary class case, a sample is correctly classified if:</p>

<p class="fragment">\$ (\\mathbf{a}\^{T}\\mathbf{y}\_{i} > 0 \\textrm{ and } \\mathbf{y}\_{i} \\textrm{ is labeled } \\omega\_{1}) \\textrm{ or } (\\mathbf{a}\^{T}\\mathbf{y}\_{i} < 0 \\textrm{ and } \\mathbf{y}\_{i} \\textrm{ is labeled } \\omega\_{2}) \$</p>

<p class="fragment">We need to find the weight vector \$\\mathbf{a}\$ that maximizes our classifier's performance.</p>

<p class="fragment">If training samples are **linearly separable**, the best classifier is perfect: ALL samples should be correctly classified.</p> 

## Weight Vector

<div class="l-double">
<div>
![Hyperplanes in Weight Space](img/augmented_feature_space.svg){width=80%}
</div>
<div>

Weight vector \$\\mathbf{a}\$ specifies a point in **weight space**, the space of all possible weight vectors.

<p class="fragment">For a sample \$\\mathbf{y}\_{i}\$, the equation \$\\mathbf{a}\^{T}\\mathbf{y}\_{i} = 0\$ defines a hyperplane through the origin of weight space, with \$\\mathbf{y}\_{i}\$ as a normal vector.</p>

</div>
</div>

## Weight Vector

<div class="l-double">
<div>
![Hyperplanes in Weight Space](img/augmented_feature_space.svg){width=80%}
</div>
<div>
The optimal solution vector is "constrained" by each training sample.

The solution vector lies within the intersection of these half-planes.

</div>
</div>

## Normalization

There's a trick to finding the optimal weight vector, called **normalization**.

Since we know which samples belong to \$\\omega\_{1}\$ or \$\\omega\_{2}\$ during training, it's actually easier to find a weight vector where \$\\mathbf{a}\^{T}\\mathbf{y}\_{i} > 0\$ for **all** samples.

We can do this by simply flipping the signs of \$\\mathbf{y}\$ for one of the classes. 

Thus we're looking for a weight vector \$\\mathbf{a}\$ that is on the positive side of all possible hyperplanes defined by the training samples.

## Pre-Normalization

<div class="l-double">
<div>
![Pre-Normalized Space](img/pre_normalized.svg){width=100%}
</div>
<div>
The red dotted line is one of the possible separating hyperplanes.

The solution vector is normal (and positive) to the hyperplane.

The grey region denotes the region of possible solution vectors, which we call the **solution space**.

Note that each of the possible solutions is orthogonal to one of \$\\mathbf{y}\_{i}\$.

</div>
</div>

## Post-Normalization

<div class="l-double">
<div>
![Post-Normalized Space](img/post_normalized.svg){width=100%}

</div>
<div>


Following "normalization", the sign of the cases labeled \$\\omega\_{2}\$ is
flipped.

Now we have a solution that corresponds to the discriminant function
\$g(\\mathbf{x}) = \\mathbf{a}\^{T}\\mathbf{y}\$.

Again, using what we have so far, the solution vector \$\\mathbf{a}\$ is not unique
-- any vector in the "solution region" is valid.

</div>
</div>

## Selecting Optimal Solutions

<div class="l-double">
<div>
![Potential Solutions in Grey](img/solution_space.svg){width=80%}
</div>
<div>


Which of these solutions is "best"?

We can specify that we want our chosen solution vector to be the one that **maximizes the minimum distance** from training samples to the separating hyperplane.

</div>
</div>

## Selecting Optimal Solutions

<div class="l-double">
<div>
![Potential Solutions in Grey](img/solution_space_margin.svg){width=80%}
</div>
<div>
Thus, we want to obtain a solution vector for which:

\$\\mathbf{a}\^{T}\\mathbf{y}\_{i}\\geq b \\text{ for all } i\$

where \$b>0\$ is some **margin** factor.

This is stronger than saying we want \$\\mathbf{a}\^{T}\\mathbf{y}\_{i}\\geq 0\$, since now we have some margin that we're using to "insulate" the decision region with a distance of \$\\frac{b}{||\\mathbf{y}\_{i}||}\$.

</div>
</div>

## Introducing a Margin Constraint

<div class="l-double">
<div>
![Potential Solutions in Grey](img/solution_space.svg){width=80%}
</div>
<div>
![Margin Provides "Padding"](img/solution_space\_margin.svg){width=80%}
</div>
</div>

## We Still Need \$\\mathbf{a}\$!

We've now defined our criteria for what we'd like a "good" \$\\mathbf{a}\$ to look like, but we still need to use numerical optimization to get it.

This basic technique is good for any problem where there is not a "closed-form" solution -- in other words, we can't simply "solve for \$\\mathbf{a}\$", we have to use a kind of trial-and-error or hot-and-cold approach.

# 
## Numerical Optimization 

## Numerical Optimization

The basic strategy behind numerical optimization is:

1. You have a cost function (\$J(\\mathbf{a})\$) you want to minimize.
2. You have a (random?) set of parameters that define \$\\mathbf{a}\$.
3. On iteration 1, calculate the cost \$J(\\mathbf{a})\$ for your initial conditions.
4. On the next iteration, "nudge" your parameters and see how \$J(\\mathbf{a})\$ changes.

	- If the cost goes up, go back and try a different "nudge".
	- If the cost goes down, keep "nudging" in the same direction.
	- If the cost is the same, stay where you are.

5. Repeat Step 4 until you reach convergence, where your cost function is barely changing.

## Numerical Optimization Demonstration

Some great examples of numerical optimization can be found here:

*An Interactive Tutorial on Numerical Optimization*:
: [http://www.benfrederickson.com/numerical-optimization/](http://www.benfrederickson.com/numerical-optimization/)
  
## Basic Gradient Descent

<div class="l-double">
<div>
![](img/gradient_descent.svg){width=100%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}, \\theta, \\eta\_{k}(\\cdot)\$
<ol>
<li class="fragment">\$k \\leftarrow k+1\$</li>
<li class="fragment">\$\\mathbf{a} \\leftarrow \\eta\_{k}\\Delta
J(\\mathbf{a})\$</li>
<li class="fragment">Repeat 1-2 until \$|\\eta\_{k}\\Delta J(\\mathbf{a})| < \\theta\$</li>
</ol>

</div>
</div>

## Recap: Questions to Consider for Gradient Descent

Once you dig into the details, there are a number of questions:

1. How far should we "nudge" our parameter set? (Setting the learning rate)
2. What form should our optimization function be?
3. How do we avoid getting stuck in local minima?
4. When should we stop "nudging"? (Identifying convergence)
5. How computationally complex is our algorithm?

# 
## Setting the Learning Rate

## Setting the Learning Rate

We need to specify the learning rate \$\\eta\$ such that it is reasonably fast
(i.e. not too small) but also will not overshoot (i.e. not too big).

I'll spare you the math, but using a second-order expansion around
\$\\mathbf{a}\_{k}\$ and some second partial derivatives, we can get:

\$ \\eta\_{k} = \\frac{||\\Delta J||\^{2}}{\\Delta J\^{T}\\mathbf{H}\\Delta J} \$

where \$\\mathbf{H}\$ is the **Hessian matrix**, which is the matrix of second
partial derivatives of \$J\$.

If the criterion function is quadratic everywhere, then \$\\eta\_{k}\$ is the same
for all \$k\$.

## Setting the Learning Rate: Not Trivial!

![Setting the Learning Rate (NeurIPS 1992)](img/lecunpaper.png){width=50%}

# 
## Form of the Optimization Function

## Choosing a Form for \$J(\\mathbf{a})\$: Piecewise Constant Function

What function should we create for \$J(\\mathbf{a})\$?

We could let \$J(\\mathbf{a}; \\mathbf{y}\_{1},\\\cdots,\\mathbf{y}\_{n})\$ be the number
of misclassified samples: If \$\\mathbf{a}\$ leads to more misclassifications, this
means a higher value for \$J(\\mathbf{a})\$. If all samples are correctly
classified, then \$J(\\mathbf{a})=0\$.

This is difficult to optimize because it's
**piecewise constant**: In other words, if the number of misclassified samples
doesn't change, then \$\\Delta J(\\mathbf{a}) = 0\$, and we don't go anywhere.

## \$J(\\mathbf{a})\$: Piecewise Constant Function

![Piecewise Constant Function](img/piecewise_constant_function.svg){width=50%}

## \$J\_{p}(\\mathbf{a})\$: Perceptron Criterion Function

Alternatively, we could use the **Perceptron criterion function**:

\$ J\_{p}(\\mathbf{a}) = \\sum\_{\\mathbf{y}\\in\\mathcal{Y}}(-\\mathbf{a}\^{T}\\mathbf{y}) \$

where \$\\mathcal{Y}\$ is the set of misclassifications due to \$\\mathbf{a}\$.

If \$\\mathcal{Y} = \\emptyset\$ (there are no misclassifications), then
\$J\_{p}(\\mathbf{a}) = 0\$ and we're done.

## \$J\_{p}(\\mathbf{a})\$: Perceptron Criterion Function

\$ J\_{p}(\\mathbf{a}) = \\sum\_{\\mathbf{y}\\in\\mathcal{Y}}(-\\mathbf{a}\^{T}\\mathbf{y}) \$

Remember before, how we said that thanks to our "normalization", we can define a
solution vector such that \$\\mathbf{a}\^{T}\\mathbf{y}>0\$ for all samples?

This means that \$\\mathbf{a}\^{T}\\mathbf{y} \\leq 0\$ only if \$\\mathbf{y}\$ is
misclassified; in other words, there are samples in \$\\mathcal{Y}\$.

Thus, during optimization, \$J\_{p}(\\mathbf{a})\$ is never negative and is only 0
when \$\\mathbf{a}\$ is a solution vector or on the decision boundary (i.e.
\$\\mathcal{Y}\$ is empty).

Geometrically, this is proportional to the sum of the distances from
misclassified samples to the decision boundary.

## \$J\_{p}(\\mathbf{a})\$: Perceptron Criterion Function

![Perceptron Criterion Function](img/perceptron_criterion_function.svg){width=50%}

## Updating \$\\mathbf{a}\$ Using the PCF

To find \$\\Delta J\_{p}\$, we're calculating \$\\frac{\\partial J\_{p}}{\\partial
a\_{j}}\$ for each component in \$\\mathbf{a}\$, so:

\$ \\Delta J\_{p} = \\sum\_{\\mathbf{y}\\in\\mathcal{Y}}(-\\mathbf{y}) \$

So our update rule is now:

\$ \\mathbf{a}\_{k+1} = \\mathbf{a}\_{k} + \\eta\_{k}\\sum\_{\\mathbf{y}\\in\\mathcal{Y}}(\\mathbf{y}) \$

In other words: We add some multiple of the sum of the misclassified samples to
the present weight vector to get the next weight vector.

This is called the **batch Perceptron algorithm** since we can use batches of
samples.

## Batch Perceptron Algorithm

<div class="l-double">
<div>
![](img/batch_perceptron_algorithm.svg){width=100%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}, \\theta, \\eta\_{k}(\\cdot)\$
<ol>
<li class="fragment">\$k \\leftarrow k+1\$</li>
<li class="fragment">\$\\mathbf{a} \\leftarrow \\mathbf{a} + \\eta\_{k}\\sum\_{\\mathbf{y}\\in\\mathcal{Y}\_{k}}(\\mathbf{y})\$</li>
<li class="fragment">Repeat 1-2 until \$|\\eta\_{k}\\sum\_{\\mathbf{y}\\in\\mathcal{Y}\_{k}}(\\mathbf{y})| < \\theta\$</li>
</ol>
</div>
</div>

## Batch Perceptron Algorithm

<div class="l-double">
<div>
![](img/batch_perceptron_algorithm.svg){width=100%}
</div>
<div>
Process:

- Start with \$\\mathbf{a}\_{k}=\\mathbf{0}\$: All three samples are misclassified,
  so you add \$\\mathbf{y}\_{1}+\\mathbf{y}\_{2}+\\mathbf{y}\_{3}\$.
- Then sample \$\\mathbf{y}\_{3}\$ is misclassified, so that is added.
- Then \$\\mathbf{y}\_{1}\$, and \$\\mathbf{y}\_{3}\$ again.
- After that, the vector lands in the solution region, so the process ends.

</div>
</div>

## Convergence Properties

To examine whether this process will converge, we look at a few simplifications.

Instead of building a set \$\\mathcal{Y}\$ of all misclassified samples, let's look
at each sample in sequence and modify the weights for each individual
misclassified sample.

Further, we'll assume that our learning rate \$\\eta\_{k}\$ is constant for all \$k\$,
known as the **fixed-increment** case.

To investigate convergence behavior, we simply cycle through the samples
infinitely many times and keep modifying the weight until the algorithm changes.

For this, we'll denote misclassified cases with superscripts, so
\$\\mathbf{y}\^{k}\$ denote the \$k\$-th misclassified sample drawn from the full set
\$\\mathcal{Y}\$.

## Convergence Properties

Thus, if we have three samples and consider them in an infinite sequence, with
misclassified samples underlined:

\$ \\underline{\\mathbf{y}\_{1}}, \\mathbf{y}\_{2}, \\underline{\\mathbf{y}\_{3}}, \\underline{\\mathbf{y}\_{1}}, \\underline{\\mathbf{y}\_{2}}, \\mathbf{y}\_{3}, \\mathbf{y}\_{1}, \\underline{\\mathbf{y}\_{2}}, \\mathbf{y}\_{3}, \\ldots \$

then the sequence \$\\mathbf{y}\^{1}, \\mathbf{y}\^{2}, \\mathbf{y}\^{3},
\\mathbf{y}\^{4}, \\mathbf{y}\^{5}\$ represents samples \$\\mathbf{y}\_{1},
\\mathbf{y}\_{3}, \\mathbf{y}\_{1}, \\mathbf{y}\_{2}, \\mathbf{y}\_{2}\$.

With all this, we can write the **fixed-increment rule**:

\$\\begin{array}{cl}
a\_{1} & \quad\\text{arbitrary} \\
a\_{k+1} = a\_{k} + \\mathbf{y}\^{k} & \quad k \\geq 1\\
\\end{array}\$

where \$\\mathbf{a}\_{k}\^{T}\\mathbf{y}\^{k}\\leq 0\$ for all \$k\$.

## Representation of Fixed-Increment

<div class="l-double">
<div>
![](img/fixed_increment_01.svg){width=70%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}\$
<ol>
<li class="fragment">\$k \\leftarrow k+1\$</li>
<li class="fragment">if \$\\mathbf{y}\^{k}\\in\\mathcal{Y}\$ then:</li>
<ul>
<li class="fragment">\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\mathbf{y}\^{k}\$ </li>
</ul>
<li class="fragment">Repeat until \$|\\mathcal{Y}=\\emptyset\$</li>
</ol>
</div>
</div>

## Representation of Fixed-Increment

<div class="l-double">
<div>
![](img/fixed_increment_02.svg){width=70%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}\$
<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{y}\^{k}\\in\\mathcal{Y}\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\mathbf{y}\^{k}\$ </li>
</ul>
<li>Repeat until \$|\\mathcal{Y}=\\emptyset\$</li>
</ol>
</div>
</div>

## Representation of Fixed-Increment

<div class="l-double">
<div>
![](img/fixed_increment_03.svg){width=70%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}\$
<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{y}\^{k}\\in\\mathcal{Y}\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\mathbf{y}\^{k}\$ </li>
</ul>
<li>Repeat until \$|\\mathcal{Y}=\\emptyset\$</li>
</ol>
</div>
</div>

## Representation of Fixed-Increment

<div class="l-double">
<div>
![](img/fixed_increment_04.svg){width=70%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}\$
<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{y}\^{k}\\in\\mathcal{Y}\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\mathbf{y}\^{k}\$ </li>
</ul>
<li>Repeat until \$|\\mathcal{Y}=\\emptyset\$</li>
</ol>
</div>
</div>

## Representation of Fixed-Increment

<div class="l-double">
<div>
![](img/fixed_increment_05.svg){width=70%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}\$
<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{y}\^{k}\\in\\mathcal{Y}\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\mathbf{y}\^{k}\$ </li>
</ul>
<li>Repeat until \$|\\mathcal{Y}=\\emptyset\$</li>
</ol>
</div>
</div>

## Representation of Fixed-Increment

<div class="l-double">
<div>
![](img/fixed_increment_06.svg){width=70%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}\$
<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{y}\^{k}\\in\\mathcal{Y}\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\mathbf{y}\^{k}\$ </li>
</ul>
<li>Repeat until \$|\\mathcal{Y}=\\emptyset\$</li>
</ol>
</div>
</div>

## Representation of Fixed-Increment

<div class="l-double">
<div>
![](img/fixed_increment_07.svg){width=70%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}\$
<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{y}\^{k}\\in\\mathcal{Y}\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\mathbf{y}\^{k}\$ </li>
</ul>
<li>Repeat until \$|\\mathcal{Y}=\\emptyset\$</li>
</ol>

</div>
</div>

## Representation of Fixed-Increment

<div class="l-double">
<div>
![](img/fixed_increment_08.svg){width=70%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}\$
<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{y}\^{k}\\in\\mathcal{Y}\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\mathbf{y}\^{k}\$ </li>
</ul>
<li>Repeat until \$|\\mathcal{Y}=\\emptyset\$</li>
</ol>
</div>
</div>

## Representation of Fixed-Increment

<div class="l-double">
<div>
![](img/fixed_increment_09.svg){width=70%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a}\$
<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{y}\^{k}\\in\\mathcal{Y}\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\mathbf{y}\^{k}\$ </li>
</ul>
<li>Repeat until \$|\\mathcal{Y}=\\emptyset\$</li>
</ol>
</div>
</div>

## Fixed-Increment Single-Sample Perceptron

This is a very simple approach. Geometrically we can prove that each move is
guaranteed to move the hyperplane in the right direction.

The system will converge **if the samples are linearly separable**, and the speed
with which it converges is dependent on the separation of the samples.

In some cases, this can take a very long time: if the points are lined up along
a plane parallel to the hyperplane, then the adjustment will be very small for
each misclassified sample.

## Variable Increment with Margins

We can add a **variable increment** \$\\eta\_{k}\$ and a **margin** \$b\$, so we
obtain misclassifications whenever the hyperplane fails to exceed the margin and
update a variable amount.

In this case the update rule is:

\$\\begin{array}{cl}
a\_{1} \& \\quad\\text{arbitrary} \\\\
a\_{k+1} = a\_{k} + \\eta\_{k}\\mathbf{y}\^{k} \& \\quad k\\geq 1\\\\
\\end{array}\$

And now \$\\mathbf{a}\_{k}\^{T}\\mathbf{y}\^{k}\\leq b\$ for all \$k\$.

## Variable-Increment Perceptron with Margin

<div class="txt-left">
Set \$k=0\$ and initialize \$\\mathbf{a},\\theta,b,\\eta(\\cdot)\$

<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{a}\^{T}\\mathbf{y}\^{k}\\leq b\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} + \\eta\_{k}\\mathbf{y}\^{k}\$ </li>
</ul>
<li>Repeat until \$\\mathbf{a}\^{T}\\mathbf{y}\^{k}>b\$ for all \$k\$</li>
</ol>
</div>

## Convergence

We know the samples are linearly separable if:

\$ \\eta\_{k}\\geq 0 \$

\$ \\lim\_{m\\rightarrow\\infty}\\sum\_{k=1}\^{m}\\eta\_{k}=\\infty \$

\$ \\lim\_{m\\rightarrow\\infty}\\frac{\\sum\_{k=1}\^{m}\\eta\_{k}\^{2}}{(\\sum\_{k=1}\^{m}\\eta\_{k})\^{2}} = 0 \$

Then \$\\mathbf{a}\_{k}\$ converges to a solution vector satisfying
\$\\mathbf{a}\^{T}\\mathbf{y}\_{i} > b\$ for all \$i\$.

These conditions are satisfied if \$\\eta\_{k}\$ is a positive constant OR if it
decreases as \$\\frac{1}{k}\$.

# 
## Relaxation Procedures

## Generalization to Other Functions

We specified a Perceptron criterion to be one that is easily optimized.

What if we are dealing with a criterion function for which second-order
optimization is NOT possible?

Then we generalize the minimization approach via **relaxation procedures**.

In addition to \$J\_{p}(\\mathbf{a})\$, we can define a similar criterion:

\$ J\_{q}(\\mathbf{a}) = \\sum\_{\\mathbf{y}\\in\\mathcal{Y}}(\\mathbf{a}\^{T}\\mathbf{y})\^{2} \$

This is the **squared error criterion**.

## \$J\_{q}(\\mathbf{a})\$: Squared Error Criterion

![Squared Error Criterion](img/squared_error_criterion_function.svg){width=50%}

## Perceptron Criterion vs. Squared Error

<div class="l-double">
<div>
![PCF: \$J\_{p}(\\mathbf{a}) = \\sum\_{\\mathbf{y}\\in\\mathcal{Y}}(-\\mathbf{a}\^{T}\\mathbf{y})\$](img/perceptron_criterion_function.svg){width=100%}
</div>
<div>
![SEC: \$J\_{q}(\\mathbf{a}) = \\sum\_{\\mathbf{y}\\in\\mathcal{Y}}(\\mathbf{a}\^{T}\\mathbf{y})\^{2}\$](img/squared_error_criterion_function.svg){width=100%}
</div>
</div>

## Modification of Squared Error Criterion

Both the perceptron criterion and the squared error criterion rely on
misclassified samples.

The gradient of \$J\_{p}(\\mathbf{a})\$ is not smooth, while \$J\_{q}(\\mathbf{a})\$ is,
providing an easier search space.

However, there are still some problems:

- It might converge to a point on the boundary, e.g. \$\\mathbf{a} = \\mathbf{0}\$
  (a "degenerate" solution)
- The value can be dominated by the longest sample vectors (e.g. outliers).

To avoid these, we use a modified **batch relaxation with margin** criterion:

\$ J\_{r}(\\mathbf{a}) = \\frac{1}{2} \\sum\_{\\mathbf{y}\\in\\mathcal{Y}} \\frac{(\\mathbf{a}\^{T}\\mathbf{y}-b)\^{2}}{||\\mathbf{y}||\^{2}} \$

## \$J\_{r}(\\mathbf{a})\$: Batch Relaxation with Margin

![Batch Relaxation with Margin](img/batch_relaxation_margin_criterion_function.svg){width=50%}

## Batch Relaxation with Margin

\$ J\_{r}(\\mathbf{a}) = \\frac{1}{2}\\sum\_{\\mathbf{y}\\in\\mathcal{Y}}\\frac{(\\mathbf{a}\^{T}\\mathbf{y}-b)\^{2}}{||\\mathbf{y}||\^{2}} \$

In this case, \$\\mathcal{Y}\$ is the set of samples for which
\$\\mathbf{a}\^{T}\\mathbf{y} \\leq b\$ (they are misclassified).

\$J\_{r}(\\mathbf{a})\$ is never negative, and is zero only when
\$\\mathbf{a}\^{T}\\mathbf{y} > b\$ (all training samples are correctly classified).

## Batch Relaxation with Margin

The gradient of \$J\_{r}\$ is given by:

\$ \\Delta J\_{r} = \\sum\_{\\mathbf{y}\\in\\mathcal{Y}}\\frac{\\mathbf{a}\^{T}\\mathbf{y} - b}{||\\mathbf{y}||\^{2}}\\mathbf{y}\$

And the update rule is:

\$\\begin{array}{cl}
a\_{1} \& \quad\\text{arbitrary} \\\\
a\_{k+1} = a\_{k} + \\eta\_{k}\\sum\_{\\mathbf{y}\\in\\mathcal{Y}} \\frac{b-\\mathbf{a}\^{T}\\mathbf{y}}{||\\mathbf{y}||\^{2}}\\mathbf{y} \& k\\geq 1 \\\\
\\end{array}\$

## Single-Sample Relaxation with Margin

Just like we did previously, we can simplify this to the case where we have a
constant learning rate, and adjust the weight vector on a sample-by-sample
basis.

This leads us to the single-sample relaxation with margin algorithm:

<div class="txt-left">
Set \$k=0\$ and initialize \$\\mathbf{a},\\eta(\\cdot)\$

<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{a}\^{T}\\mathbf{y}\^{k}\\leq b\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} +
\\eta\_{k}\\frac{b-\\mathbf{a}\^{T}\\mathbf{y}\^{k}}{||\\mathbf{y}\^{k}||\^{2}} \$ </li>
</ul>
<li>Repeat until \$\\mathbf{a}\^{T}\\mathbf{y}\^{k}>b\$ for all \$k\$</li>
</ol>
</div>

## Single-Sample Relaxation with Margin

<div class="l-double">
<div>
![](img/single_sample_margin_blue.png){width=100%}
</div>
<div>
Set \$k=0\$ and initialize \$\\mathbf{a},\\eta(\\cdot)\$

<ol>
<li>\$k \\leftarrow k+1\$</li>
<li>if \$\\mathbf{a}\^{T}\\mathbf{y}\^{k}\\leq b\$ then:</li>
<ul>
<li>\$\\mathbf{a}\\leftarrow\\mathbf{a} +
\\eta\_{k}\\frac{b-\\mathbf{a}\^{T}\\mathbf{y}\^{k}}{||\\mathbf{y}\^{k}||\^{2}}\\mathbf{y}\^{k} \$ </li>
</ul>
<li>Repeat until \$\\mathbf{a}\^{T}\\mathbf{y}\^{k}>b\$ for all \$\\mathbf{y}\^{k}\$</li>
</ol>
</div>
</div>

## Single-Sample Relaxation with Margin

<div class="l-double">
<div>
![](img/single_sample_margin_blue.png){width=100%}
</div>
<div>
The quantity:

\$ r\_{k} = \\frac{b-\\mathbf{a}\_{k}\^{T}\\mathbf{y}\^{k}}{||\\mathbf{y}\^{k}||} \$

is the distance from \$\\mathbf{a}\_{k}\$ to \$\\mathbf{a}\^{T}\\mathbf{y}\^{k} = b\$.

The unit normal vector for the hyperplane is
\$\\frac{\\mathbf{y}\^{k}}{||\\mathbf{y}\^{k}||}\$, meaning that:

\$ \\eta\_{k} (b-\\mathbf{a}\^{T}\\mathbf{y}\^{k}) \\frac{\\mathbf{y}\^{k}}{||\\mathbf{y}\^{k}||\^{2}} \$

is a fraction of the distance to the hyperplane.

</div>
</div>

## Single-Sample Relaxation with Margin

<div class="l-double">
<div>
![](img/relaxation.png){width=100%}
</div>
<div>
Update amount: \$\\eta(b-\\mathbf{a}\_{k}\^{T}\\mathbf{y}\^{k})\\frac{\\mathbf{y}\^{k}}{||\\mathbf{y}\^{k}||\^{2}}\$, or \$\\eta r\$ in the direction of the hyperplane.

Thus:

- If \$\\eta = 1\$, then \$\\mathbf{a}\_{k+1}\^{T}\\mathbf{y}\^{k} = b\$.
- If \$\\eta < 1\$, then \$\\mathbf{a}\_{k+1}\^{T}\\mathbf{y}\^{k} < b\$ (under-relaxation).
- If \$\\eta > 1\$, then \$\\mathbf{a}\_{k+1}\^{T}\\mathbf{y}\^{k} > b\$ (over-relaxation).

</div>
</div>

## Single-Sample Relaxation with Margin

Each update moves \$\\mathbf{a}\_{k}\$ a fraction, \$\\eta\$, of the distance towards
the hyperplane.

If \$\\eta = 1\$, \$\\mathbf{a}\_{k}\$ is moved exactly TO the hyperplane, thus
"relaxing" the tension created by the inequality
\$\\mathbf{a}\_{k}\^{T}\\mathbf{y}\^{k} \\leq b\$.

The single-sample relaxation with margin approach is guaranteed to find a
solution, although again, sometimes the point will lie on the boundary.

The value of \$\\eta\$ determines how fast we converge:

- If \$\\eta\$ is too small \$(<1)\$, convergence will take needlessly long
  (under-relaxation)
- If \$\\eta\$ is too large \$(1<\\eta<2)\$, we will over-shoot the solution
  (over-relaxation)

## Under- and Over-Relaxation

![Under- vs. Over-Relaxation](img/relaxation_comparison.svg){width=70%}

## Consequences of Setting Values of \$\\eta\$

Why not set \$\\eta=1\$ all the time?

- There may be several thousand training samples, and so setting \$\\eta > 1\$
  **might** help us converge faster.
- Our criterion function \$J\_{r}(\\mathbf{a})\$ may be irregular, so setting $\\eta
  < 1$ will ensure we don't miss the global optima.

Generally, you will want to set \$0 < \\eta < 2\$, but this is usually tuned to the
particular application.

# 
## Minimum Squared Error

## Sample Usage and Simplifying Equations

So far, we have considered \$\\mathcal{Y}\$, the set of misclassified samples.

Recall the main drawback of cross-validation...

We want to use ALL samples to train, not just a fraction of them!

Instead of trying to find \$\\mathbf{a}\$ such that \$\\mathbf{a}\^{T}\\mathbf{y}\_{i} >
0\$, we will now try to make \$\\mathbf{a}\^{T}\\mathbf{y}\_{i}=b\_{i}\$, where \$b\_{i}\$
are arbitrary positive constants.

Instead of trying to solve linear **inequalities**, we are now solving linear
**equations** -- which is a better-understood problem.

## Solving Linear Equations

Let's convert to matrix notation. Let \$\\mathbf{Y}\$ be the \$n\$-by-\$\\widehat{d}\$
matrix (\$\\widehat{d} = d + 1\$), where the \$i\$-th row is \$\\mathbf{y}\_{i}\^{T}\$.

Also, let \$\\mathbf{b}=(b\_{1}, \\ldots, b\_{n})\^{T}\$.

Then we want to find the weight vector \$\\mathbf{a}\$ such that:

\$\\begin{pmatrix}
y\_{10} \& y\_{11} \& \\cdots \& y\_{1\\widehat{d}}\\\\
\\vdots \& \\vdots \& \\ddots \& \\vdots\\\\
y\_{n0} \& y\_{n1} \& \\cdots \& y\_{n\\widehat{d}}
\\end{pmatrix}
\\begin{pmatrix}
a\_{0}\\\\
a\_{1}\\\\
\\vdots\\\\
a\_{\\widehat{d}}
\\end{pmatrix}=
\\begin{pmatrix}
b\_{1}\\\\
b\_{2}\\\\
\\vdots\\\\
b\_{n}
\\end{pmatrix}\$

\$\\mathbf{Y}\\mathbf{a} = \\mathbf{b}\$

## Minimizing Error

\$ \\mathbf{Y}\\mathbf{a} = \\mathbf{b}\$

We want \$\\mathbf{a}\$, so can't we just divide everything by \$\\mathbf{Y}\$?

Recall that the inverse of a matrix, \$\\mathbf{Y}\^{-1}\$, can only be written for
nonsingular, **square** matrices.

But \$\\mathbf{Y}\$ is an \$n\$-by-\$\\widehat{d}\$ matrix, and \$n\\neq\\widehat{d}\$.

Since we want to solve \$\\mathbf{Y}\\mathbf{a} = \\mathbf{b}\$, the optimal
\$\\mathbf{a}\$ would be the one for which \$\\mathbf{Ya-b=0}\$.

So let's define the error vector as:

\$ \\mathbf{e=Ya-b}\$

Now we have a function we can try to minimize!

## Sum-of-Squared-Error Criterion Function

Thus, we have our new criterion function:

\$ J\_{s}(\\mathbf{a}) = ||\\mathbf{Ya-b}||\^{2} = \\sum\_{i=1}\^{n}(\\mathbf{a}\^{T}\\mathbf{y}\_{i}-b\_{i})\^{2} \$

This is the **sum-of-squared-error** criterion, which is a traditional
optimization problem.

As before, we find the gradient as:

\$ \\Delta J\_{s} = \\sum\_{i=1}\^{n} 2 (\\mathbf{a}\^{T}\\mathbf{y}\_{i} - b\_{i})\\mathbf{y}\_{i} = 2\\mathbf{Y}\^{T}(\\mathbf{Ya-b}) \$

When our error function is zero, \$\\mathbf{e=Ya-b=0}\$, which means that both our
criterion and our gradient are zero.

## Sum-of-Squared-Error Criterion Function

Setting \$\\Delta J\_{s}\$ to zero yields the **necessary condition**:

\$ \\mathbf{Y}\^{T}\\mathbf{Ya}=\\mathbf{Y}\^{T}\\mathbf{b} \$

which replaces the original problem and allows us to find \$\\mathbf{a}\$ much more
easily.

## Solving for \$\\mathbf{a}\$

Why is it easier to find \$\\mathbf{a}\$ with \$\\mathbf{Y}\^{T}\\mathbf{Ya} =
\\mathbf{Y}\^{T}\\mathbf{b}\$ versus \$\\mathbf{Ya=b}\$?

The matrix \$\\mathbf{Y}\^{T}\\mathbf{Y}\$ is square and **often** nonsingular.

Thus, we know an inverse matrix \$(\\mathbf{Y}\^{T}\\mathbf{Y})\^{-1}\$ **does**
exist, so we can solve for \$\\mathbf{a}\$:

\\begin{align}
\\mathbf{a} \&= (\\mathbf{Y}\^{T}\\mathbf{Y})\^{-1}\\mathbf{Y}\^{T}\\mathbf{b} \\\\
\\mathbf{a} \&= \\mathbf{Y}\^{\\dagger}\\mathbf{b}
\\end{align}

Here, \$\\mathbf{Y}\^{\\dagger}\$ is an \$n\$-by-\$\\widehat{d}\$ matrix
\$\\mathbf{Y}\^{\\dagger}\\equiv(\\mathbf{Y}\^{T}\\mathbf{Y})\^{-1}\\mathbf{Y}\^{T}\$ called
the **pseudoinverse**. If \$\\mathbf{Y}\$ is square and nonsingular,
\$\\mathbf{Y}\^{\\dagger}\$ coincides with \$\\mathbf{Y}\^{-1}\$.

Thus, a **minimum squared error (MSE)** solution exists, and \$\\mathbf{a} =
\\mathbf{Y}\^{\\dagger}\\mathbf{b}\$ is an MSE solution to \$\\mathbf{Ya=b}\$.

## Some Properties of the MSE Solution

The MSE solution \$\\mathbf{a}=\\mathbf{Y}\^{\dagger}\\mathbf{b}\$ clearly depends on
the margin vector \$\\mathbf{b}\$, which we haven't defined yet. We can actually
choose different \$\\mathbf{b}\$ and get solutions with different properties.

Let's look at an example...

## Example MSE Solution

<div class="l-double">
<div>
![](img/mse01.png){width=100%}
</div>
<div>
Set up our matrix \$\\mathbf{Y}\$:

\$\\mathbf{Y}=
\\begin{pmatrix}
1  \& 1  \&  2 \\\\
1  \& 2  \&  0 \\\\
-1 \& -3 \& -1 \\\\
-1 \& -2 \& -3
\\end{pmatrix}\$

Class labels (+1 for \$\\omega\_{1}\$, -1 for \$\\omega\_{2}\$)

Solve for the pseudoinverse:

\$\\mathbf{Y}\^{\\dagger}=
\\begin{pmatrix}
\\frac{5}{4} \& \\frac{13}{12} \& \\frac{3}{4} \& \\frac{7}{12} \\\\
-\\frac{1}{2} \& -\\frac{1}{6} \& -\\frac{1}{2} \& -\\frac{1}{6} \\\\
0 \& -\\frac{1}{3} \& 0 \& -\\frac{1}{3}
\\end{pmatrix}\$

</div>
</div>

## Example MSE Solution

<div class="l-double">
<div>
![](img/mse02.png){width=100%}
</div>
<div>
\$\\mathbf{Y}\^{\\dagger}=
\\begin{pmatrix}
\\frac{5}{4} \& \\frac{13}{12} \& \\frac{3}{4} \& \\frac{7}{12} \\\\
-\\frac{1}{2} \& -\\frac{1}{6} \& -\\frac{1}{2} \& -\\frac{1}{6} \\\\
0 \& -\\frac{1}{3} \& 0 \& -\\frac{1}{3}
\\end{pmatrix}\$

Arbitrarily, we let all margins be equal, i.e. \$\\mathbf{b}=(1,1,1,1)\^{T}\$, so
our solution is:

\$ \\mathbf{a}=\\mathbf{Y}\^{\dagger}\\mathbf{b}=\\left(\\frac{11}{3}, -\\frac{4}{3}, -\\frac{2}{3}\\right)\^{T} \$

This leads to the decision boundary on the left.

</div>
</div>

## Sometimes We Need Something Else

One of the conditions for computing the pseudoinverse \$\\mathbf{Y}\^{\dagger}\$ are
that the matrix \$\\mathbf{Y}\^{T}\\mathbf{Y}\$ be nonsingular. If it **is**
singular, we can't compute the pseudoinverse.

Also, if we have a lot of training samples (which is common), then computing
\$\\mathbf{Y}\^{T}\\mathbf{Y}\$ is very difficult.

In these cases, we can avoid the pseudoinverse by going back to gradient descent
and using that to obtain our optimized criterion function \$J\_{s}(\\mathbf{a}) =
||\\mathbf{Ya-b}||\^{2}\$.

## Least Mean Squared Procedure

Recall the gradient \$\\Delta J\_{s} = 2 \\mathbf{Y}\^{T}(\\mathbf{Ya-b})\$ and set our
update rule:

\$ \\mathbf{a}\_{k+1} = \\mathbf{a}\_{k} + \\eta\_{k}\\mathbf{Y}\^{T}(\\mathbf{Y}\\mathbf{a}\_{k} - \\mathbf{b}) \$

We can consider samples sequentially and use the **LMS rule**:

\$ \\mathbf{a}\_{k+1} = \\mathbf{a}\_{k} + \\eta\_{k}(b\_{k} - \\mathbf{a}\_{k}\^{T}\\mathbf{y}\^{k})\\mathbf{y}\^{k} \$

## Least Mean Squares Procedure

\$ \\mathbf{a}\_{k+1} = \\mathbf{a}\_{k} + \\eta\_{k}(b\_{k} - \\mathbf{a}\_{k}\^{T}\\mathbf{y}\^{k})\\mathbf{y}\^{k} \$

Compare this with the relaxation rule from before:

\$ \\mathbf{a}\_{k+1} = \\mathbf{a}\_{k} + \\eta \\frac{ b - \\mathbf{a}\_{k}\^{T} \\mathbf{y}\^{k} } { ||\\mathbf{y}\^{k}||\^{2} } \\mathbf{y}\^{k} \$

What are the differences?

- Relaxation is an error-correction rule, so corrections will continue until all
  samples are classified.
- LMS is NOT concerned with perfect error-correction, so the hyperplane may NOT
  separate all the samples perfectly.
- Thus, LMS can be used when samples are **not linearly separable**.

## Minimum Squared Error vs. Least Mean Squares

![Least Mean Squares Allows for Misclassifications](img/lms.png){width=50%}

# 
## Support Vector Machines

## High-Dimensional Mapping Kernels

Recall how we replaced \$g(\\mathbf{x}) = \\mathbf{w}\^{T}\\mathbf{x} + w\_{0}\$ with
\$g(\\mathbf{x}) = \\mathbf{a}\^{T}\\mathbf{y}\$, where \$\\mathbf{a}\^{T}\$ is a
\$\\widehat{d}\$-dimensional weight vector and the \$\\widehat{d}\$ functions
\$y\_{i}(\\mathbf{x})\$ are arbitrary functions of \$\\mathbf{x}\$.

## Projections to Higher Dimensions

<div class="l-double">
<div>
![Polynomial Discriminant Function](img/polynomial_disc_func.svg){width=100%}
</div>
<div>
![Complex Discriminant Function](img/polynomial_complex_disc_func.png){width=100%}
</div>
</div>

## Support Vector Machine Setup

Let's modify the notation a bit, and say that each sample or pattern
\$\\mathbf{x}\_{k}\$ is transformed to \$\\mathbf{y}\_{k}=\\varphi(\\mathbf{x}\_{k})\$,
where \$\\varphi(\\cdot)\$ is a nonlinear mapping into "sufficiently" high
dimension.

For each of the \$n\$ patterns, \$k=1,2,\\ldots,n\$, we let \$z\_{k}=\\pm 1\$, where the
sign indicates whether the pattern is in \$\\omega\_{1}\$ or \$\\omega\_{2}\$.

The linear discriminant in this space is:

\$ g(\\mathbf{y}) = \\mathbf{a}\^{T}\\mathbf{y} \$

where the weight and transformed pattern vectors are augmented (\$a\_{0}=w\_{0}\$
and \$y\_{0}=1\$).

## Separating Hyperplane Properties

A separating hyperplane ensures:

\$ z\_{k}g(\\mathbf{y}\_{k}) \\geq 1, k=1,\\ldots,n \$

Remember: the sign of \$g(\\mathbf{y}\_{k})\$ indicates the class. For a correctly
classified sample, \$g(\\mathbf{y}\_{k})\$ is positive (and \$z\_{k}=+1\$) if
\$\\mathbf{y}\$ belongs to \$\\omega\_{1}\$, and is negative (and \$z\_{k}=-1\$) if it
belongs to \$\\omega\_{2}\$.

## Maximizing the Margin

Previously, we set the margin to be \$b>0\$ and told our optimization functions to
find any solution within that margin. Now, we want to set the hyperplane such
that we **maximize** \$b\$.

Why does this make sense?

- **Generalization**!
- Hyperplanes close to the training samples are likely to misclassify subsequent
  testing data.

## Recall the Graphical Situation

![Discriminating Hyperplanes](img/discplane.png){width=40%}

## Distance to the Hyperplane (Margin)

\$ z\_{k}g(\\mathbf{y}\_{k}) \\geq 1, k=1,\\ldots,n \$

The distance from the transformed pattern \$\\mathbf{y}\$ to the hyperplane is
\$\\frac{|g(\\mathbf{y})|}{||\\mathbf{a}||}\$.

The above equation implies:

\$ \\frac{ z\_{k} g(\\mathbf{y}\_{k})} {||\\mathbf{a}||} \\geq b \$

Assuming that a positive margin exists.

## Maximizing the Margin

\$ \\frac{ z\_{k} g(\\mathbf{y}\_{k})} {||\\mathbf{a}||} \\geq b \$

Our goal is to find the vector \$\\mathbf{a}\$ that maximizes \$b\$.

To avoid problems with arbitrary scaling, we add an additional constraint:
\$b||\\mathbf{a}||=1\$, thus forcing us to minimize \$||\\mathbf{a}||\^{2}\$ as
we maximize \$b\$.

## The "Support Vector" Part

**Support vectors** are samples for which the transformed vectors
\$\\mathbf{y}\_{k}\$ represent \$z\_{k}g(\\mathbf{y}\_{k}) = 1\$.

Since our hyperplane satisfies \$z\_{k}g(\\mathbf{y}\_{k}) \\geq 1\$, the support
vectors minimize the numerator above -- they are closest to the hyperplane, and
moreover they are all the same distance away.

This also means that they are the "most interesting" from a training point of
view, and most difficult to classify correctly from a testing point of view.

## Support Vector Machine Diagram

![SVM Diagram](img/svm.svg){width=50%}

## SVM vs. Perceptron Training

Recall that when we trained the PCF, we looked at randomly misclassified
samples.

In an ideal case, with SVM, we look for the **worst-classified** samples --
the misclassified samples that are farthest away from the current hyperplane,
which constitute our support vectors.

## Selecting Our Kernels

We start out training by deciding on the form of our mapping functions or
**kernel**, \$\\varphi(\\cdot)\$.

Typically these are chosen either by the problem domain, or as arbitrary
polynomials, Gaussians, etc.

The dimensionality of the transformed space may be **arbitrarily high**,
though in practice is limited by computational resources.

## SVM Training

Start by recasting the minimization problem using Lagrange undetermined
multipliers.

From \$\\frac{z\_{k} g(\\mathbf{y}\_{k})}{||\\mathbf{a}||} \\geq b\$, and knowing
that we want to minimize \$||\\mathbf{a}||\$:

\$ L(\\mathbf{a}, \\boldsymbol{\\alpha}) = \\frac{1}{2}||\\mathbf{a}||\^{2} - \\sum\_{k=1}\^{n}\\alpha\_{k}\\left[z\_{k}\\mathbf{a}\^{T}\\mathbf{y}\_{k} - 1\\right] \$

## SVM Training

So we seek to minimize \$L(\\cdot)\$ with respect to \$\\mathbf{a}\$ and maximize it
with respect to the multipliers \$\\alpha\_{k}\\geq 0\$.

We reformulate the problem such that we only need to maximize according to
\$\\mathbf{a}\$:

\$ L(\\mathbf{a}) = \\sum\_{i=1}\^{n}\\alpha\_{k} - \\frac{1}{2}\\sum\_{k,j}\^{n}\\alpha\_{k}\\alpha\_{j}z\_{k}z\_{j}\\mathbf{y}\_{j}\^{T}\\mathbf{y}\_{k} \$

Subject to the constraints: \$\\sum\_{k=1}\^{n}z\_{k}\\alpha\_{k} = 0, \\alpha\_{k}\\geq
0, k=1,\\ldots,n\$

## SVM Example: XOR

<div class="l-double">
<div>
![](img/xor01.png){width=100%}
</div>
<div>
This is the simplest problem that cannot be solved using a linear discriminant.

We can map these to a higher dimension using an expansion up to the second
order:

\$ (1, \\sqrt{2}x\_{1}, \\sqrt{2}x\_{2}, \\sqrt{2}x\_{1}x\_{2}, x\_{1}\^{2}, x\_{2}\^{2}) \$

where \$\\sqrt{2}\$ is chosen for convenient normalization.

Thus we translate each of the points to a six-dimensional space.

</div>
</div>

## SVM Example: XOR

<div class="l-double">
<div>
![](img/xor02.png){width=100%}
</div>
<div>
This is a two-dimensional representation of that projection, where we're looking at two out of the six dimensions.

The only thing that changed is the y-axis is now the \$x\_{1}\$ value times the \$x\_{2}\$ value (times \$\\sqrt{2}\$).

- \$\\mathbf{y}\_{1}\$ was at \$(1, 1)\$, so \$x\_{1}x\_{2} = 1\$
- \$\\mathbf{y}\_{2}\$ was at \$(1,-1)\$, so \$x\_{1}x\_{2} = -1\$
- \$\\mathbf{y}\_{3}\$ was at \$(-1,1)\$, so \$x\_{1}x\_{2} = -1\$
- \$\\mathbf{y}\_{4}\$ was at \$(-1,-1)\$, so \$x\_{1}x\_{2} = 1\$

</div>
</div>

## SVM Example: XOR

<div class="l-double">
<div>
![](img/xor03.png){width=100%}
</div>
<div>
![](img/xor04.png){width=100%}
</div>
</div>

## SVM Example: XOR

<div class="l-double">
<div>
![](img/xor04.png){width=100%}
</div>
<div>
\$ L(\\mathbf{a}) = \\sum\_{k=1}\^{4} \\alpha\_{k} - \\frac{1}{2}\\sum\_{k,j}\^{4}\\alpha\_{k}\\alpha\_{j}z\_{k}z\_{j}\\mathbf{y}\_{j}\^{T}\\mathbf{y}\_{k} \$

Subject to the constraints:

\$ \\alpha\_{1} - \\alpha\_{2} + \\alpha\_{3} - \\alpha\_{4} = 0 \$

\$ 0 \\leq \\alpha\_{k}, k=1,2,3,4 \$

So \$\\alpha\_{1}=\\alpha\_{3}\$ and \$\\alpha\_{2}=\\alpha\_{4}\$; thus \$\\alpha\_{k}\^{\\ast} = \\frac{1}{8}\$, and that all the samples are support vectors.

</div>
</div>

# 
## Summary

## Linear Discriminant Functions

This is a huge topic, but it provides a very solid foundation for decision
theory.

If your samples are linearly separable, you can very easily create a hyperplane
that achieves perfect classification.

Try to select your features such that you have a very good separation between
your training classes - this makes classification trivial!

For small problems, you can develop an analytic solution where you use the
pseudoinverse to calculate your weight vectors directly.

For larger problems, select an appropriate criterion function and solve using
gradient descent.

## What Was Covered?

Different cost functions to minimize:

- Perceptron Criterion: \$J\_{p}\$
- Squared Error Criterion: \$J\_{q}\$
- Batch Relaxation: \$J\_{r}\$

## What Was Covered?

How to "nudge" your parameter set:

- Iteration-based: \$\\eta\_{k} = \\frac{||\\Delta J||\^{2}}{\\Delta
  J\^{T}\\mathbf{H}\\Delta J}\$
- Constant: If \$J\$ is quadratic everywhere, then \$\\eta\_{k}\$ is the same for all
  \$k\$.
- Batches: Update \$\\mathbf{a}\$ using "batches" of misclassified samples.
- Relaxation: \$\\eta\_{k}\\sum\_{\\mathbf{y}\\in\\mathcal{Y}}
  \\frac{b-\\mathbf{a}\^{T}\\mathbf{y}}{||\\mathbf{y}||\^{2}}\\mathbf{y}\$

## What Was Covered?

When to stop your algorithm?

- Threshold: Stop when \$\\Delta J(\\mathbf{a}) < \\theta\$, where you choose \$\\theta\$.
- Batch Sample: Stop when \$\\mathcal{Y}=\emptyset\$, where \$\\mathcal{Y}\$ is the set of misclassified samples.
- Margin: Stop when \$\\mathbf{a}\^{T}\\mathbf{y}\^{k} > b\$.

## Support Vector Machines

For complex problems, you can transform your features into a high-dimensional
space and try to find a linearly separable set of dimensions for classification.

You must choose your mapping functions appropriately.

If you have non-convex solution regions, you may need to choose a different
classification method altogether.

# 
## Next Class

## Parameter Estimation

We will return to the world of Bayes and Gaussian distributions.

How do we estimate the values of the parameters we need from our training data?

- **Maximum Likelihood Parameter Estimation**, which assumes the parameters
  are fixed quantities that we just don't know.
- **Bayesian Parameter Estimation**, which assumes the parameters are random
  variables drawn from some kind of distribution.

You can use these methods in a large number of fields, not just for
classification -- this is a general statistical technique for investigating the
structure of your data!
