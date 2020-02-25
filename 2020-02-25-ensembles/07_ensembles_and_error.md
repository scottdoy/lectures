---
# YAML Preamble
fontsize: 10pt
documentclass: beamer
classoption:
- xcolor=table
- aspectratio=169
theme: metropolis
slide-level: 2

title: ENSEMBLES AND ERRORS
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
institute: scottdoy@buffalo.edu
date: Febraury 28, 2019
---

# Recap

## Recap: So Far, Two Approaches to Classification

\begincols

\column{0.5\linewidth}

![Bayesian Decision Theory](../imgs/03_probability_density_function.pdf){width=100%}

\column{0.5\linewidth}

![Decision Tree](../imgs/05_decisiontree_bin.pdf){width=100%}

\stopcols

## Recap: Choosing a Classifier

Good data (descriptive features, low noise, high $N$) means that ***any classifier
will do a decent job***. However, some are more suited to different situations than
others.

\begincols

\column{0.5\linewidth}

***Bayesian Decision Theory***

- Large $N$
- Metric (numeric) features
- Probabilistic classification

\column{0.5\linewidth}

***Decision Trees***

- Smaller (but still significant!) $N$
- Nonmetric (categorical) features
- Binary / All-or-nothing classification (except with Random Forests)

\stopcols

## Recap: Bayesian Decision Theory

Posterior probability:

$$ P(\omega_{j}|\mathbf{x}) = \frac{p(\mathbf{x}|\omega_{j})P(\omega_{j})}{p(\mathbf{x})}$$
$$ p(\mathbf{x}) = \sum_{j=1}^{c}p(\mathbf{x}|\omega_{j})P(\omega_{j}) $$

## Recap: Decision Trees

![Decision Tree](../imgs/05_decisiontree_complex.pdf)

## Recap: Bootstrap Aggregation (Bagging)

- Bootstrap aggregation ("bagging"): Multiple classifiers trained on ***sub-samples*** of training data
- The classification is some combination (average, weighted average, majority vote, etc.) of each individual classifier's guess
- This ***reduces variance*** (sensitivity to training data)

## Recap: Bagging

\definecolor{cmap1}{HTML}{1f77b4}
\definecolor{cmap2}{HTML}{ff7f0e}
\centering
\begin{tikzpicture}[every node/.style={inner sep=0cm, node distance=.25, thick, opacity=0.5, text opacity=1}]
    \node (D) [circle, fill=cmap1, draw=black, text centered, anchor=north, minimum size=1.5cm]{
        $\mathcal{D}$
    };
    \node (d3) [below=of D, circle, fill=cmap1, draw=black, rounded corners, text centered, yshift=-.25cm, minimum size=1cm]{
        $\mathcal{D}_{3}$
    };
    \node (d2) [left=of d3, circle, fill=cmap1, draw=black, rounded corners, text centered, minimum size=1cm]{
        $\mathcal{D}_{2}$
    };
    \node (d1) [left=of d2, circle, fill=cmap1, draw=black, rounded corners, text centered, minimum size=1cm]{
        $\mathcal{D}_{1}$
    };
    \node (dots) [right=of d3, rectangle, rounded corners, text centered, minimum size=1cm]{
        $\cdots$
    };
    \node (dm) [right=of dots, circle, fill=cmap1, draw=black, rounded corners, text centered, minimum size=1cm]{
        $\mathcal{D}_{m}$
    };

    \node (c1) [below=of d1, rectangle, fill=cmap2, draw=black, rounded corners, text centered, minimum size=1cm, yshift=-.25cm]{
        $\mathcal{C}_{1}$
    };
    \node (c2) [below=of d2, rectangle, fill=cmap2, draw=black, rounded corners, text centered, minimum size=1cm, yshift=-.25cm]{
        $\mathcal{C}_{2}$
    };
    \node (c3) [below=of d3, rectangle, fill=cmap2, draw=black, rounded corners, text centered, minimum size=1cm, yshift=-.25cm]{
        $\mathcal{C}_{3}$
    };
    \node (dots2) [right=of c3, rectangle, rounded corners, text centered, minimum size=1cm]{
        $\cdots$
    };
    \node (cm) [below=of dm, rectangle, fill=cmap2, draw=black, rounded corners, text centered, minimum size=1cm, yshift=-.25cm]{
        $\mathcal{C}_{m}$
    };

    \node (ca) [below=of c3, rectangle, fill=cmap2, draw=black, rounded corners, text centered, minimum size=1cm, yshift=-.25cm]{
        $\mathcal{C}_{A}$
    };

    \draw[<-, thick] (d1.north) -- +(0,0.25) -| (D.south);
    \draw[<-, thick] (d2.north) -- +(0,0.25) -| (D.south);
    \draw[<-, thick] (d3.north) -- +(0,0.25) -| (D.south);
    \draw[<-, thick] (dm.north) -- +(0,0.25) -| (D.south);
    \draw[<-, thick] (c1.north) to (d1.south);
    \draw[<-, thick] (c2.north) to (d2.south);
    \draw[<-, thick] (c3.north) to (d3.south);
    \draw[<-, thick] (cm.north) to (dm.south);
    \draw[->, thick] (c1.south) -- +(0,-0.25) -| (ca.north);
    \draw[->, thick] (c2.south) -- +(0,-0.25) -| (ca.north);
    \draw[->, thick] (c3.south) -- +(0,-0.25) -| (ca.north);
    \draw[->, thick] (cm.south) -- +(0,-0.25) -| (ca.north);
\end{tikzpicture}

# Classifier Ensembles

## Bagging

In bagging, training subsets of $\mathcal{D}$ are created by drawing $n^{\prime}<n$ samples with replacement.

Each ***component*** classifier casts a "vote" for the classification of a sample point, and the final classification is the result of this vote.

Each component classifier can be the same type, although the learned parameter values may vary (since they are created using different training sets).

This increases the ***stability*** of the final classifier by averaging over the differences incurred by using different training sets.

## Boosting

In ***boosting***, the goal is to increase the ***accuracy of the final classifier***.

In this case, we use component classifiers, each of which is the "most informative" out of all possible component classifiers.

So we are less concerned about reducing "noise" or increasing stability; here, we want to combine different aspects of the dataset (represented by classifiers) to reach the right decision.

## Boosting Example

***Bagging***: Giving a patient's file to 5 general practice doctors, and getting them to vote on the patient's disease.

***Boosting***: Getting a pathologist, radiologist, and geneticist to deliberate and decide on the patient's disease.

## Boosting

Imagine we have a two-category problem and we create three component classifiers $C_{1}, C_{2}, C_{3}$.

Each is trained on training sets $\mathcal{D}_{1}, \mathcal{D}_{2}, \mathcal{D}_{3}$.

However, in this case we select training samples such that half of the samples in $\mathcal{D}_{2}$ should be misclassified by $C_{1}$, and half correctly classified.

$\mathcal{D}_{3}$ is completely made up of samples for which $C_{1}$ and $C_{2}$ return different classification results.

## Boosting Example

![Training (top), Components (mid), Combined (bot)](../imgs/14_boosting_example.pdf){width=60%}

## Practical Considerations

How should we select samples for our first classifier, $C_{1}$?

We'd like to use all available data, so ideally $n_{1}\simeq n_{2}\simeq n_{3} \simeq \frac{n}{3}$.

However if the problem is too easy, then $C_{1}$ has a very high accuracy, and boosting does not help very much.

If the problem is too hard, then $C_{1}$ performs badly and $C_{2}$ will have to have too many samples.

In practice, several runs of boosting will be required to ensure that $\mathcal{D}$ is used in its entirety to train the component classifiers.

# Adaptive Boosting

## Variations on Boosting

***AdaBoost*** is a type of boosting, in which you design how many "weak learners" (component classifiers) you add to the system.

Training samples are given a ***weight*** indicating its probability of being selected for one of the component classifiers.

If a sample is misclassified by $C_{i}$ then it is "hard" to classify and its weight is increased (it is ***more*** likely to be included in $D_{i+1}$).

If the sample is correctly classified by $C_{i}$, then its weight is reduced (it is ***less*** likely to be selected for $D_{i+1}$).

If the samples and labels in $\mathcal{D}$ are denoted $\mathbf{x}^{i}$ and $y_{i}$, and $W_{k}(i)$ is the $k$th discrete distribution over the training, then the algorithm is...

## Adaptive Boosting

\begin{algorithm}[H]\footnotesize
    \emph{begin initialize $\mathcal{D}=\left\{(\mathbf{x}^{1},y_{1}),\cdots,(\mathbf{x}^{n},y_{n})\right\},k_{max},W_{1}(i)=1/n,i=1,\cdots,n$}\;
    $k\leftarrow 0$\;
    \Repeat{$k=k_{max}$}{
        $k\leftarrow k+1$\;
        train weak learner $C_{k}$ using $\mathcal{D}$ sampled according to $W_{k}(i)$\;
        $E_{k}\leftarrow$ training error of $C_{k}$ measured on $\mathcal{D}$ using $W_{k}(i)$\;
        $\alpha_{k}\leftarrow\frac{1}{2}\ln{\left[(1-E_{k})/E_{k}\right]}$\;
        $W_{k+1}(i)\leftarrow\frac{W_{k}(i)}{Z_{k}}\times\left\{
            \begin{array}{ll}
                e^{-\alpha_{k}} & \quad \text{if $h_{k}(\mathbf{x}^{i})=y_{i}$ (correctly  classified)}\\
                e^{\alpha_{k}} & \quad \text{if $h_{k}(\mathbf{x}^{i})\neq y_{i}$ (incorrectly classified)}\\
            \end{array}
            \right.$ \;
    }
    \KwRet{$C_{k}$ and $\alpha_{k}$ for $k=1$ to $k_{max}$ (ensemble of classifiers with weights)}
\end{algorithm}

## Explanation of AdaBoost Algorithm

$E_{k}$ is the error with respect to $W_{k}(i)$; this means that $0<E_{k}<1$.

Thus $\alpha_{k}=\frac{1}{2}\ln{\left[(1-E_{k})/E_{k}\right]}$ ranges from $-\infty$ when $E=1.0$ to $0$ when $E=0.5$ to $\infty$ when $E=0.0$.

## Explanation of AdaBoost Algorithm

If a component classifier has ***low error***, then $\alpha_{k}$ is ***high*** (positive).

- ***Incorrect*** samples are adjusted by a ***high*** amount ($e^{\alpha}$).
- ***Correct*** samples are adjusted by a ***low amount*** ($e^{-\alpha}$).

Translation: If the classifier is good, then errors are rare and we should try to add them to the training set.

## Explanation of AdaBoost Algorithm

If a component classifier has ***high error***, then $\alpha_{k}$ is ***low*** (negative).

- ***Incorrect*** samples are adjusted by a ***low*** amount ($e^{\alpha}$).
- ***Correct*** samples are adjusted by a ***high*** amount ($e^{-\alpha}$).

Translation: If the classifier is bad, the correct samples are more valuable, so they are weighted more.

## Alphas and Adjustments

![](../imgs/07_adaboost_correct.pdf){width=80%}

## Output of AdaBoost and Its Error

The final classification of a point $\mathbf{x}$ is based on a weighted sum of the outputs:

$$ g(\mathbf{x})=\left[\sum_{k=1}^{k_{max}}\alpha_{k}h_{k}(\mathbf{x})\right]$$

The magnitude of $g(\mathbf{x})$ is the result of the different weights assigned to the component classifiers (and their signs), while the classification result is simply the sign of $g(\mathbf{x})$.

How should we set our stopping criteria? That is, how many component classifiers should we collect?

## AdaBoost Error

The training error for $C_{k}$ can be written as $E_{k}=1/2-G_{k}$ for some $G_{k}>0$ (since we're using a distribution to modulate our training error).

Then the ensemble error is simply the product:

$$ E=\prod_{k=1}^{k_{max}}\left[2\sqrt{E_{k}(1-E_{k})}\right]=\prod_{k=1}^{k_{max}}\sqrt{1-4G_{k}^{2}}$$

Thus if we keep increasing $k_{max}$, by adding more component classifiers, our error on the training set should be arbitrarily low!

## Alphas and Adjustments

![Individual learners (grey), ensemble training (black), and ensemble testing (red).](../imgs/15_adaboost_error.pdf)

## Wait... "Arbitrarily Low"?

We usually get "arbitrarily low" error on the training set when we've massively over-fit the classifier, but it turns out this isn't usually the case with AdaBoost... so what's up?

- We rarely get arbitrarily low error on ***testing*** data, so generalization usually isn't perfect.
- Our component classifiers ***must do better than chance***!
- Our component classifiers ***must be (relatively) independent***!

You can't set $k_{max}$ arbitrarily high with all of these conditions.

Nonetheless, AdaBoost is a very powerful algorithm that has been used and modified in a number of different applications.

## Adaboost Example: Sklearn Dataset

\begincols

\column{1.0\linewidth}

![](../imgs/07_adaboost_sklearn_exdata.pdf){width=50%}

\stopcols

## Adaboost Example: Sklearn Dataset

\begincols

\column{1.0\linewidth}

![](../imgs/07_adaboost_sklearn_example.pdf){width=100%}

\stopcols

## Adaboost Example: FNA Dataset

\begincols

\column{1.0\linewidth}

![](../imgs/07_adaboost_sklearn_fnadata.pdf){width=50%}

\stopcols

## Adaboost Example: FNA Dataset

\begincols

\column{1.0\linewidth}

![](../imgs/07_adaboost_sklearn_fna.pdf){width=100%}

\stopcols

# Active Learning

## Learning with Queries / Active Learning

We are often presented with the issue of partially-labeled or
expensively-labeled data, somewhere between supervised and unsupervised.

In this case we can use an expert to label this data, but we want to maximize
its effectiveness -- thus we must choose which of the unlabeled samples is
***most informative***.

This approach is variously called learning with queries, active learning,
interactive learning, or cost-based learning. (I prefer "active learning",
although it overlaps with topics in education.)

## Active Learning Types

In ***confidence-based*** active learning, a pattern is informative if two discriminant functions are nearly equal: $g_{i}(\mathbf{x})\approx g_{j}(\mathbf{x})$.

In ***voting-*** or ***committee-based*** active learning, patterns are informative if component classifiers $C_{k}$ "disagree" on the class.

## Alphas and Adjustments

![](../imgs/15_active_learning.pdf){width=50%}

# Combining Classifiers

## Creating Component Classifiers

AdaBoost provides a way of combining classifiers, but there are others!

These are ***mixture-of-expert*** models, ensemble, modular, or pooled classifiers.

Assume each output is produced by a ***mixture model*** of $k$ component classifiers.

## Testing and Training Learning Graphs

![Architecture of the mixture-of-experts model. Each of the $k$ models has a parameter set $\boldsymbol{\theta}_{i}$. Each estimate of the category membership for a sample $\mathbf{x}$ is $g_{ir}=P(\omega_{r}|\mathbf{x},\boldsymbol{\theta}_{i})$, and the outputs are weighted by the gating subsystem.](../imgs/15_component_classifiers.pdf){width=40%}

## Designing a Component Classifier

How do we choose $k$?

- Use prior knowledge if you know how many processes contribute to a
  classification output.
- Use cross-validation or bias / variance tuning to estimate the number of
  optimal components.
- Just over-estimate, since having more components should generalize better than
  having fewer components.

As with everything, it's up to the designer.

The best thing to do is always to just test it out and see what works.

***However***, you should be ready to justify your choices somehow!

# Classifier Evaluation

## How Did You Do?

When designing a classifier, the main question is: What is the performance of the classifier?

There are several ways to answer this question...

## Errors in Classification

Recall that often (in the Bayes case, at least) we are given the probability that a feature vector belongs to a particular class: $p(\omega_{i}|\mathbf{x})$.

We then ***threshold*** that probability, based on our risk assessment. This gives us a "hard" classification.

## Errors in Classification

![](../imgs/03_cumulative_density_function_featval.pdf)

## Confusion Matrix

In the two-class case, we typically talk about the two classes as being "positive" or "negative".

So we have two possible classification outputs (Postivie or Negative), and each of those can be right or wrong. This gives us the following table, known as a ***Confusion Matrix***:

\begin{table}
    \begin{tabular}{cccc}
    \multicolumn{2}{c}{} & \multicolumn{2}{c}{Classifier Prediction}\\
    \cline{3-4}
    & & \multicolumn{1}{|c|}{Positive} & \multicolumn{1}{c|}{Negative} \\
    \cline{2-4}
    \multirow{2}{*}{Actual Class} & \multicolumn{1}{|c}{Positive} & \multicolumn{1}{|c|}{TP} & \multicolumn{1}{c|}{FN} \\
    \cline{2-4}
     & \multicolumn{1}{|c}{Negative} & \multicolumn{1}{|c|}{FP} & \multicolumn{1}{c|}{FN} \\
    \cline{2-4}
    \end{tabular}
\end{table}

## Evaluation Metrics

From the confusion matrix, you can calculate a lot of ***performance metrics***:

\begin{table}
    \renewcommand{\arraystretch}{1.2}
    \begin{tabular}{rl}
    \toprule
    Accuracy & $\frac{TP+TN}{TP+FP+TN+FN}$ \\
    Precision (Positive Predictive Value, PPV) & $\frac{TP}{TP + FP}$ \\
    Sensitivity (True Positive Rate, TPR) & $\frac{TP}{TP + FN} = \frac{TP}{P}$ \\
    Specificity (True Negative Rate, TNR) & $\frac{TN}{TN + FP} = \frac{TN}{N}$ \\
    Negative Predictive Value (NPV) & $\frac{TN}{TN + FN}$ \\
    F1 Score & $\frac{2TP}{2TP + FP + FN}$ \\
    \bottomrule
    \end{tabular}
\end{table}

Some of these are more appropriate to use than others, depending on your problem
setup. For example, what if your classes are massively imbalanced (90% vs. 10%)?

## Sensitivity and Specificity

This evaluation takes place ***after*** we've thresholded our probabilities, but
how certain are we in our classifier output? 

By setting our thresholds to extremes, we can achieve either high Sensitivity or
Specificity:

*High Sensitivity:*
: TP is close to P (all positive samples)
: You correctly identified all positive samples
: Maximize by ***calling everything positive***

*High Specificity:*
: TN is close to N (all negative samples)
: You correctly identified all negative samples
: Maximize by ***calling everything negative***

## Receiver Operating Characteristic Curves

Clearly we need a way to take the threshold of classification into account.

Enter ***Receiver Operating Characteristics (ROC)*** curves, which can quantify
this relationship. This is a plot of the True Positive Rate (Sensitivity)
against the False Positive Rate (1 - Specificity).

The area under the ROC curve (AUC) is a measurement of how well
the classifier performs in terms of the tradeoff between the two.

## ROC Example

![ROC Curve Examples](../imgs/14_roc_ver2.png){width=50%}

# Bias and Variance

## Determining Bias and Variance

Since bias and variance are unavoidable, it is necessary to compensate for them in your classifier.

To do this, we create a number of different training and testing samples through ***resampling***.

The goal of these techniques is to determine the ***generalization performance***, which will predict how well you'll do on real-world examples and not just the samples you have access to at the time.

## Training vs. Testing

In its simplest form, you just split $\mathcal{D}$ into two parts: a ***training*** and ***testing*** set.

You use your training set to create the classifier, and ***after*** training is done, you evaluate the performance on the test set.

You must be careful about training on your test set! This includes:

- Using your testing data to create model parameters directly; or
- Training to minimize test error on a ***single*** test set.

## Training vs. Validation vs. Testing

A more complex example is to also use a ***validation*** set, which you use to evaluate your model *while* you're training.

This way, you can be sure that you don't even look at your testing data until you are completely done creating your classifier.

## Validation Error vs. Training Error

![](../imgs/15_validation_error.pdf)

## Cross-Validation

How do you know which samples go into your training, testing, and validation sets? What if you have some samples that are outliers? 

A common way to handle this is through ***cross-validation***.

Let's say you have your training set $\mathcal{D}$. You want to know how your classifier changes based on which samples it trains on. 

- ***High Variance*** means that your classifier changes a lot depending on the training data, which means your model may be ***overfitting***. 
- ***Low Variance*** means that your classifier doesn't change much, which is good as long as you ALSO have low bias. 

## $k$-fold Cross-Validation

The basic strategy is:

1. Break your training set up randomly into $k$ subsets.
2. Select one subset to "hold out", and then train a classifier on the
   remaining $k-1$ subsets.
3. Evaluate your classifier against the "hold out" subset.
4. ***Repeat*** Steps 2-3 $k$ times, each time selecting a new subset to hold
   out. Each iteration is a ***round*** of cross-validation.
5. After $k$ iterations, each datapoint will have been classified once.
   Calculate the performance metric for this trial.
6. ***Repeat*** 1-5 for many ***trials***, each time randomly breaking the data up into $k$
   subsets (so that in each trial, there are different combinations of training
   data being used). 

## $k$-fold Cross-Validation

Some considerations:

- ***Class Distribution***: What should the class distribution look like for
   each of the $k$ subsets? Should they have equal numbers of classes, should
   the distribution match the overall dataset distribution (and be the same for
   each subset), or should we ignore classes and just randomly split up the
   data? 
- ***Folds***: What should our $k$ be? If we set it high, then each classifier
  will be trained on more samples; if it is low, then we get a larger testing
  set in each round. 

## Leave-One-Out Cross-Validation

In the extreme case, we can set $k = |\mathcal{D}|$. In that case, $k$ is equal
to the number of samples in our dataset.

This means that in each "round", we are just removing one sample, training on
the others, and then evaluating that one sample.

What are the advantages of this approach? Disadvantages?

## LOO Cross-Validation

***Advantages:***

- Uses as much data as possible for training
- Useful if you don't have much data, and $k$-fold would eat away too much of
  your training
- Allows you to see how well your model adapts to tiny changes in the training
  set (one sample at a time)
  
***Disadvantages:***

- Hard to see how groups of "outliers" are affecting your model, since they are
  always grouped with the rest of the dataset
- The model doesn't change much, so you may be overfitting to your 
  dataset

## Estimation of Accuracy

When we say that the "accuracy" of a system is 90%... how sure are we that it's not 89% or 91% or 70% instead?

Accuracy is a statistical estimation of performance, but as with any statistic, ***it too has a distribution***.

## Estimations of Accuracy

In addition to dividing up our data and doing $k$-fold cross-validation, we also want to calculate the ***mean*** and ***standard deviation*** of our performance metrics.

We can do this by simply randomly assigning samples to our training and validation sets, running $k$-fold cross-validation, and then re-assigning the samples and running again.

This will give a distribution of performance values, which can be reported.

Think of this as quality assurance!

## Comparing Classifiers Using Jackknife Estimates

![Jackknife estimates of two classifier accuracies. The accuracy is on the x-axis, the y-axis is the likelihood of seeing that accuracy (estimated from the $n$ training sets), and the bars represent the full widths (2x the square root of the jackknife estimate of the variances).](../imgs/15_leave_one_out_estimates.pdf)

## Predicting Performance from Learning Curves

In practice, $\mathcal{D}$ is far smaller than the (infinite) true dataset.

If you had 100 samples to work with, how can you decide which classifier is
"best" when you can't fully test them all?

***Learning curves*** plot test error against the size of the training set, and
are typically described by a power-law function of the form:

$$ E_{test} = a + \frac{b}{n^{\prime\alpha}} $$

$$ E_{train} = a - \frac{c}{n^{\prime\beta}} $$

Here, $a$ is the error with a theoretically infinite sample size; ideally, it is
equal to the Bayes error (optimum).

## Learning Rate Graph

![Test error for three classifiers. Note how, if we only had 500 samples, we might rank the classifiers differently than if we had 10,000 samples.](../imgs/15_learning_rate.pdf)

## Testing and Training Learning Graphs

![Test and training error on a fully-trained classifier. Training error is 0 at low $n^{\prime}$, since the classifier can perfectly classify those few points. As $n^{\prime}\rightarrow \infty$, both approach the same error rate; if the classifier is strong, then $a=E_{B}$ (the Bayes error).](../imgs/15_learning_rate_02.pdf)

## Finding the Error Rate

We can manipualate the training and testing error formulae to get:

$$ E_{test} + E_{train} = 2a+\frac{b}{n^{\prime\alpha}} - \frac{c}{n^{\prime\beta}} $$

$$ E_{test}-E_{train} = \frac{b}{n^{\prime\alpha}} + \frac{c}{n^{\prime\beta}} $$

If we assume $\alpha=\beta$ and $b=c$ (the learning rates of the testing and
training are the same), then:

$$ E_{test} + E_{train} = 2a $$

$$ E_{test} - E_{train} = \frac{2b}{n^{\prime\alpha}} $$

## Testing and Training Learning Graphs

![If we make a log plot, we can see that $E_{test}-E_{train}$ is a straight line, and the sum $s=b+c$ can be found from the height of the $\log{[E_{test}+E_{train}]}$ curve.](../imgs/15_power_law.pdf)

# Parting Words

## Classifier Evaluation is Important!

A lot of engineering  papers focus on "benchmark" datasets and evaluating new classifiers against others using a standard set of metrics.

Why is your method better than others? 

Performance evaluation allows us to compare methods quantitatively. 

***Error is not the only consideration!*** In some cases, reducing costs / time / resources may be just as important.

## Train / Test Splits

Bias, variance, and over-training must be accounted for.

Creating train / validation / testing splits is basic, standard practice for machine learning work.

If you had to sell your algorithm to someone who will use it on data you've never seen before, how confident are you that it will work well?

In some cases, even doing proper cross-validation ***isn't good enough!***

# Next Class

## Linear Discriminants

We will return to a basic classification techniques: ***Linear Discriminants***.

These are useful when your data is linearly separable, and is an introduction to more complex methods like ***support vector machines***.

These also form the basis for ***neural networks***, which we will discuss in the second half of the course.