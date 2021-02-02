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
  | RECURRENT NEURAL
  | NETWORKS
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
institute: scottdoy@buffalo.edu
date: April 18, 2019
---

# Announcements

## Next Tuesday: Assignment 3

Assignment 3 focuses on using ***unsupervised clustering*** to process your
data. In this case, your job is to create clusters of data without using labels,
and see if you can find groups that align with the classes you know are there
(from Assignment 2).

## Final Two Weeks of Classes

Starting ***April 30***, class will be held back in Cooke Hall, Room 127B until the end of the semester.

# Recurrent Neural Networks

## Outside Resources

Andrej Karpathy: "The Unreasonable Effectiveness of Recurrent Neural Networks" 2015 

[https://karpathy.github.io/2015/05/21/rnn-effectiveness/](https://karpathy.github.io/2015/05/21/rnn-effectiveness/)

Christopher Olah: "Understanding LSTM Networks", 2015 

[https://colah.github.io/posts/2015-08-Understanding-LSTMs/](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)

## Limitations of Neural Nets

CNNs were created to exploit structured input data, but have limitations:

- They accept a ***fixed-size input*** (the image);
- They produce a ***fixed-size output*** (classes or images);
- They operate using a ***fixed-size number of operations*** (layers);

This works if your data is fixed, but not all data is.

## Unfixed Data Examples

- Genomic sequences;
- Translation between languages;
- Time series data;
- Classifying variable input sequences

## Sequence Examples

\begincols

\column{0.8\linewidth}

***One-to-One***:

- One input, one output (class)
- Example:
  - Input: "Think"
  - Output: "Verb"

\column{0.2\linewidth}

![](../imgs/19_inputs_to_outputs_1_to_1.png){height=100%}

\stopcols

## Sequence Examples: One-to-Many

\begincols

\column{0.8\linewidth}

***One-to-Many***:

- Single input, variable-length output (sequence)
- Example: 
  - Input: "Hello"
  - Output: "Hello, how are you?"

\column{0.2\linewidth}

![](../imgs/19_inputs_to_outputs_1_to_many.png){height=100%}

\stopcols

## Sequence Examples: Many-to-One

\begincols

\column{0.8\linewidth}

***Many-to-One***:

- Variable-length input, single output (class)
- Example:
  - Input: "I didn't like this movie, it was terrible."
  - Output: "Negative"

\column{0.2\linewidth}

![](../imgs/19_inputs_to_outputs_many_to_1.png){height=100%}

\stopcols

## Sequence Examples: Many-to-Many (Delayed)

\begincols

\column{0.6\linewidth}

***Many-to-Many (Delayed)***:

- Variable-length input, variable-length output
- Example:
  - "Hello, my name is Scott."
  - "Hola, me llamo Scott."

\column{0.4\linewidth}

![](../imgs/19_inputs_to_outputs_many_to_many.png){height=100%}

\stopcols


## Sequence Examples: Many-to-Many (Synchronized)

\begincols

\column{0.8\linewidth}

***Many-to-Many (Synchronized)***:

- Variable-length input, synchronized output
- Example: 
  - Input: Video Sequence
  - Output: Real-time Tracking Locations

\column{0.2\linewidth}

![](../imgs/19_inputs_to_outputs_many_to_many_sync.png){height=100%}

\stopcols

## Modifications Needed for Sequence Processing

Sequence data processing requires ***memory***.

As you read this sentence, the words only make sense in the context of what's come before it. 
Even the words are pronounced based on the sequence of the letters, rather than an independent processing of each unit of input.

So there is a ***preserved state*** that is updated and changed based on the sequence of inputs.

## Diagram of Recurrent Neural Network

\begincols

\column{0.35\linewidth}

![Recurrent Neural Network Unit](../imgs/19_rnn_rolled.png){width=45%}

\column{0.65\linewidth}

- $x$ is the input, $o$ is the output.
- $s$ is the "hidden state" of the network, updated at each timestep.
- $\mathbf{W}$ is the weight of the hidden state.
- $\mathbf{U}$ and $\mathbf{V}$ are the weights of the input-to-hidden and hidden-to-output layers, respectively.
- Each time we "loop around" $\mathbf{W}$, we have another timestep in the sequence. 

So we can modify this architecture by "unrolling" it to give us a similar, traditional neural network architecture.

\stopcols

## Diagram of Recurrent Neural Network

![Unrolled Recurrent Network](../imgs/19_rnn.jpg){width=100%}

## Explanation of RNN Calculations

\begincols

\column{0.5\linewidth}

![Unrolled Recurrent Network](../imgs/19_rnn_unrolled.png){width=100%}

\column{0.5\linewidth}

- $x_{t}$ is the input at time step $t$ (e.g. the $t$-th word in a sentence).
- $s_{t}$ is the "hidden state" at $t$, where: $$ s_{t} = f(\mathbf{U}x_{t} + \mathbf{W}s_{t-1})$$ and where $f(\cdot)$ is a nonlinearity.
- $o_{t}$ is the output at step $t$. If we're predicting the next word in a sentence, it is a vector of probabilities across all possible words in a vocabulary.

\stopcols

## Defining Terms

$s_{t}$ is the "memory" of the network, capturing information at all previous timesteps. The output $o_{t}$ is computed based on the memory at $t$.

Traditional deep networks use different parameters at each layer, but RNNs share parameters $\mathbf{U}, \mathbf{V}, \mathbf{W}$ across all time steps. 

This means that we don't have to train a unique network on every single input (which would be overfitting).

Outputs depend on the purpose: Many-to-one networks may only be interested in the final output.

# RNNs for NLP

## Text Generation

Simple problem: Given a sequence of words, can we predict the most likely next word?

- Input: Sequence of words
- Output: Next predicted word

This is a ***Many-to-One*** problem, or a ***Many-to-Many (Delayed)*** problem if we want to generate a sequence of words from a "starting" sequence.

## First: How Do We Represent Text?

In images, an RGB pixel with values [0, 255, 0] (green) is more similar to a pixel [0, 250, 0] (slightly darker green) than to a pixel [255, 0, 0] (red). 

With language, this doesn't hold. 

***Car[d]*** vs. ***Car[e]***: "d" and "e" are right next to each other in the alphabet, but these two words have nothing in common.

So we want to translate words into a vector representation of some kind that can give us greater descriptive power over what the words mean (and how they might be related to one another).

## Types of Word Representations

***Vector space models*** (VSMs): Embed words into a continuous vector space, where distance is proportional to similarity in meaning.

One method is to represent words in a sparse, high-dimensional form called a ***one-hot*** vector, where each word is a vector with dimensionality equal to the vocabulary size.

The vector is zeros everywhere except the location that represents the word of interest.

## One-Hot Vector Illustration

![Image from \href{http://veredshwartz.blogspot.co.il/2016/01/representing-words.html}{Vered Shwartz (http://veredshwartz.blogspot.co.il/2016/01/representing-words.html)}](../imgs/19_one_hot_vector.png){width=100%}

## Alternate Representations

Problems: Large vocabularies lead to large, very sparse dimensional spaces (most of the elements are zero). 

Alternative representations of words, such as ***word2vec***, attempt to create a vector representation of a word based on its use in actual language (how often it appears next to other words or in different contexts).

If you are interested in word embeddings and search-and-retrieval, it's worth looking at the ***[Tensorflow Tutorial on word2vec: https://www.tensorflow.org/tutorials/word2vec](https://www.tensorflow.org/tutorials/word2vec)***

# Training RNNs

## Training Sequence Data

Training is done through  ***Backpropagation Through Time*** (BPP): Since parameters are shared by all time steps in the network, we have to take into account all previous timesteps in order to calculate the gradient at time $t$.

## Defining Terms

Our hidden state (memory) and the output of a single unit are, respectively:

\begincols

\column{0.5\linewidth}

![Unrolled Recurrent Network](../imgs/19_rnn_unrolled.png){width=80%}

\column{0.5\linewidth}

\begin{align*}
s_{t} &= \tanh(\mathbf{U}x_{t} + \mathbf{W}s_{t-1}) \\
o_{t} &= \textrm{softmax}(\mathbf{V}s_{t})
\end{align*}

\stopcols

## Defining Error

Gradient descent and backpropagation require an error function. If $\hat{o}$ is the "correct" output, then:

\begin{align*}
E_{t}(\hat{o}_{t},o_{t}) &= -\hat{o}_{t}\log o_{t} \\
E(\hat{o},o) &= \sum_{t} E_{t}(\hat{o}_{t}, o_{t}) \\
&= -\sum_{t} \hat{o}_{t}\log o_{t}
\end{align*}

$\hat{o}_{t}$ is the correct word at time step $t$, and $o_{t}$ is our prediction. This is the ***cross-entropy loss function***, and is calculated over all timesteps (since we typically treat one full sentence as a training exmaple).

## Summing Error Partials

Since we're learning the gradient of the error with respect to the parameters, we can sum over the timesteps for each of $\mathbf{U}, \mathbf{V}, \mathbf{W}$:

\begin{align*}
\frac{\partial E}{\partial \mathbf{V}} &= \sum_{t}\frac{\partial E_{t}}{\partial
\mathbf{V}} \\
\frac{\partial E}{\partial \mathbf{U}} &= \sum_{t}\frac{\partial E_{t}}{\partial
\mathbf{U}} \\
\frac{\partial E}{\partial \mathbf{W}} &= \sum_{t}\frac{\partial E_{t}}{\partial
\mathbf{W}} \\
\end{align*}

## Calculating Backpropagation: $\mathbf{V}$

We calculate the chain rule working backwards from the output $o_{t}$. So starting with $\mathbf{V}$:

\begincols

\column{0.5\linewidth}

![Unrolled Recurrent Network](../imgs/19_rnn_unrolled.png){width=80%}

\column{0.5\linewidth}

\begin{align*}
\frac{\partial E_{t}}{\partial \mathbf{V}} &= \frac{\partial
E_{t}}{\partial o_{t}}\frac{\partial o_{t}}{\partial\mathbf{V}} \\
&=\frac{\partial E_{t}}{\partial o_{t}}\frac{\partial o_{t}}{\partial
z_{t}}\frac{\partial z_{t}}{\partial\mathbf{V}} \\
&=(o_{t} - \hat{o}_{t}) \otimes s_{t}
\end{align*}

\stopcols

where $z_{t} = \mathbf{V}s_{t}$ and $\otimes$ is the outer product. The error at time $t$ depends only on the last line, $\hat{o}_{t}, o_{t}, s_{t}$.

## Calculating Backpropagation: $\mathbf{W}$

Now we calculate the gradient for $\mathbf{W}$, the weights carried between the hidden states:

\begincols

\column{0.5\linewidth}

![](../imgs/19_rnn_unrolled.png){width=80%}

\column{0.5\linewidth}

$$ \frac{\partial E_{t}}{\partial \mathbf{W}} = \frac{\partial
E_{t}}{\partial o_{t}}\frac{\partial o_{t}}{\partial
s_{t}}\frac{\partial s_{t}}{\partial\mathbf{W}} $$

\stopcols

Now, since $s_{t} = \tanh(\mathbf{U}x_{t} + \mathbf{W}s_{t-1})$, and we're taking the derivative with respect to $\mathbf{W}$, we can no longer ignore the fact that $s_{t}$ relies on $s_{t-1}$ which in turn relies on $s_{t-2}$ and so on.

## Calculating Backpropagation Through Time

So in reality, if we apply the chain rule again, we end up summing across all
timepoints up to the current one:

$$ \frac{\partial E_{t}}{\partial\mathbf{W}} = \sum_{k=0}^{t}\frac{\partial
E_{t}}{\partial \hat{y}_{t}}\frac{\partial\hat{y}_{t}}{\partial
s_{t}}\frac{\partial s_{t}}{\partial s_{k}}\frac{\partial
s_{k}}{\partial\mathbf{W}} $$

Since $\mathbf{W}$ is used in every step up until the step we're interested in, we sum up throughout the network.

## Illustration of BPTT

![Backpropagation Through Time](../imgs/19_bptt.png){width=80%}

## Difficulties in Training RNNs

A similar BPTT process is used for calculating $\frac{\partial E_{t}}{\partial\mathbf{U}}$.

As our sequences get longer, we calculate more and more gradients -- equivalent to stacking more layers in the network.

Thus RNNs have difficulty with long sequences, both because of the amount of computation needed as well as another problem which we will discuss next.

These are the same problems as with "very deep" convolutional networks.

# Vanishing Gradient Problem

## Meaning of Deep Layers in RNNs

As deep networks grow, gradients tend to get lost as they are propagated back from the outputs to the inputs.

In CNNs, additional layers correspond to learning "higher-level" features.

In RNNs, more layers correspond to more timepoints, we're talking about learning connections between inputs (word embeddings) at widely varied timepoints (words that are very far apart from each other).

## Example of Word Distance and Meaning

> It was the Dover road that lay, on a Friday night late in November, before the
> first of the persons with whom this history has business.

## Example of Word Distance and Meaning

> It was the Dover road that lay, on a Friday night late in November, before the
> first of the persons with whom this history has business.

This is a line from the beginning of "A Tale of Two Cities"; it's an example of a sentence where word distances have a lot of meaning:

- There is a road (a road in Dover, a town in Kent in England);
- The description includes a time: Friday night in late November;
- There is a person;
- The road is "before" the person (here, "before" means "in front of");
- The person is the "first" involved in the story, implying that there may be others.

## Example of Word Distance and Meaning

> It was the Dover road that lay, on a Friday night late in November, before the
> first of the persons with whom this history has business.

There are 15 words between "road" and "persons", so understanding the relationship between the two (the road is in front of the person) requires at least 15 timepoints (layers, with their chained gradients).

You can imagine that as the timesteps increase, the layer stacks get larger and larger -- eventually larger than ResNet!

## Extending Gradients is Even Worse Than You Think

$$ \frac{\partial E_{t}}{\partial\mathbf{W}} = \sum_{k=0}^{t}\frac{\partial
E_{t}}{\partial o_{t}}\frac{\partial o_{t}}{\partial
s_{t}}\frac{\partial s_{t}}{\partial s_{k}}\frac{\partial
s_{k}}{\partial\mathbf{W}} $$

The $\mathbf{W}$ gradient includes the term $\frac{\partial s_{t}}{\partial s_{k}}$, which itself expands to a chain rule. So for $t=3, k=1$:

$$ \frac{\partial s_{3}}{\partial s_{1}} = \frac{\partial s_{3}}{\partial s_{2}}\frac{\partial s_{2}}{\partial s_{1}}$$

And so on; as $t$ increases, then the distance between $t$ and 1 increases, so we end up with more and more terms in the gradient.

## Jacobian Matrix

Since $\frac{\partial s_{t}}{\partial s_{k}}$ is a partial derivative of a vector function with respect to a vector input, the result is a ***Jacobian matrix*** whose elements are pointwise derivatives. We can rewrite the gradient as:

$$ \frac{\partial E_{t}}{\partial \mathbf{W}} = \sum_{k=0}^{t}\frac{\partial E_{t}}{\partial o_{t}}\frac{\partial o_{t}}{\partial
s_{t}}\left(\prod_{j=k+1}^{t}\frac{\partial s_{j}}{\partial
s_{j-1}}\right)\frac{\partial s_{k}}{\partial \mathbf{W}} $$

The 2-norm of this matrix has an upper bound of 1, since our activation function ($\tanh$) maps all the inputs to between $[-1, 1]$. The derivative, $\frac{\partial \tanh(x)}{\partial x} = 1 - \tanh^{2}(x)$, is bounded above by 1 as well.

## Activation Function and Its Derivative

![Activation ($\tanh$) and Derivative](../imgs/19_transfer.pdf){width=70%}

## Consequences of Gradient Functions

\begincols

\column{0.5\linewidth}

![Activation ($\tanh$) and Derivative](../imgs/19_transfer.pdf){width=100%}

\column{0.5\linewidth}

Recall that the purpose of the activation function is twofold:

1. Model nonlinear interactions between inputs, and
2. "Squash" the inputs $x$ into a specified range, typically $[0, 1]$ or $[-1, 1]$.

However, the ***derivative*** of $\tanh$ approaches 0 as the inputs become larger. This means that as you perform BPTT, your gradient calculations include multiplication by very small numbers.

\stopcols

## Consequences of Vanishing Gradients

If the gradients go to zero, that means that ***you aren't altering your weights based on errors generated at large timesteps***.

> At the library, patrons are able to select and take home a wide variety of
> [BLANK].

In training, the predicted word [BLANK] is compared with the actual word of the training sentence; so if the actual sentence ended with "books", but the classifier returns "puppies", that would be an error. We would then start calculating our gradient.

***This is an error is due to the word "library", which is 12 words away!*** If the gradients vanish before they get to that point in the BPTT, then we won't use the information at the beginning of the sentence to adjust to the prediction at the end.

## Vanishing vs. Exploding Gradients

There is a similar problem of ***exploding gradients***, but:

1. Exploding gradients will crash your program, so it's obvious; and
2. You can fix exploding gradients by clipping them if they get too large.

Conversely, vanishing gradients won't crash your computer, they will just cause your RNN to train very poorly. So you may not notice that that's the problem unless you look for it.

## Solutions to Vanishing Gradients

Luckily, the solutions to this problem are pretty simple:

1. Initialize $\mathbf{W}$ carefully;
2. Regularize the results of gradient calculations to prevent vanishing;
3. Don't use an activation function with this problem (the RELU unit we discussed earlier has derivatives of just 0 or 1);
4. Use a non-vanilla implementation of RNNs that don't suffer from this issue.

Solutions 3 and 4 are the most popular; in fact, RNNs are rarely used in vanilla form nowadays because of their limited sequence capacity.

# Example Results of RNNs

## Stealing Others' Examples

The following examples are from Andrej Karpathy's website, where he implements a ***character-level RNN***: instead of predicting words, he's predicting the next letter or symbol in a sequence.

In each case, the training set is just a text file, and the classifier is given a "starter word" to initialize. Then it predicts character-by-character what should come next.

Also, he's using a ***long-short term memory (LSTM)*** network, which we'll cover next lecture.

## English Text

The first sample is a set of essays on startups and business. This is a 1MB text file, which contains about 1 million characters (a fairly small sample size for this kind of work).

Here's a sample produced by the RNN:

```
The surprised in investors weren’t going to raise money. I’m not the company
with the time there are all interesting quickly, don’t have to get off the
same programmers. There’s a super-angel round fundraising, why do you can do.
If you have a different physical investment are become in people who reduced
in a startup with the way to argument the acquirer could see them just that
you’re also the founders will part of users’ affords that and an alternation
to the idea. [2] Don’t work at first member to see the way kids will seem in
advance of a bad successful startup. And if you have to act the big company
too.
```

## Shakespeare's Plays {.fragile}

Next is a set of Shakespeare's plays, which are a 4.4MB text file.

```
PANDARUS:
Alas, I think he shall be come approached and the day
When little srain would be attain'd into being never fed,
And who is but a chain and subjects of his death,
I should not sleep

Second Senator:
They are away this miseries, produced upon my soul,
Breaking and strongly should be buried, when I perish
The earth and thoughts of many states.

DUKE VINCENTIO:
Well, your wit is in the care of side and that.
```

## Wikipedia Text

The RNN can also produce markup. Here it's trained on Wikipedia markup text.

```
Naturalism and decision for the majority of Arab countries' capitalide was grounded
by the Irish language by [[John Clair]], [[An Imperial Japanese Revolt]], associated
with Guangzham's sovereignty. His generals were the powerful ruler of the Portugal
in the [[Protestant Immineners]], which could be said to be directly in Cantonese
Communication, which followed a ceremony and set inspired prison, training. 
Many governments recognize the military housing of the
[[Civil Liberalization and Infantry Resolution 265 National Party in Hungary]],
that is sympathetic to be to the [[Punjab Resolution]]
(PJS)[http://www.humah.yahoo.com/guardian.
cfm/7754800786d17551963s89.htm Official economics Adjoint for the Nazism, Montgomery
was swear to advance to the resources for those Socialism's rule,
was starting to signing a major tripad of aid exile.]]
```

## From Text to Code: LaTeX

Let's get more complex and generate a mix of code and text, like a \LaTeX document.

![Fake Math](../imgs/19_fake_math_01.pdf){width=50%}

## From Text to Code: LaTeX

One of these is from the RNN, and the other is from a previous lecture that I wrote.

```
\section{Recap Last Lecture}\label{recap-last-lecture}
\begin{frame}{Recap: Rationale for Deep Networks}
In theory, neural networks can replicate any function (decision
surface), no matter how complicated. "In theory".
In reality, this would require an unreasonable number of:
\begin{itemize}
\tightlist
\item \textbf{\emph{Input nodes}}, to describe increasingly large data
  types;
\item \textbf{\emph{Input samples}}, to accurately describe a complex,
  varied class space;
\item \textbf{\emph{Hidden nodes}}, to capture the nonlinear relationship
  between the inputs and desired outputs;
\item \textbf{\emph{Hours}}, to fully train a network on all this data;
\item \textbf{\emph{Dollars}}, to buy the hardware required for such a
  massive undertaking.
\end{itemize}
```

## From Text to Code: LaTeX

One of these is from the RNN, and the other is from a previous lecture that I wrote.

```
\begin{proof}
We may assume that $\mathcal{I}$ is an abelian sheaf on $\mathcal{C}$.
\item Given a morphism $\Delta : \mathcal{F} \to \mathcal{I}$
is an injective and let $\mathfrak q$ be an abelian sheaf on $X$.
Let $\mathcal{F}$ be a fibered complex. Let $\mathcal{F}$ be a category.
\begin{enumerate}
\item \hyperref[setain-construction-phantom]{Lemma}
\label{lemma-characterize-quasi-finite}
Let $\mathcal{F}$ be an abelian quasi-coherent sheaf on $\mathcal{C}$.
Let $\mathcal{F}$ be a coherent $\mathcal{O}_X$-module. Then
$\mathcal{F}$ is an abelian catenary over $\mathcal{C}$.
\item The following are equivalent
\begin{enumerate}
\item $\mathcal{F}$ is an $\mathcal{O}_X$-module.
\end{lemma}
```

## The Ultimate Test: Source Code

By training on the Linux source code, the RNN can almost write a program.

```
/*
 * Increment the size file of the new incorrect UI_FILTER group information
 * of the size generatively.
 */
static int indicate_policy(void)
{
  int error;
  if (fd == MARN_EPT) {
    /*
     * The kernel blank will coeld it to userspace.
     */
    if (ss->segment < mem_total)
      unblock_graph_and_set_blocked();
    else
      ret = 1;
    goto bail;
  }
```

# Parting Words

## More with Recurrent Nets

Next lecture we will go over extensions to RNNs, including ***Long-Short Term Memory (LSTM)*** and ***Gated Recurrent Unit (GRU)*** networks -- these are designed to explicitly account for vanishing gradients, allowing them to train much longer sequences.

Just like ResNet has kind of taken over as the "default" CNN, LSTM networks have done the same for RNNs in recent papers.
