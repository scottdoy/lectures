---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade
slide-level: 2

title: LSTM NEURAL NETWORKS
subtitle: Machine Learning for Biomedical Data
author: Scott Doyle
contact: scottdoy@buffalo.edu
date: 2020-04-16
---

# 
## Recap

## Recurrent Neural Networks

Just as CNNs were created specifically to deal with image data, RNNs are
specifically focused on **sequence data** -- that is, data that can unrolled
in time.

<p class="fragment">
You can visualize this as a single unit (neuron) that repeats itself and learns
to predict instances based on what came before it.
</p>

## Recap: RNN Units

![Single RNN Unit](img/rnn_rolled.png){width=15%}

## Recap: RNN "Unrolled"

![Unrolled RNN](img/rnn_unrolled.png){width=70%}

## Recap: Weights

As with all NNs, we have a set of weights we need to learn. For RNNs, there are
three sets of weights:

<ul>
<li class="fragment">\$\\mathbf{U}\$, the set of weights from the **inputs** to the **hidden layer** (the "memory state")</li>
<li class="fragment">\$\\mathbf{W}\$, the weights between **hidden layers** across timepoints</li>
<li class="fragment">\$\\mathbf{V}\$, the weights between the **hidden layers**
and the **output layer**</li>
</ul>

<p class="fragment">
And as with traditional NNs, we learn these weights through
**backpropagation**.
</p>

## Recap: Defining Terms

Our hidden state (memory) and the output of a single unit are, respectively:

<div class="l-double">
<div>
![Unrolled Recurrent Network](img/rnn_unrolled.png){width=80%}
</div>
<div>
\\begin{align}
s\_{t} \&= \\tanh(\\mathbf{U}x\_{t} + \\mathbf{W}s\_{t-1}) \\\\
\\hat{y}\_{t} \&= \\textrm{softmax}(\\mathbf{V}s\_{t})
\\end{align}
</div>
</div>

## Recap: Defining Error

Gradient descent and backpropagation require an error function to differentiate
against, so let's define that:

\\begin{align}
E\_{t}(y\_{t},\\hat{y}\_{t}) \&= -y\_{t}\\log\\hat{y}\_{t} \\\\
E(y,\\hat{y}) \&= \\sum\_{t} E\_{t}(y\_{t}, \\hat{y}\_{t}) \\\\
\&= -\\sum\_{t} y\_{t}\\log\\hat{y}\_{t}
\\end{align}

<p class="fragment">
\$y\_{t}\$ is the correct word at time step \$t\$, and \$\\hat{y}\_{t}\$ is our
prediction. This is the **cross-entropy loss function**, and is calculated
over all timesteps (since we typically treat one full sentence as a training
exmaple.
</p>

## Recap: Calculating \$\\mathbf{V}\$

Let's start with \$\\mathbf{V}\$, the weights between the hidden and output layers:

<div class="l-double">
<div>
![Unrolled Recurrent Network](img/rnn_unrolled.png){width=80%}
</div>
<div>
\\begin{align}
\\frac{\\partial E\_{t}}{\\partial \\mathbf{V}} \&= \\frac{\\partial
E\_{t}}{\\partial\\hat{y}\_{t}}\\frac{\\partial\\hat{y}\_{t}}{\\partial\\mathbf{V}} \\\\
\&=\\frac{\\partial E\_{t}}{\\partial \\hat{y}\_{t}}\\frac{\\partial\\hat{y}\_{t}}{\\partial
z\_{t}}\\frac{\\partial z\_{t}}{\\partial\\mathbf{V}} \\\\
\&=(\\hat{y}\_{t} - y\_{t}) \\otimes s\_{t}
\\end{align}

</div>
</div>

where \$z\_{t} = \\mathbf{V}s\_{t}\$ and \$\\otimes\$ is the outer product.

Thus, the error at time \$t\$ depends only on the last line, \$y\_{t}, \\hat{y}\_{t},
s\_{t}\$.

## Recap: Calculating \$\\mathbf{W}\$

Now let's try calculating the gradient for \$\\mathbf{W}\$, the weights carried
between the hidden states:

<div class="l-double">
<div>
![](img/rnn_unrolled.png){width=80%}
</div>
<div>
\$ \\frac{\\partial E\_{t}}{\\partial \\mathbf{W}} = \\frac{\\partial
E\_{t}}{\\partial\\hat{y}\_{t}}\\frac{\\partial\\hat{y}\_{t}}{\\partial
s\_{t}}\\frac{\\partial s\_{t}}{\\partial\\mathbf{W}} \$

</div>
</div>

<p class="fragment">
Now, since \$s\_{t} = \\tanh(\\mathbf{U}x\_{t} + \\mathbf{W}s\_{t-1})\$, and we're
taking the derivative with respect to \$\\mathbf{W}\$, we can no longer ignore the
fact that \$s\_{t}\$ relies on \$s\_{t-1}\$ which in turn relies on \$s\_{t-2}\$ and so
on.
</p>

## Recap: Calculating BPTT

So in reality, if we apply the chain rule again, we end up summing across all
timepoints up to the current one:

\$ \\frac{\\partial E\_{t}}{\\partial\\mathbf{W}} = \\sum\_{k=0}\^{t}\\frac{\\partial
E\_{t}}{\\partial \\hat{y}\_{t}}\\frac{\\partial\\hat{y}\_{t}}{\\partial
s\_{t}}\\frac{\\partial s\_{t}}{\\partial s\_{k}}\\frac{\\partial
s\_{k}}{\\partial\\mathbf{W}} \$

<p class="fragment">
Since \$\\mathbf{W}\$ is used in every step up until the step we're interested in,
we sum up throughout the network. (This is also true for \$\\frac{\\partial
E\_{t}}{\\partial\\mathbf{U}}\$.)
</p>

## Recap: Vanishing Gradients

Since we're calculating throughout time, as our sequences get longer, we have to
calculate more and more gradients -- this is equivalent to stacking more layers
in our network!

<p class="fragment">
With more layers / timepoints, calculating the gradient becomes difficult due to
the **vanishing gradient** problem, which means that vanilla RNNs have
difficulty with sequences that are very long, equivalent to trying to learn
proper sentence completion that involves many words between meanings.
</p>

## Recap: Source of Vanishing Gradients

<div class="l-double">
<div>
![Activation \$\\tanh\$ and Derivative](img/transfer.svg){width=100%}

</div>
<div>


Recall that the purpose of the activation function is twofold:

1. Model nonlinear interactions between inputs, and
2. "Squash" the inputs \$x\$ into a specified range, typically \$[0, 1]\$ or $[-1,
   1]$.

However, the derivative of \$\\tanh\$ approaches 0 as the inputs become larger.
This means that as you perform BPTT, your gradient calculations include
multiplication by very small numbers.

</div>
</div>

## Recap: Long Sequences

If the gradients go to zero, that means that **you aren't altering your weights
based on errors generated at large timesteps**.

> At the library, patrons are able to select and take home a wide variety of
> [BLANK].

There are 12 words between "library" and [BLANK], meaning for us to correctly
learn this relationship, we would need at least that many timepoints.

If the gradients vanish before then, we can't use the information at the
beginning of the sentence to adjust to the prediction at the end.

## Recap: Solutions to Vanishing Gradients

Luckily, the solutions to this problem are pretty simple:

1. Initialize \$\\mathbf{W}\$ carefully;
2. Regularize the results of gradient calculations to prevent vanishing;
3. Don't use an activation function with this problem (the RELU unit we
   discussed earlier has derivatives of just 0 or 1);
4. Use a non-vanilla implementation of RNNs that don't suffer from this issue.

Solutions 3 and 4 are the most popular; in fact, RNNs are rarely used in vanilla
form nowadays because of their limited sequence capacity.

# 
## More RNN Examples

## RoboProf v0.1

Of course, the first thing AI researchers try to do is create their own robotic
replacement. So I trained an RNN on a text file of all of my lectures.

<p class="fragment">
I write my lectures in Markdown, which is a text file format that looks like
this...
</p>

## Markdown Example

```
## Recap: Source of Vanishing Gradients

<div class="l-double">

<div>


![Activation (\$\\tanh\$) and Derivative](img/transfer.pdf){width=100%}

</div>
<div>


Recall that the purpose of the activation function is twofold:

1. Model nonlinear interactions between inputs, and
2. "Squash" the inputs \$x\$ into a specified range, typically \$[0, 1]\$ or $[-1,
1]$.

</div>
</div>
```

This gets converted to \\LaTeX, which in turn is used to create the PDF slides
you see.

## Training Process

So I took all my markdown files for the lectures so far, concatenated them, and
ran an RNN on them.

<p class="fragment">
I selected chunks of the output that delineated individual slides, and straight
cut-and-pasted them into this document. The following 3 slides haven't been
modified by me at all, and were compiled along with the rest of this document.
</p>

<p class="fragment">How'd it do?</p>

## Recap: Quariate Renition of Training Estimation

The will be a low of the samples of the data to do the component to \$\\mathbf{x}\$
samples are can be from the simplify the size signal of the training independent
neured and the samples are the split to the to the output in the samples.

## Recap: Recap: Example of Classification

We we can training the samples are classifier to the error of the same in
\$\\mathbf{a}\$, and \$\\mathbf{a}\$ and \$\\mathbf{a}\$ is a set of the reduces of the
training classifiers.

## Recap: Parameter Errors \$\\mathbf{a}\$ is so the class convergence all
\$\\mathbf{x}\$ is a probability of the probability of the probability of the
deristical training between the component in the component classifiers that
\$\\mathbf{x}\$ and \$\\mathbf{a}\$ and \$\\mathbf{y}\$ is the matrix to \$\\mathbf{y}\$ to
the weights with \$\\mathbf{a}\$ is the network \$\\mathbf{y}\$ subcestative estimate
the classifier and the look to a porterior sumptors to the component in the
component to the mean the layers are a sample of \$\\mathbf{x}\$ and \$\\mathbf{y}\$
and \$\\mathbf{x}\$ is the feature values converges the samples the samples of
\$\\mathbf{x}\$.

## More Examples (This Is Really Me Now)

Sometimes, the system would try to create "image slides" which won't compile
because the images don't exist.

<p class="fragment">
But the markdown is formatted properly, it recognizes my image directory, it
even identified the naming scheme I use for my images (two digits represent the
lecture number, then a bunch of underlined phrases for the name), and the fact
that I have to include the width parameter on the image:
</p>

```
## Classification (Samples

![Lect To Decision Functions](img/simple\_signal\_signal\_00.pdf){width=100%}
```

## More Examples (No Really, It's Me!)

It would also recognize the overall structure of "column slides", but would
usually forget to close them:

```
## Learning Discriminant Function Space

<div class="l-double">

</div>
<div>


![](img/decision\_threshold\_b.pdf){width=100%}

</div>
<div>


![](img/disc\_func\_noncontint.pdf){width=100%}

```

## More Examples

Amazingly, it sometimes created fairly complicated functions using LaTeX syntax
(and occasionally drops in some bold words for some reason)...

## Comparing Recap: Simpling \$\\boldsymbol{\\Sigma}\_{i}\$

\$ \\mathcal{D}\_{1} =
\\frac{1}{2}\\ln{\\frac{1}{2}\\sum\_{k=1}\^{n}\\hat{\\boldsymbol{\\Sigma}}_{i}\^{2}} \$

The sample of the probability of the with of the network with \$\\mathcal{D}\_{i}\$
is the convergences are mean distribution as a linear probability of the
component function with **distance**.

The here and \$x\$ is the incouting a have a set of the many into a probability of
\$\\mathbf{a}\$ and \$\\mathbf{x}\$, we estimate the space (assume the network we can
be to a network representation of the samples is the same that a set of a points
in the samples of the component is \$\\mathbf{a}\$ is the distance random that is
\$\\mathbf{y}\$.

In the sample of the samples rearning the training and the evaluate the
classifier samples.

# 
## Learning to Speak

## Online Implementations

There are lots of online implementations of various RNN flavors, including
LSTMs. 

One of them, from [this Github
site](https://github.com/yxtay/char-rnn-text-generation), actually provides
periodic updates on the generated text as it goes through the training process.

The training text was Shakespeare, again. 

[https://github.com/yxtay/char-rnn-text-generation](https://github.com/yxtay/char-rnn-text-generation)

## Shakespeare, Epoch 1, 92s, loss: 3.138

Generating with seed: "re":

```
reoltaoud alet miaele iesl saur thee withe
 whe an hass wo hocse,
sor nant mote iouthe bor iod
iwd al it alirsd morhy whr ortee, ane sit snasd mere helart an sn hire mise wo wor biu hout ne bor iitthe haus,

eous isren ans biul wan hos syenlisd.

AROTES:
IENR:ONOO she me ion bod,
tantsininl,
teasenl wn sasite het ais meerded tott tradd shhy mhats teiss.

I ward
tiir
stinr ton oraln
tes, I toun boset, aaus bul muss orelhe ne m int,
Ahes ae sareneulot il bo anns, tis bateito he toenns iurd wimis hine taue ie to
```

## Shakespeare: Epoch 15, 93s, loss: 1.6338

Generating with seed: "hen the kite bui":

```
hen the kite buith:
Whele with our him me strose sume to tay
And by welt of her so hen his lied thick?

FRLARD:
A hear that buch their hope as you he dences,
To this would at is beaty, the harbing;
My hee heary, whine one told with hath:' by love see thrould
Who care wither is no bedientiar.

LUSEN:
What we had al be my sence
His better shour me striend your sousing
To thou sill of the suckie abeid?

Secord:
That henchen you seve sent of the weeth
The duke to some his curn before hath with.
```

## Shakespeare, Epoch 33, 3003s

Generating with seed: "ivines":

```
ivines.

HASTINGS:
Go, fellow, give me that with thee the was shall
With men, the sease to see as the stand them, and the cannot,
Who in thine.

PRINCO:
I was and the will be the words the chance
The cause and them women to best blood to
chall and heard too lives to strong of too man,
And so welcome true this with myself was her
And so, and well this wifting stranger, thou art
Than his storm'd, tell the word of this breath,
As I have too much and thine one and head
I words a triend and them would his part,
Whet are he would him the will be a man
The stranged and to be and their house.

```

## ScottBot: Epoch 1

Seed: once I make

```
l o ong tor weet iceretonl tang ot mume tee an thild wo bering. me coung ot ind I Ie wat it she be the is hacer.. the ani tuns hen o ank itid thes bee tols Iille bhititinss hanes at inl inss
thent ankthalt, ors and ose som ar and art sonetes hitond I and m bere sirghangithas at arteren Iill Iey I'd ho wheer heelims I tur sous it selto sans at ther he beeg tinser one ciro ta seat alg tharista tuntenas af cars ceamasgrene wor busnind boudel it wtol att wol ant wand mannin tin int an the ingim te wend whe slis
```

## ScottBot: Epoch 50

Seed: h was the orig

```
inal flation of a book constricting
""As one of achice... on the comestey to get that ask of a boos
You get the considenelity." My going? The messem by it in a marter that a spart point brain
He was status oft that.
I mean. I are anyone are sittg about the spyen to all that there's a leal would the targeth party of slots, but you're point
Start of the ficture: a tould be option has good for a mudy and the there is that a fixse for them weekend
Assome to suggestion as once if he's been with packed at the shut 
```

## ScottBot: Epoch 100

Seed: Okay, so
```
 I'll be a bit to be a lot of the could have been a lot of more too and started a band of the stand is this point, but I do think it's a store to be start and the same one of more to to that a statistic or so started a beer to the candy in the politics and and a beer are a little beach, but if you don't know what I was the conferencing to the story about it and the cares of the posters. 
 And I was a sending the same the choice of the same to be the status thing to that. 
 It's no bean and a sense and too stands to say, and if you wouldn't be trying 
```

# 
## Building on RNNs

## Additional Information

The diagrams from this lecture come from Chris Olah, an engineer at Google,
who runs a blog at [colah.github.io](colah.github.io). He provides a very clear
and straightforward explanation of LSTMs (among other things -- check his site
out!).

## Short-term Dependencies

![Short-term Dependencies are Easy to Learn](img/rnn_shorttermdependencies.png){width=100%}

## Long-term Dependencies

![Long-term Dependencies are Difficult to Learn](img/rnn_longtermdependencies.png){width=100%}

## Overcoming Limitations of RNNs

As we said, the major drawback to vanilla RNNs is the lack of good support for
long term sequences. We made some suggestions on how to solve this issue, but
the most common one is to replace the individual hidden state neurons with a
more complicated unit.

<p class="fragment">First, let's re-examine the RNN hidden unit.</p>

## Simple RNN: \$\\tanh\$ Layer

![Simple \$\\tanh\$ layer](img/lstm_simplernn.png){width=100%}

## LSTM: Repeating Module Structure

In "Long Short Term Memory" networks, the main hidden layer is replaced with a
more complex structure.

<p class="fragment">
The way to think about this is instead of having a single set of neurons in the
hidden layer, there are several intermediate hidden units that operate **at one
timepoint**, so the center layer ends up with a lot of sub-units that operate
at \$t\$ until they produce an output.
</p>

## LSTM: Repeating Module Structure

![Complex LSTM Layer](img/lstm_chain.png){width=100%}

## LSTM Graphical Notation

<div class="l-double">
<div>
![LSTM Structure](img/lstm_chain.png){width=100%}
</div>
<div>
We'll go over each component in the structure in time, but it's important to
understand the basic notation:

<ul>
<li class="fragment">Yellow boxes are **learned layers**.</li>
<li class="fragment">Pink circles are **pointwise operations**.</li>
<li class="fragment">Vector lines are **transfer lines**.</li>
<li class="fragment">Merging vector lines are **concatenation operations**.</li>
<li class="fragment">Splitting vector lines are **copy operations**.</li>
</ul>

</div>
</div>

## Core Innovation of LSTMs

<div class="l-double">
<div>
![\$C\_{t}\$ Line](img/lstm_cline.svg){width=100%}

</div>
<div>
The top line is equivalent to our "hidden memory state" that gets passed from
timestep to timestep.

<p class="fragment">
The data flowing through this connection is modified at two points; the amount
that this data is modified is controlled by "gates", which control how (and how
much) the data from previous timesteps should be used.
</p>

</div>
</div>

## Gated Information Modifications

<div class="l-double">
<div>
![LSTM Gates](img/lstm_gate.png){width=60%}

</div>
<div>
There are three gates in the system; each is a sigmoid layer followed by a
pointwise multiplication.

<p class="fragment">
The output of the sigmoid layer is \$[0, 1]\$, so these gates learn **how much
information to add to the hidden state** at each timepoint.
</p>

<p class="fragment">
The action of these gates determines what happens to the cell state, so they can
intentionally keep the gradient alive (or squash it if it's getting too large).
</p>

</div>
</div>

# 
## Step-by-Step Walkthrough

## Forget Gate Layer

<div class="l-double">
<div>
![Forget Gate](img/lstm_focusf.svg){width=100%}
</div>
<div>
The first thing that happens in the cell is that the cell state is modified by a
**forget gate**, which simply multiplies the cell state by a fraction.

<p class="fragment">
This path takes \$h\_{t-1}\$ (the **output** from the previous timepoint) and
\$x\_{t}\$ (the **input** at the current timepoint), and the
\$\\boldsymbol{\\sigma}\$ gate spits out \$f\_{t}\\in[0,1]\$ which is multiplied by the
cell state from the previous step \$C\_{t-1}\$.
</p>

<p class="fragment">\$ f\_{t} = \\sigma(W\_{f}\\cdot [h\_{t-1}, x\_{t}] - b\_{f}) \$</p>

</div>
</div>

## Input Gate Layer

<div class="l-double">
<div>
![Input Gate](img/lstm_focusi.svg){width=100%}

</div>
<div>
Next we need to add information to the cell state with an **input gate**. This
takes place in two steps:

<ol>
<li class="fragment">Decide **which** values of \$x\_{t}\$ and \$h\_{t}\$ will
be updated by an input gate \$\\sigma\$.</li>
<li class="fragment">Decide **what** the values are going to be by applying a \$\\tanh\$ layer to \$x\_{t}\$ and \$h\_{t-1}\$.</li>
</ol>

<p class="fragment">
\\begin{align}
i\_{t} \&= \\sigma(W\_{i}\\cdot [h\_{t-1}, x\_{t}] + b\_{i}) \\\\
\\tilde{C}\_{t} \&= \\tanh(W\_{C}\\cdot [h\_{t-1}, x\_{t}] + b\_{C}) \\\\
\\end{align}
</p>

</div>
</div>

## Modify Cell State

<div class="l-double">
<div>
![Modify \$C\_{t-1}\$](img/lstm_focusc.svg){width=100%}
</div>
<div>
Now that we know **how** to modify \$C\_{t-1}\$, we go ahead and do it:

\$ C\_{t} = f\_{t} \\ast C\_{t-1} + i\_{t} \\ast \\tilde{C}\_{t} \$

<p class="fragment">
This gives us a new cell state for the current timepoint, \$C\_{t}\$, which then
exits the cell to become the \$C\_{t-1}\$ of the next state.
</p>

<p class="fragment">
But we aren't done! What about calculating the **output** of this cell,
\$h\_{t}\$?
</p>

</div>
</div>

## Calculate Output

<div class="l-double">
<div>
![Calculate \$h\_{t}\$](img/lstm_focuso.svg){width=100%}

</div>
<div>


\$h\_{t}\$ is a **filtered version of the new cell state**. The process is the same as
\$C\_{t-1}\$: Figure out **which** values to modify by calculating a sigmoid
operating on \$h\_{t-1}\$ and \$x\_{t}\$, and then calculate **what** values to use
by passing the cell state through a \$\\tanh\$ op:

<p class="fragment">
\\begin{align}
o\_{t} \&= \\sigma(W\_{o} \\cdot [h\_{t-1}, x\_{t}] + b\_{o})\\\\
h\_{t} \&= o\_{t} \\ast \\tanh(C\_{t}) \\\\
\\end{align}
</p>

</div>
</div>

## Calculate Output

<div class="l-double">
<div>
![Calculate \$h\_{t}\$](img/lstm_focuso.svg){width=100%}

</div>
<div>


\\begin{align}
o\_{t} \&= \\sigma(W\_{o} \\cdot [h\_{t-1}, x\_{t}] + b\_{o})\\\\
h\_{t} \&= o\_{t} \\ast \\tanh(C\_{t}) \\\\
\\end{align}

<p class="fragment">
Note that in this case, the \$\\tanh\$ layer is **not** a trained layer, but
simply the \$\\tanh\$ function!
</p>

<p class="fragment">
In other words, we don't calculate on \$C\_{t}\$ with a set of weights and biases
before passing it through the \$\\tanh\$ function, and we don't have to learn
another set of gradients here.
</p>

</div>
</div>

## And On and On and On...

![LSTM Chain](img/lstm_chain.png){width=50%}

We have more weights to learn, and so this system is more computationally
expensive, but it nicely allows us to sidestep the issues with long-term
information connections -- by controlling how the gradient is modified at each
step, we can prevent the gradient from shrinking to 0 or exploding to \$\\infty\$.

# 
## Variants on LSTMs

## Rolling Your Own LSTM

As with CNNs, there are variants on this setup. The main differences have to do
with how the connections are "wired up" inside the cell.

<p class="fragment">
We'll go through the ones that Olah mentions in his blog post, although (as
pointed out) there are a LOT of these.
</p>

<p class="fragment">
Remember the golden rule: **Keep It Simple**! Use the simplest algorithm that
works on your problem. If it doesn't work, figure out why; if you have enough
data, and you train for long enough, only then should you consider using a more
complex architecture.
</p>

## Peephole Connections

<div class="l-double">
<div>
![Peephole Connections](img/lstm_var_peepholes.svg){width=100%}
</div>
<div>
This variant of LSTMs allows the cell state to influence the operation of the
gates by providing connections between \$C\_{t-1}\$ and each of the \$\\sigma\$
layers:

\\begin{align}
f\_{t} \&= \\sigma(W\_{f}\\cdot [C\_{t-1}, h\_{t-1}, x\_{t}] + b\_{f}) \\\\
i\_{t} \&= \\sigma(W\_{i}\\cdot [C\_{t-1}, h\_{t-1}, x\_{t}] + b\_{i}) \\\\
o\_{t} \&= \\sigma(W\_{o}\\cdot [C\_{t}, h\_{t-1}, x\_{t}] + b\_{o}) \\\\
\\end{align}

</div>
</div>

## Coupled Forget and Input Gates

<div class="l-double">
<div>
![Tied Connections](img/lstm_var_tied.svg){width=100%}
</div>
<div>
Another variation is to couple the forget and input gates, so these are learned
together. This way we can forget only when there's something to take its place,
and we can input something when we forget something else (so we're
**updating** explicitly).

\\begin{align}
C\_{t} \&= f\_{t} \\ast C\_{t-1} + (1 - f\_{t}) \\ast \\tilde{C}\_{t} \\\\
\\end{align}

</div>
</div>

## Gated Recurrent Units (GRUs)

<div class="l-double">
<div>
![Gated Recurrent Units](img/lstm_var_gru.svg){width=100%}

</div>
<div>


The Gated Recurrent Unit (GRU) is often thought of as a natural successor to
LSTM. This computes an "update" gate instead of forget and input gates
(as with the Coupled variant), but also merges the cell and hidden states, as
well as other changes. It ends up with fewer calculations than LSTM:

\\begin{align}
z\_{t} \&= \\sigma(W\_{z} \\cdot [h\_{t-1}, x\_{t}]) \\\\
r\_{t} \&= \\sigma(W\_{r} \\cdot [h\_{t-1}, x\_{t}]) \\\\
\\tilde{h}\_{t} \&= \\tanh(W \\cdot [r\_{t} \\ast h\_{t-1}, x\_{t}]) \\\\
h\_{t} \&= (1 - z\_{t}) \\ast h\_{t-1} + z\_{t} \\ast \\tilde{h}\_{t} \\\\
\\end{align}

</div>
</div>

## Comparison of the Variants

There are a few papers that look into the question of which variant of RNNs
works "best", and of course, the answer is: "Depends."

<p class="fragment">
Some applications work better with different architectures, so try different
architectures and see what you get.
</p>

<p class="fragment">However, make sure you know what you're doing and why you're doing it!</p>

<ul>
<li class="fragment">How long are your sequences?</li>
<li class="fragment">How big is your "vocabulary"?</li>
<li class="fragment">How much data do you have to train?</li>
</ul>

<p class="fragment">**Start simple, move to complex.**</p>

# 
## Parting Words

## Keeping Up With The Times

The key to staying on top of the field is to read, read, read... and experiment!

<p class="fragment">
Play with these algorithms if they are implemented in publicly available
codebases, and try implementing some of them on your own.
</p>

<p class="fragment">
It isn't terribly hard to wire up new sets of connections and see how the output
looks, and you can't really "break" anything, so go nuts (and publish what you
find)!

</p>
