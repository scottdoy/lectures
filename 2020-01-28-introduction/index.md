---
theme: scottdoy
highlight-style: atelier-dune-light
transition: fade
slide-level: 2

author: Scott Doyle
contact: scottdoy@buffalo.edu
title: Introduction
subtitle: Machine Learning for Biomedical Data
date: 2020-01-28
---

# 

## Welcome! 

## Instructor Information

<div class="txt-left">
Scott Doyle

PAS, BME, BMI

Downtown: 4205 JSMBS (M, W, F)<br/>
North: 215-J Bonner Hall (Tu, Th)

<i class="fa fa-envelope"></i> scottdoy@buffalo.edu<br/>
<i class="fa fa-phone"></i> 716.829.2005<br/>
<i class="fa fa-globe"></i> [***www.scottdoy.com***](http://www.scottdoy.com)
</div>

## Machine Learning

A Brief Introduction

## Machine Learning Definitions

**Machine Learning** (ML) uses **collected data** to do something useful.

<div class="txt-left">
<ul>
<li class="fragment">Find underlying patterns (**knowledge discovery**)</li>
<li class="fragment">Simplify a complex phenomenon (**model building**)</li>
<li class="fragment">Place data into categories (**classification**)</li>
<li class="fragment">Predict future data (**regression**)</li>
</ul>
</div>

## Machine Learning Definitions

The job of the ML expert is to:

<div class="txt-left">
<ul>
<li class="fragment">Understand and identify the **goal**</li>
<li class="fragment">Collect **data**</li>
<li class="fragment">Select an appropriate **model** or **algorithm**</li>
<li class="fragment">Evaluate the system in terms of **costs**</li>
</ul>
</div>

## Types of Machine Learning

<div class="l-double">
<div>
**Supervised Learning**

<p class="fragment">Use **labeled datasets** to classify new, unseen data</p>
</div>
<div> 
**Unsupervised Learning**

<p class="fragment">Use **unlabeled data** to identify natural groups</p>
</div>
</div>

<div class="l-double">
<div>
**Semi-Supervised Learning**

<p class="fragment">Use **partially labeled** data 
to handle the process</p>
</div>
<div> 
**Reinforcement Learning**

<p class="fragment">An **agent** learns to complete a task **policy** of rewards</p>
</div> 
</div>

# 

## Data Definitions

The starting point for all ML algorithms is **data**.

<p class="fragment">So... what do we mean by "data"?</p>

## Data Comes in Many Forms

![Complex, Multi-Modal Data](img/data_formats.png){ width=70% }

## Data Comes from Many Places

|              |            |
|:-------------|-----------:|
| Symptoms     | Lab Tests  |
| Demographics | Imaging    |
| Cultures     | Sequencing |

<p class="fragment">Data quantifies a specific **subject** (e.g. patient) through a set of
**measurements**.</p>

## Computational Pathology:<br/> Expression of Disease State

<p class="fragment">Biological structure is **primary data**. </p>
<p class="fragment">We can quantify **biological structure**.</p>
<p class="fragment">We can **model** relationships between **structure and disease**.</p>

## Fundamental Hypothesis

Changes in genomic expression manifest as physical changes in tumor morphology

<div class="fragment l-double">
<div>
![ ](img/badve2008_fig4b1.svg){ width=80% }
</div>
<div> 
![ ](img/badve2008_fig4b2.svg){ width=80% }
</div>
</div>

<p class="fragment" style="text-align: left;"><small>
S. S. Badve et al., JCO (2008),
Paik et al., N Engl J Med (2004)
</small></p>

## Fundamental Hypothesis 

Changes in genomic expression manifest as physical changes in tumor morphology

<div>
![](img/paik2004_fig2.svg){ width=80% }
</div>

<p style="text-align: left;"><small>
S. S. Badve et al., JCO (2008),
Paik et al., N Engl J Med (2004)
</small></p>

## Data Fusion Improves Predictions

<div class="l-multiple" style="grid-template-columns: auto auto auto;">
<div style="grid-row: 1;">
![Quantitative Histology](img/lee2015_quanthisto.png){ height=30% }
</div>
<div style="grid-row: 1;">
![&nbsp;](img/lee2015_lowdim1.png){ height=30% }
</div>
<div style="grid-row: 1 / span 2;vertical-align: middle;">
![Combined Embeddings](img/lee2015_combined.png){ height=30% }
</div>
<div style="grid-row: 2;">
![Mass Spectrometry](img/lee2015_massspect.png){ height=30% }
</div>
<div style="grid-row: 2;">
![Low-Dimensional Embeddings](img/lee2015_lowdim2.png){ height=30% }
</div>
</div>

## Atoms to Anatomy Paradigm

<div class="l-multiple" style="grid-template-columns: 1.5fr 1fr 1fr 1fr; row-gap:0;">
<div style="grid-row: 1 / span 2;">
![](img/ata01.png){ width=100% }
</div>
<div style="grid-row: 1;">
![](img/ata02.png){ height=356 width=456 }
</div>
<div style="grid-row: 1;">
![](img/ata03.png){ height=356 width=456 }
</div>
<div style="grid-row: 1;">
![](img/ata04.png){ height=356 width=456 }
</div>
<div style="grid-row: 2;">
![](img/ata05.png){ height=356 width=456 }
</div>
<div style="grid-row: 2;">
![](img/ata06.png){ height=356 width=456 }
</div>
<div style="grid-row: 2;">
![](img/ata07.png){ height=356 width=456 }
</div>
</div>

#

## Example Problem

Fine Needle Aspirate Classification

## Example: Biomedical Image Analysis

<div class="l-double">
<div>
![](img/fna_92_5311_benign.png){ width=80% }
</div>
<div> 
![](img/fna_91_5691_malignant.png){ width=80% }
</div>
</div>

## Fine Needle Aspirates

<div class="l-double">
<div>
![Benign FNA Image](img/fna_92_5311_benign.png){ width=80% }
</div>
<div> 
![Malignant FNA Image](img/fna_91_5691_malignant.png){ width=80% }
</div>
</div>

<p class="fragment">
**Problem Statement** 
</p>
<p class="fragment">Predict whether a patient's tumor is benign or malignant, given an FNA image</p>

## Building Informative Features

<p class="fragment">**Domain knowledge** identifies useful features.</p>

<p class="fragment">Pathologists already distinguish **benign** from **malignant** tumors.</p>

<p class="fragment">Our job is to convert **qualitative** features to **quantitative** ones.</p>

## Building Informative Features

The pathologist lists **cell nuclei** features of importance:

<div class="l-double">
<div>
1. Radius
2. Texture
3. Perimeter
4. Area
5. Smoothness
</div>
<div>
6. Compactness
7. Concavity
8. Concave Points
9. Symmetry
10. Fractal Dimension
</div>
</div>

<p class="fragment">**Feature extraction** results in 30 feature values per image.</p>

## Selecting Features for the FNA

To begin, we collect **training samples** to build a model.

<div class="txt-left">
<ul>
<li class="fragment">Collect a lot of example images for each class</li>
<li class="fragment">Get our expert to label each image as "Malignant" or "Benign"</li>
<li class="fragment">Measure the features of interest (image analysis or by hand)</li>
<li class="fragment">Build a histogram of the measured feature</li>
</ul>
</div>

## Texture of the Nuclei

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/texture_mean.html"></iframe>

## Selecting New Features

The histograms are similar, with **significant overlap** between classes.

<p class="fragment">
Our pathologist tells us that malignant nuclei are often **larger** than
benign nuclei, so we can build a second histogram for this new feature.
</p>

## Average Radius of the Nuclei

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/radius_mean.html"></iframe>

## Characteristics of Good Features

Better! In general, "good" features for ML are:

<div class="txt-left">
<p class="fragment">**Descriptive:** Similar within a class, and different between classes</p>
<p class="fragment">**Relevant:** Features should make sense</p>
<p class="fragment">**Invariant:** Not dependent on how you measure them</p>
</div>

## Calculating Probabilities from Features

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/pdf_cdf.html"></iframe>

## Combinations of Features

**Combining features** often yields greater class separation.

<p class="fragment">
We can visualize two features as a **scatter plot**.
</p>
<p class="fragment">
Each point is an object,
and each axis is a dimension or a feature.
</p>

## Multivariate Distribution

<iframe frameborder="0" seamless='seamless' scrolling=no src="plots/scatter_histogram_plot.html"></iframe>

## Variance vs. Generalization

<p>Linear boundaries do not model **variance** and miss obvious trends.</p>
<p class="fragment">Complex boundaries fit training perfectly, but do not **generalize**.</p>
<p class="fragment">In general, you want the **simplest** model with the best **performance**.</p>

## Tradeoff: Variance vs. Generalization

<p>Each of these decision boundaries makes errors!</p>
<p class="fragment">There is always a tradeoff; we need to consider the **cost**.</p>
<p class="fragment">Cost is defined by our goals and acceptable performance.</p>

## Costs

It is important to recognize the difference between **cost** and **risk**. 

<div class="txt-left">
<ul>
<li class="fragment">Risk is the likelihood you have made an error.</li>
<li class="fragment">Cost is the penalty associated with making that error.</li>
</ul>
</div>

## Cost Weighting

Should we prioritize some kinds of errors over others?

<div class="fragment txt-box">
Not all mistakes carry the same cost. For example:

- A patient is told they have a tumor when they do not (**false positive**)
- A patient is told they are cancer-free when they are not (**false negative**)
</div>

## In Summary

Machine learning turns raw data into **actionable** insights.

<p class="fragment">
We will cover theory and implementation of popular methods.
</p>

<ul>
<li class="fragment">
How could you apply these methods to your own data?
</li>
<li class="fragment">
How could quantitative techniques help test hypotheses and generate novel experiments?
</li>
</ul>


# 

## Course Assessments

## Course Resources: Textbooks

<div class="l-double">
<div>
*Pattern Classification, 2nd Edition*

- Duda, Hart, and Stork (DHS)
- ISBN 978-0-471-05669-0

<p class="fragment">
NOT required, but HIGHLY recommended if you are interested in the
material.
</p>
<p class="fragment">
Also sold with a companion MATLAB book by Stork and Yom-Tov.
</p>
</div>
<div>
![](img/duda_textbook.png){width=60%}
</div>
</div>

## Course Resources: Textbooks

<div class="l-double">
<div>
*Hands On Machine Learning with Scikit-Learn and Tensorflow*

- Aurélien Géron
- ISBN 978-1491962299

<p class="fragment">Very focused on applications in Python programming.</p> 

<p class="fragment">We will not use Tensorflow but we will make heavy use of Scikit-Learn.</p>
</div>
<div>
![](img/handson_textbook.jpg){width=60%}
</div>
</div>
## Programming Expectations

<p class="fragment">
This is **not** a programming class; it is a class that **uses**
programming.
</p>

<p class="fragment">Spend time designing algorithms, not debugging.</p>

<p class="fragment">
Programming is a tool to get what you want -- if you define what you
want up front, the programming almost doesn't matter. (Almost.)
</p>

## Programming Tools: MATLAB

Most of you probably have experience with ***MATLAB***.

<ul>
<li class="fragment">You should already know it</li>
<li class="fragment">Toolboxes implement most low-level algorithms</li>
<li class="fragment">It's what the BME Department supports</li>
<li class="fragment">If you are new, it's not hard to pick up</li>
</ul>

## Programming Tools: Python

***Python*** is a great alternative.

<ul>
<li class="fragment">It's free! It's cross-platform!</li>
<li class="fragment">Libraries extend the base Python, just like MATLAB Toolboxes</li>
<li class="fragment">"Anaconda" distribution is specifically for data science: [https://www.continuum.io/downloads](https://www.continuum.io/downloads)</li>
<li class="fragment">Can convert between MATLAB to Python relatively easily</li>
<li class="fragment">Most common language for academic / "hobby" ML</li>
</ul>

## Programming Tools: Other

Feel free to use any language you like for your
assignments (R, Lua, C++, Fortran...)

<p class="fragment">I will provide help where I can, but I am most familiar with MATLAB and Python.</p>

## Programming Help

Here's a reasonable workflow when you get an error:

<ol>
<li class="fragment">Write down in English what you're trying to do.</li>
<li class="fragment">Google that sentence with "PYTHON" on the end.</li>
<li class="fragment">If that doesn't work, ask someone!</li>
</ol>

## Programming Help

<div class="txt-left">
*General:*
: Google is your friend!
: StackOverflow: ***[stackoverflow.com](http://www.stackoverflow.com)***
</div>
<div class="txt-left fragment">
*MATLAB:*
: Mathworks Website: ***[mathworks.com](http://www.mathworks.com)***
</div>
<div class="txt-left fragment">
*Python:*
: Beginner's Guide: ***[wiki.python.org/moin/SimplePrograms](http://wiki.python.org/moin/SimplePrograms)***
: Scientific Docs: ***[docs.scipy.org/doc/](http://docs.scipy.org/doc/)***
</div>

## Course Evaluation

You will be graded primarily on four components:

<ul>
<li class="fragment">**Participation**: Show up to class, ask questions, and answer them</li>
<li class="fragment">**Assignments**: 4-5 assignments per semester</li>
<li class="fragment">**Paper Presentation**: 15-minute formal presentation to the class</li>
<li class="fragment">**Final Project**: Due at the end of the semester</li>
</ul>

## Course Evaluation: Assignments

Assignments will be assessed on:

<ul>
<li class="fragment">Whether you did the assignment</li>
<li class="fragment">Writing up what you did, what happened, and your interpretation</li>
<li class="fragment">How well you implemented what you learned in class</li>
</ul>

<p class="fragment">The assignments **build on each other** (so don't delete it when you're done!)</p>

## Course Evaluation: Presentations

Class presentations can be any of the following:

<ul>
<li class="fragment">**Paper Discussion**: Critical summary of a paper.</li>
<li class="fragment">**Assignment / Project Presentation**: Discuss the status of your project.</li>
<li class="fragment">**Research Discussion**: If you doing research outside of
class, tell us about it!</li>
</ul>

## Course Evaluation: Projects

Assignments build pieces of a classification system. 

<p class="fragment">The project just ties it all together with a formal writeup and hand-in.</p>

<ul>
<li class="fragment">If you understand and complete the assignments, the project is a breeze.</li>
<li class="fragment">If you fall behind or don't grasp the assignments, it's less pleasant.</li>
</ul>

## Group Work Policy

You can group up for assignment and project following rules:

<ul>
<li class="fragment">Keep it to **3 people** or fewer.</li>
<li class="fragment">Your group **must stay the same** throughout the semester.</li>
<li class="fragment">Assignments must include group names **and contributions**.</li>
<li class="fragment">Groups are graded **together** on the assigments and project.</li>
</ul>

# 

## Course Topics Overview

## Expected Topics: Traditional ML

<div class="txt-left">
- Bayesian Decision Theory
- Nonmetric Methods: Decision Trees and Random Forests
- Linear Discriminants and Perceptrons
- Support Vector Machines
- Parametric and Non-Parametric Techniques
- Clustering and Expectation Maximization
- Component Analysis and Dimensionality Reduction
- Boosting and Classifier Ensembles
</div>

## Expected Topics: Deep Learning and AI

<div class="txt-left">
- Neural Networks
- CNNs
- RNNs
- Reinforcement Learning
- Ethics and Meta-issues with ML / AI
</div>

## (Un)Expected Topics

If you are interested in something, e-mail me and we can cover it.

## Course Content Overview: Background

<div class="txt-left fragment">
*Linear Algebra:*
: Matrix Inner / Outer Products
: Derivatives, Determinants, Trace
: Eigenvectors / Eigenvalues
</div>
<div class="txt-left fragment">
*Probability Theroy:*
: Discrete Random Variables, Dependence
: Expected Value, Standard Deviation
: Gaussian Distributions
</div>
<div class="txt-left fragment">
*Information Theory:*
: Entropy
: Mutual Information
: Computational Complexity
</div>

# 
## Next Class

## Mathematical Background and Coding Intro

On **Thursday** we will cover:

<ul>
<li>Machine Learning design philosophy and project setup</li>
<li class="fragment">Introduction to programming environment and resources</li>
<li class="fragment">Mathematical background and foundations</li>
</ul>
## Instructor Information

<div class="txt-left">
Scott Doyle

PAS, BME, BMI

Downtown: 4205 JSMBS (M, W, F)<br/>
North: 215-J Bonner Hall (Tu, Th)

<i class="fa fa-envelope"></i> scottdoy@buffalo.edu<br/>
<i class="fa fa-phone"></i> 716.829.2005<br/>
<i class="fa fa-globe"></i> [***www.scottdoy.com***](http://www.scottdoy.com)
</div>

## Thank You!

