---
title: "Attention in Vision"
date: "2022-08-22T13:01:24.320Z"
---
In this article, popular attention mechanisms in computer vision will be examined and implemented in PyTorch. 
Particularly, each module's methodology is analyzed, with detailed descriptions, visualizations, and code, plus optional mathematical explanations
for those preferring more formal descriptions. Results on ImageNet are also included.
<!--more-->The GitHub repository for this article can be found [here](https://github.com/BobMcDear/Attention-in-Vision).

---



# Table of Contents

1. **[Introduction](#introduction)**<br>
2. **[Sqeueeze-and-Excitaiton](#squeeze-and-excitation)**<br>
3. **[Efficient Channel Attention](#efficient-channel-attention)** 
4. **[Convolutional Block Attention Module](#convolutional-block-attention-module)**
5. **[Bottleneck Attention Module](#bottleneck-attention-module)**
6. **[Gather-Excite](#gather-excite)**
7. **[Selective Kernel](#selective-kernel)**

<div id="introduction"></div>

---

# 1. Introduction
As the name suggests, attention enables deep learning models to pay attention to certain parts of the data<sup>[[1]](#general_survey)</sup>, similar to how attention in humans works<sup>[[10]](#kan)</sup>. For instance, in natural language processing (NLP), attention-based models can attend more
to words that convey more information and ignore ones that are somewhat akin to noise and do not contribute
much to the meaning of the text<sup>[[14]](#ng)</sup> ([figure 1.1](#figure_1.1)).<br><br>
<div id="figure_1.1"></div>
<img src="/posts/attention-in-vision/figure_1.1.png">
<center><sub>Figure 1.1. The words crucial to understanding this sentence are highlighted in red. Image by the author.
</sub></center><br>
However, there is no objective importance associated with each word that holds across all cases. For example, one can disregard the words "growled" and "stranger" when deducing the colour of the dog, as they have no bearing on it, and only the first few words are necessary
to determine the answer (<a href="#figure_1.2">figure 1.2</a>). A pivotal characteristic of attention in deep learning 
is that it is conditioned on the input and is sufficiently versatile to manage such situations.<br><br>
<div id="figure_1.2"></div>
<img src="/posts/attention-in-vision/figure_1.2.png">
<center><sub>Figure 1.2. Exclusively the red words pertain to the colour of the dog, and the rest 
of the sentence is irrelevant. Ibid.
</sub></center><br>
In computer vision, a similar idea would be for a neural network to focus on a particular part of an image. <a href="#figure_1.3">Figure 1.3</a>
exhibits this concept; a model would greatly benefit from an attention mechanism that could 
focus on the lionesses.<br><br>
<div id="figure_1.3"></div>
<img src="/posts/attention-in-vision/figure_1.3.png">
<center><sub>Figure 1.3. By attending to the lionesses,
a model can perform more accurate object recognition. Image from ImageNet<sup><a href="#imagenet">[4]</a></sup>.
</sub></center><br>
There are four major kinds of attention in computer vision<sup><a href="#cv_survey">[6]</a></sup>, with in-depth explanations throughout the article:
<br><br>

- Channel attention: _What_ to attend to
- Spatial attention: _Where_ to attend to
- Branch attention: _Which branch_ to attend to
- Temporal attention: _When_ to attend to<br><br>

The first three are covered in this article, but temporal attention is not discussed since temporal deep learning in general 
is a complicated topic that demands a separate article.<br><br>[Equation 1.1](#equation_1.1), proposed in [[6]](#cv_survey), is a general definition of attention
for an arbitrary input \\(X\\) that is helpful for gaining a high-level, intuitive understanding of attention. \\(g\\) is a function that
generates a set of attention values for every element (i.e., how much to attend to, that is, the importance of, each element), 
and \\(f\\) processes \\(X\\) using the output of \\(g\\), e.g., multiply every element by its attention value so the more important ones would have larger magnitudes and be more salient.
<div id="equation_1.1"></div>
$$
\textrm {Attention} := f(g(X), X)
$$
<center><sub>Equation 1.1.
</sub></center><br>
Finally, <a href="#table_1.1">table 1.1</a> summarizes recurring mathematical notations that will be referred to in this article.<br><br>

<div id="table_1.1"></div>
<center>

| **Notation**                      | **Definition**                  |
|-----------------------------------|---------------------------------|
| \\(s\\) (lowercase italicized letter)              | Scaler                          |
| \\(\bold v\\) (lowercase bold letter)     | Vector                          |
| \\(\bold M\\) (uppercase bold letter)     | Matrix                          |
| \\(T\\) (uppercase italicized letter) | Tensor with a minimum rank of 3 |
| \\(\bold v_i\\) | \\(i^{\textrm{th}}\\) element of \\(\bold v\\); one-based indexing|
| \\(\bold M_{i,j}\\) | Entry of \\(\bold M\\) at row \\(i\\) and column \\(j\\); one-based indexing|
| \\(T_{i,j,k}\\) | Entry of \\(\bold T\\) with index \\(i\\) for the first axis, \\(j\\) for the second one, \\(k\\) for the third, and so forth; one-based indexing|
| \\(:\\)                              | Used to mean entire axis when indexing matrices or tensors, e.g., \\(\bold M_{i,:}\\) accesses row \\(i\\)                          |
| \\(\delta\\)                              | ReLU                            |
| \\(\sigma\\)                           | Sigmoid                         |
| \\(\odot\\)                               | Element-wise product            |
| \\(*\\)                                 | Convolution operation           |
| \\(\textrm{BN}\\)                                 | Batch normalization; parameters not shown           |

</center><center><sub>Table 1.1. Mathematical notations prevalent in this article. Common ones, such as \(\sum\) (summation), are not included.</sub></center>

<div id="squeeze-and-excitation"></div>

---

# 2. Squeeze-and-Excitation
One of the earliest and most impactful attention mechanisms in computer vision is squeeze-and-excitation (SE)<sup>[[9]](#se)</sup>, an algorithm ubiquitous in numerous state-of-the-art (SOTA) networks such as EfficientNet<sup>[[21]](#eff)[[22]](#eff2)</sup>, Normalizer-Free Networks (NFNet)
<sup>[[2]](#charac)</sup><sup>[[3]](#high_perf)</sup>, and RegNet<sup>[[16]](#reg)</sup>. Research evinces various channels in a feature map represent different objects<sup>[[6]](#cv_survey)</sup>, and channel attention enables the network to dynamically focus on salient channels, i.e., objects.<br><br>Unsurprisingly, SE is constituted of two ingredients; the squeeze module, where the activations of each channel are aggregated through global average pooling, and the excitation module, whose job is to capture channel interactions through a bottleneck multilayer perceptron (MLP). Formally,
let \\(X \isin \reals^{c \times h \times w} \\), \\(F_{sq}\\), and \\(F_{ex}\\) be the input, squeeze module, and excitation
module respectively. \\(F_{sq}\\), i.e., global average pooling, is applied over \\(X\\) to obtain \\(\bold z \isin \reals^{c}\\) ([equation 2.1](#equation_2.1)). Without global pooling, the excitation module would not have access to global information and be confined to a small receptive field.
<div id="equation_2.1"></div>
$$
\bold z_{c'} := \frac {1} {hw} \sum_{h' = 1} ^{h} \sum_{w' = 1} ^{w} X_{c',h',w'}
$$
<center><sub>Equation 2.1. \(\forall c' \isin {\{1, 2, ..., c\}}\).
</sub></center><br>
\(\bold z\) is transformed with \(F_{ex}\) to acquire an attention vector \(\bold a \isin \reals^{c}\), where
\(\bold a_{c'}\) may be viewed as the importance of the \(c'^{\textrm{th}}\) channel for this particular input.
\(F_{ex}\) is an MLP containing two linear layers with weight matrices \(\bold W_{1} \isin \reals^{\frac {c} {r} \times c}\) and \(\bold W_{2} \isin \reals^{c \times \frac {c} {r}}\), plus ReLU between them and the sigmoid function at the end (<a href="#equation_2.2">equation 2.2</a>). \(r\) is a hyperparameter that controls the size of the bottleneck, which assuages the computational complexity and helps generalization.

<div id="equation_2.2"></div>
$$
\bold a := F_{ex}(\bold z; \bold W_1, \bold W_2) :=  \sigma (\bold W_{2} \delta (\bold W_{1}\bold z))
$$
<center><sub>Equation 2.2.
</sub></center><br>

\\(\tilde {X} \isin \reals^{c \times h \times w}\\), the final output, is computed by multiplying every feature map by its associated attention value from \\(\bold a \\) (<a href="#equation_2.3">equation 2.3</a>). 

<div id="equation_2.3"></div>
$$
\tilde{X}_{c',h', w'} := \bold a_{c'}X_{c', h', w'}
$$
<center><sub>Equation 2.3. \(\forall c' \isin {\{1, 2, ..., c\}}, \forall h' \isin {\{1, 2, ..., h\}}, \forall w' \isin {\{1, 2, ..., w\}}\).
</sub></center><br>

A graphical illustration can be found in [figure 2.1](#figure_2.1), and SE is implemented in [snippet 2.1](#snippet_2.1).<br><br>
<div id="figure_2.1"></div>
<img src="/posts/attention-in-vision/figure_2.1.png">
<center><sub>Figure 2.1. Overview of squeeze-and-excitation. Image from figure 1 of 
<a href="#se">[9]</a>.
</sub></center><br>

<div id="snippet_2.1"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	AdaptiveAvgPool2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	Sigmoid,
	)


class SE(Module):
	"""
	Squeeze-and-excitation
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor for the 
			bottleneck layer.
			Default is 16
		"""
		super().__init__()

		bottleneck_dim = in_dim//reduction_factor
		self.squeeze = AdaptiveAvgPool2d(
			output_size=1,
			)
		self.excitation = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			ReLU(),
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
			Sigmoid(),
			) 
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		squeezed = self.squeeze(input)
		attention = self.excitation(squeezed)
		
		output = attention*input
		return output
{{< / highlight >}}
<center><sub>Snippet 2.1. Owing to the spatial dimensions being \(1\) x \(1\), a convolution of kernel size \(1\) x \(1\) and a fully-connected linear layer would be identical.</sub></center>
<br>

Empirically, squeeze-and-excitation performs very well. By prepending SE immediately before the residual addition 
of the non-identity branch of each residual block in popular convolutional neural networks (CNN), top-1 accuracy is boosted
on ImageNet ([table 2.1](#table_2.1)).<br><br>
<center>
<div id="table_2.1"></div>

| **Architecture** | **Plain** | **With SE** |
|------------------|-----------|-------------|
| ResNet-50        | 75.20%    | 76.71%      |
| ResNet-101       | 76.83%    | 77.62%      |
| ResNet-152       | 77.58%    | 78.43%      |
| ResNeXt-50       | 77.89%    | 78.90%      |
| ResNeXt-101      | 78.82%    | 79.30%      |
| MobileNet        | 71.60%    | 74.70%      |
<center><sub>Table 2.1. Top-1 accuracy of CNNs, with or without SE, on ImageNet.</sub></center>
</center>

<div id="efficient-channel-attention"></div>

---

# 3. Efficient Channel Attention

[[23]](#eca) discovered a deficiency of SE 
that degrades performance and proposed efficient channel attention (ECA) to remedy it for better accuracy and efficiency. 

An important aspect of squeeze-and-excitation is the bottleneck in the excitation module, where dimensionality reduction (DR) is performed to parsimoniously compute channel interactions. Due to DR, the attention values and their associated channels have only indirect correspondence, and there is no direct mapping between the channels and their attention values ([figure 3.1](#figure_3.1)).<br><br>
<div id="figure_3.1"></div>
<img src="/posts/attention-in-vision/figure_3.1.png">
<center><sub>Figure 3.1.
Dimensionality reduction in the excitation module of SE.
Image by the author.
</sub></center><br>
Three derivatives of squeeze-and-excitation, differing in terms of their
excitation modules, were developed to measure the ramifications of dimensionality reduction. SE-Var-1, where the excitation module is merely
the sigmoid function (<a href="#figure_3.2">figure 3.2</a>), SE-Var-2, where the excitation module element-wise multiplies the pooled channels by learnable weights, followed, as usual, by sigmoid (<a href="#figure_3.3">figure 3.3</a>), and SE-Var-3, where the excitation module is a fully-connected layer, again
succeeded by sigmoid (<a href="#figure_3.4">figure 3.4</a>). SE-Var-3 is the same as effective
squeeze-and-excitation (eSE)<sup><a href="#ese">[11]</a></sup>, proposed as a refinement over SE for instance segmentation in a separate, unrelated paper.<br><br>
<div id="figure_3.2"></div>
<img src="/posts/attention-in-vision/figure_3.2.png">

<center><sub>Figure 3.2. The excitation module of SE-Var-1. Ibid.
</sub></center><br>
<div id="figure_3.3"></div>
<img src="/posts/attention-in-vision/figure_3.3.png">
<center><sub>Figure 3.3. The excitation module of SE-Var-2. Sigmoid is not shown for simplicity. Ibid.</sub></center><br>
<div id="figure_3.4"></div>
<img src="/posts/attention-in-vision/figure_3.4.png">
<center><sub>Figure 3.4. The excitation module of SE-Var-3. Sigmoid is not shown for simplicity. Ibid.</sub></center><br>

The mathematical formulations for SE-Var-1, SE-Var-2, and SE-Var-3 are nearly identical to squeeze-and-excitation, except for the definition of their excitation modules. [Equations 3.1](#equation_3.1),
[3.2](#equation_3.2), and <a href="#equation_3.3">3.3</a> define \\(F_{ex}\\) for these variants, where \\(\bold z \isin \reals^{c}\\) is the output of the squeeze module, that is, the output of global average pooling. [Snippet 3.1](#snippet_3.1) implements them.

<div id="equation_3.1"></div>
$$
F_{ex}(\bold z) := \sigma (\bold z) 
$$
<center><sub>Equation 3.1. The excitation module for SE-Var-1.</sub></center><br>
<div id="equation_3.2"></div>
$$
F_{ex}(\bold z; \bold w) := \sigma (\bold w \odot \bold z) 
$$
<center><sub>Equation 3.2. The excitation module for SE-Var-2. \(\bold w \isin \reals^{c} \).</sub></center><br>
<div id="equation_3.3"></div>
$$
F_{ex}(\bold z; \bold W) := \sigma (\bold W \bold z) 
$$
<center><sub>Equation 3.3. The excitation module for SE-Var-3. \(\bold W \isin \reals^{c \times c} \)</sub></center><br>

<div id="snippet_3.1"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	AdaptiveAvgPool2d,
	Conv2d,
	Module,
	Sequential,
	Sigmoid,
	)


class SEVar1(Module):
	"""
	Squeeze-and-excitation, variant 1
	"""
	def __init__(
		self, 
		) -> None:
		"""
		Sets up the modules
		"""
		super().__init__()

		self.squeeze = AdaptiveAvgPool2d(
			output_size=1,
			)
		self.excitation = Sigmoid()
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		squeezed = self.squeeze(input)
		attention = self.excitation(squeezed)
		
		output = attention*input
		return output


class SEVar2(Module):
	"""
	Squeeze-and-excitation, variant 2
	"""
	def __init__(
		self, 
		in_dim: int,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
		"""
		super().__init__()

		self.squeeze = AdaptiveAvgPool2d(
			output_size=1,
			)
		self.excitation = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=in_dim,
				kernel_size=1,
				groups=in_dim,
				),
			Sigmoid(),
			) 
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		squeezed = self.squeeze(input)
		attention = self.excitation(squeezed)
		
		output = attention*input
		return output


class SEVar3(Module):
	"""
	Squeeze-and-excitation, variant 3
	"""
	def __init__(
		self, 
		in_dim: int,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
		"""
		super().__init__()

		self.squeeze = AdaptiveAvgPool2d(
			output_size=1,
			)
		self.excitation = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
			Sigmoid(),
			) 
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		squeezed = self.squeeze(input)
		attention = self.excitation(squeezed)
		
		output = attention*input
		return output
{{< / highlight >}}
<center><sub>Snippet 3.1. For SE-Var-2, a convolution with \(c\) groups element-wise multiplies the input by a set of weights.</sub></center>
<br>
Since neither of these variants employs dimensionality reduction, pitting them against SE 
on ImageNet with ResNet-50 as the backbone offers an idea of the 
consequences of DR (<a href="#table_3.1">table 3.1</a>). Nonetheless, this is not an apples-to-apples comparison because SE-Var-1 and SE-Var-2
do not model channel relationships and are therefore technically not channel attention modules.<br><br>
<center>
<div id="table_3.1"></div>

| **Variant** | **Top-1 accuracy** |
|------------------|--------------------|
| Original SE        | 76.71%             |
| SE-Var-1            | 76.00%             |
| SE-Var-2            | 77.07%             |
| SE-Var-3            | 77.42%             |

</center>
<center><sub>
Table 3.1. Accuracy of SE, SE-Var-1, SE-Var-2, and SE-Var-3 on ImageNet.
</sub></center><br>
SE-Var-2 attests to the merit of direct channel-to-attention communication, for it is far lighter than SE but exceeds its score. 
Its principal drawback is that there are no inter-channel interactions, 
an issue that is solved by the more accurate SE-Var-3 using a linear layer that 
connects every channel to every attention value. Sadly, SE-Var-3’s cost is quadratic,
and there being no bottleneck, the computational complexity is markedly increased.<br><br>
A compromise between SE-Var-2 and SE-Var-3 is made; in lieu of SE-Var-3's <i>global</i> cross-channel attention, where every two channels
interact with one another, <i>local</i> cross-channel relationships are utilized. Concretely, each channel interacts only with its \(k\) neighbours
through linear layers that operate on \(k\) channels at a time.
To further reduce the parameter count, the linear layers can share parameters, thereby being equivalent to a one-dimensional
convolution sliding over the pooled channels. This final version, illustrated in <a href="#figure_3.5">figure 3.5</a> and formulated
mathematically in <a href="#equation_3.4">equation 3.4</a>, is ECA and has the advantages of both inter-channel interactions and direct channel-to-attention communication whilst preserving, for the most part, the efficiency of SE-Var-2.<br><br>
<div id="figure_3.5"></div>
<img src="/posts/attention-in-vision/figure_3.5.png">
<center><sub>Figure 3.5. 
In this example, \(k = 3\). The grey nodes represent padding for maintaining the number of 
features. Though not specified in this figure, 
the parameters of these connections are shared. Ibid.</sub></center>
<div id="equation_3.4"></div>
$$
F_{ex}(\bold z; \bold w) := \sigma (\bold w * \bold z) 
$$
<center><sub>Equation 3.4. The excitation module for ECA. \(\bold w \isin \reals^{k} \). \(\bold z\) is typically padded beforehand.</sub></center>
<br>
The computational complexity of SE-Var-2, SE-Var-3, and ECA are outlined in <a href="#table_3.2">table 3.2</a>. Notably,
assuming \(k \ll c \), which is virtually always true, ECA is far more efficient than SE-Var-3 and is almost on par with
Se-Var-2 in terms of complexity. It is also more efficient than the original SE.<br><br>
<div id="table_3.2"></div>
<center>

| **Variant** | **Complexity** |
|------------------|--------------------|
| SE-Var-2         | \\(\mathcal{O}(c)\\)|
| SE-Var-3         | \\(\mathcal{O}(c^2)\\)            |
| ECA            | \\(\mathcal{O}(kc)\\)            |
</center>
<center><sub>
Table 3.2. Computational complexity of SE-Var-2, SE-Var-3, and ECA.
</sub></center>
<br>
The PyTorch implementation of ECA is in <a href="#snippet_3.2">snippet 3.2</a>, and its
chief difference from the other variants is that the output of the average pooling layer
must be in a channels-last format to be able to be
passed to a one-dimensional convolution. These modifications
need to be reverted before the output is returned.<br><br>
<div id="snippet_3.2"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	AdaptiveAvgPool2d,
	Conv1d,
	Module,
	Sequential,
	Sigmoid,
	)


class ECA(Module):
	"""
	Efficient channel attention
	"""
	def __init__(
		self, 
		kernel_size: int = 3,
		) -> None:
		"""
		Sets up the modules

		Args:
			kernel_size (int): Kernel size.
			Default is 3
		"""
		super().__init__()

		padding = kernel_size//2
		self.avg_pool = AdaptiveAvgPool2d(
			output_size=1,
			)
		self.conv = Conv1d(
			in_channels=1,
			out_channels=1,
			kernel_size=kernel_size,
			padding=padding,
			bias=False,
			)
		self.sigmoid = Sigmoid() 
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		avg_pooled = self.avg_pool(input)

		avg_pooled = avg_pooled.squeeze(2)
		avg_pooled = avg_pooled.transpose(1, 2)

		attention = self.conv(avg_pooled)
		attention = self.sigmoid(attention)

		attention = attention.transpose(1, 2)
		attention = attention.unsqueeze(2)
		
		output = attention*input
		return output
{{< / highlight >}}
<center><sub>Snippet 3.2.</sub></center>
<br>
ECA outperforms SE with ResNets of different depths and MobileNetV2 at a lower computational cost and parameter count. The positioning of ECA is identical to that of SE.<br><br>
<div id="table_3.3"></div>
<center>

| **Architecture** | **Plain** | **With SE** | **With ECA** |
|------------------|-----------|-------------|--------------|
| ResNet-50        | 75.20%    | 76.71%      | 77.48%       |
| ResNet-101       | 76.83%    | 77.62%      | 78.65%       |
| ResNet-152       | 77.58%    | 78.43%      | 78.92%       |
| MobileNetV2       | 71.64%    | 72.42%      | 72.56%       |
<center><sub>Table 3.3. Top-1 accuracy of ResNets and MobileNetV2, with SE, ECA, or neither, on ImageNet.</sub></center>
</center><br>

Lastly, it must be noted that \\(k\\) is an important hyperparameter that should be tuned. The authors propound a heuristic to automatically select the kernel size given the number of channels that raises accuracy, but they mysteriously dispense with it and do not inspect it further. Hence, it was not discussed here either.

<div id="convolutional-block-attention-module"></div>

---

# 4. Convolutional Block Attention Module

Following the efficacy of attention-based networks, <a href="#cbam">[24]</a> proposed convolutional block attention module (CBAM), another attention mechanism for CNNs. CBAM is a hybrid attention module that mixes channel attention with spatial attention. Whereas channel attention can be thought of as a dynamic module for determining what channels to focus on, spatial attention dynamically decides where to focus on, i.e., which spatial positions or pixels are more important.

The channel attention module of CBAM is similar to SE, but the two diverge in one aspect, viz., CBAM also utilizes max pooling in addition to average pooling. Concretely, there are two squeeze modules, \\(F_{sq}^1\\) and \\(F_{sq}^2\\), corresponding to global average and global max pooling, that transform the input \\(X \isin \reals^{c \times h \times w}\\) to get \\(\bold z_1 \isin \reals^{c}\\) and \\(\bold z_2 \isin \reals^{c}\\) ([equations 4.1](#equation_4.1) and [4.2](#equation_4.2)). Intuitively, \\(\bold z_1\\) is a smooth
representation of every channel, but \\(\bold z_2\\) contains the most noticeable activation for each one, and they thus complement one another.

<div id="equation_4.1"></div>
$$
\bold z_{1_{c'}} := F_{sq}^1(X) := \frac {1} {hw} \sum_{h' = 1} ^{h} \sum_{w' = 1} ^{w} X_{c',h',w'}
$$
<center><sub>Equation 4.1. \(\forall c' \isin {\{1, 2, ..., c\}}\).
</sub></center><br>
<div id="equation_4.2"></div>
$$
\bold z_{2_{c'}} := F_{sq}^2(X) := \max (X_{c',:,:})
$$
<center><sub>Equation 4.2. \(\forall c' \isin {\{1, 2, ..., c\}}\).
</sub></center>
<br>
Since \(\bold z_1\) and \(\bold z_2\) are semantically related, they do not require separate MLPs for modelling channel relationships, and a single MLP, again parameterized by \(\bold W_{1} \isin \reals^{\frac {c} {r} \times c}\) and \(\bold W_{2} \isin \reals^{\frac {c} {r} \times c}\), transforms them both to obtain two attention vectors, \(\bold a'_1 \isin \reals^{c}\) and \(\bold a'_2 \isin \reals^{c}\) (<a href="#equation_4.3">equations 4.3</a> and <a href="#equation_4.4">4.4</a>).
<div id="equation_4.3"></div>
$$
\bold a'_1 := \bold W_{2} \delta (\bold W_{1}\bold z_1)
$$
<center><sub>Equation 4.3.
</sub></center><br>
<div id="equation_4.4"></div>
$$
\bold a'_2 := \bold W_{2} \delta (\bold W_{1}\bold z_2)
$$
<center><sub>Equation 4.4.
</sub></center><br>
Sigmoid was not applied because \(\bold a'_1\) and \(\bold a'_2\) first need to be combined through summation, and then they can be normalized to acquire a single attention vector \(\bold a \isin \reals^{c}\) (<a href="#equation_4.5">equation 4.5</a>), with the remainder of the process being the same as SE, i.e., multiplying each channel in the original input by its attention value from \(\bold a\).
<div id="equation_4.5"></div>
$$
\bold a := \sigma (\bold a'_1 + \bold a'_2)
$$
<center><sub>Equation 4.5.
</sub></center><br>
<a href="#figure_4.1">Figure 4.1</a> illustrates this process, and it is implemented in <a href="#snippet_4.1">snippet 4.1</a>.<br><br>
<div id=figure_4.1></div>
<img src="figure_4.1.png">
<center><sub>Figure 4.1. Demonstration of CBAM's channel attention module. Multiplication of the input by the attention vector is not shown. Image from figure 2 of <a href="#cbam">[24]</a>.
</sub></center><br>
<div id="snippet_4.1"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	AdaptiveAvgPool2d,
	AdaptiveMaxPool2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	Sigmoid,
	)


class CBAMChannelAttention(Module):
	"""
	CBAM's channel attention
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor for the 
			bottleneck layer.
			Default is 16
		"""
		super().__init__()

		self.avg_pool = AdaptiveAvgPool2d(
			output_size=1,
			)
		self.max_pool = AdaptiveMaxPool2d(
			output_size=1,
			)

		bottleneck_dim = in_dim//reduction_factor
		self.mlp = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			ReLU(),
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
			)
		self.sigmoid = Sigmoid()
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		avg_pooled = self.avg_pool(input)
		max_pooled = self.max_pool(input)

		avg_attention = self.mlp(avg_pooled)
		max_attention = self.mlp(max_pooled)
		
		attention = avg_attention+max_attention
		attention = self.sigmoid(attention)
		
		output = attention*input
		return output
{{< / highlight >}}
<center><sub>Snippet 4.1. Owing to the spatial dimensions being \(1\) x \(1\), a convolution of kernel size \(1\) x \(1\) and a fully-connected linear layer would be identical.</sub></center>
<br>
Compared to average pooling alone, this version of channel attention attains better accuracy on ImageNet with a ResNet-50 (<a href="#table_4.1">table 4.1</a>).<br><br>
<center>
<div id="table_4.1"></div>

| **Channel attention** | **Top-1 accuracy** |
|------------------|--------------------|
| SE         | 76.86%             |
| CBAM's channel attention            | 77.20%             |

</center>
<center><sub>
Table 4.1. Accuracy of SE versus CBAM's channel attention.
</sub></center><br>
CBAM’s channel attention can also be tailored for spatial attention. To do so, the data is first average and max pooled along the channel axis; that is, rather than aggregating the activations of each channel, the activations of every spatial position \((h', w')\) are aggregated to get \(\bold Z_1 \isin \reals^{h \times w}\) and \(\bold Z_2 \isin \reals^{h \times w}\) (<a href="#equation_4.6">equations 4.6</a> and <a href="#equation_4.7">4.7</a>).

<div id="equation_4.6"></div>
$$
\bold Z_{1_{h', w'}} := \frac {1} {c} \sum_{c' = 1} ^{c} X_{c',h',w'}
$$
<center><sub>Equation 4.6. \(\forall h' \isin {\{1, 2, ..., h\}}\), \(\forall w' \isin {\{1, 2, ..., w\}}\).
</sub></center><br>
<div id="equation_4.7"></div>
$$
\bold Z_{2_{h', w'}} := \max(X_{:,h',w'})
$$
<center><sub>Equation 4.7. \(\forall h' \isin {\{1, 2, ..., h\}}\), \(\forall w' \isin {\{1, 2, ..., w\}}\).
</sub></center><br>
\(\bold Z_1\) and \(\bold Z_2\) are concatenated to acquire a double-channelled tensor \(Z \isin \reals^{2 \times h \times w}\). \(Z\) can be seen as the equivalent of \(\bold z\) from SE or \(\bold z_1\) and \(\bold z_2\) from CBAM's channel attention. Therefore, the next step is to use this descriptor to capture spatial interactions. Naively, an MLP may be employed, as was the case with channel attention, but that would not be practical for a few reasons. First, for large feature maps early in the network, a fully-connected layer would be too inefficient because of its quadratic cost. Second, a fully-connected layer’s input dimension is static, meaning it would not be usable for any data size other than what it was trained on. Finally, it would have properties like variance to translation (i.e., a small shift in the input would wholly change the output), which are not suitable for spatial data.
<br><br>
Instead, convolutions would be more apposite; they are inexpensive, able to manage different resolutions, and have inductive biases such as invariance to translation that are appropriate for spatial data. Ergo, CBAM uses convolutions to model spatial dependencies without the defects of a multilayer perceptron (albeit one disadvantage is that the receptive field is confined to the kernel size).
<br><br>
An attention matrix \(\bold A \isin \reals^{h \times w}\) is extracted by running \(Z\) through a \(k\) x \(k\) convolutional layer with weight matrix \(W \bold \isin \reals^{1 \times 2 \times k \times k}\) and normalizing it with sigmoid (<a href="#equation_4.8">equation 4.8</a>). The strategy for calculating the output, \(\tilde{X} \isin \reals^{c \times h \times w}\), is similar to that of the channel attention module; each activation is multiplied by its attention value from \(\bold A \) (<a href="#equation_4.9">equation 4.9</a>).
<div id="equation_4.8"></div>
$$
\bold A := \sigma (W * Z) 
$$
<center><sub>Equation 4.8. \( Z\) is typically padded beforehand to retain the spatial dimensions.</sub></center><br>
<div id="equation_4.9"></div>
$$
\tilde{X}_{c', h', w'} := \bold A_{h',w'}X_{c',h',w'}
$$
<center><sub>Equation 4.9. \(\forall c' \isin {\{1, 2, ..., c\}}\), \(\forall h' \isin {\{1, 2, ..., h\}}\), \(\forall w' \isin {\{1, 2, ..., w\}}\).
</sub></center><br>
<a href="#figure_4.2">Figure 4.2</a> depicts CBAM's spatial attention module, and it is implemented in <a href="#snippet_4.2">snippet 4.2</a>.<br><br>
<div id=figure_4.2></div>
<img src="figure_4.2.png">
<center><sub>Figure 4.2. Demonstration of CBAM's spatial attention module. Ibid.
</sub></center><br>
<div id="snippet_4.2"></div>
{{< highlight python "linenos=true">}}
from torch import (
	Tensor,                                                          
	cat,
	)
from torch.nn import (
	Conv2d,
	Module,
	Sigmoid,
	)


class ChannelAvgPool(Module):
	"""
	Average pool along the channel axis
	"""
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		output = input.mean(
			dim=1,
			keepdim=True,
			)
		return output


class ChannelMaxPool(Module):
	"""
	Max pool along the channel axis
	"""
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		output = input.max(
			dim=1,
			keepdim=True,
			).values
		return output


class CBAMSpatialAttention(Module):
	"""
	CBAM's spatial attention
	"""
	def __init__(
		self, 
		kernel_size: int = 7,
		) -> None:
		"""
		Sets up the modules

		Args:
			kernel_size (int): Kernel size.
			Default is 7
		"""
		super().__init__()

		self.avg_pool = ChannelAvgPool()
		self.max_pool = ChannelMaxPool()

		padding = kernel_size//2
		self.conv = Conv2d(
			in_channels=2,
			out_channels=1,
			kernel_size=kernel_size,
			padding=padding,
			)
		self.sigmoid = Sigmoid()
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		avg_pooled = self.avg_pool(input)
		max_pooled = self.max_pool(input)
		pooled = cat(
			tensors=[avg_pooled, max_pooled],
			dim=1,
			)

		attention = self.conv(pooled)
		attention = self.sigmoid(attention)		
		
		output = attention*input
		return output
{{< / highlight >}}
<center><sub>Snippet 4.2.</sub></center><br>
These two modules (channel and spatial attention) are run sequentially to get CBAM (<a href="#snippet_4.3">snippet 4.3</a>).<br><br>
<div id="snippet_4.3"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                            
from torch.nn import Module


class CBAM(Module):
	"""
	Convolutional block attention module
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		kernel_size: int = 7,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor for the 
			bottleneck layer in the channel attention module.
			Default is 16
			kernel_size (int): Kernel size in the spatial 
			attention module.
			Default is 7
		"""
		super().__init__()

		self.channel_attention = CBAMChannelAttention(
			in_dim=in_dim,
			reduction_factor=reduction_factor,
			)
		self.spatial_attention = CBAMSpatialAttention(
			kernel_size=kernel_size,
			)
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		output = self.channel_attention(input)
		output = self.spatial_attention(output)
		return output
{{< / highlight >}}
<center><sub>Snippet 4.3.</sub></center>
<br>
Thorough experiments on ImageNet reveal that CBAM typically outperforms SE with a host of backbones. Some are included in <a href="#table_4.2">table 4.2</a>. Once again, CBAM is situated before the residual summation of the non-identity branch of each residual block.<br><br>
<center>
<div id="table_4.2"></div>

| **Architecture**    | **Plain** | **With SE** | **With CBAM** |
|---------------------|-----------|-------------|---------------|
| ResNet-50           | 75.44%    | 76.86%      | 77.34%        |
| ResNet-101          | 76.62%    | 77.65%      | 78.49%        |
| ResNeXt-50          | 77.15%    | 78.09%      | 78.08%        |
| ResNeXt-101         | 78.46%    | 78.83%      | 78.93%        |
| MobileNet | 68.61%    | 70.03%      | 70.99%        |

</center>
<center><sub>
Table 4.2. Top-1 accuracy of SE, CBAM, and vanilla networks on ImageNet.
</sub></center>

<div id="bottleneck-attention-module"></div>

---

# 5. Block Attention Module
Bottleneck attention module (BAM)<sup>[[15]](#bam)</sup> was released concurrently with CBAM by the same researchers, and the overall philosophy remains intact (i.e., SE-esque channel attention plus spatial attention). However, BAM's attention modules differ from the ones considered so far, in that they return unnormalized attention tensors as opposed to returning the input multiplied by normalized attention values, for reasons that shall be explained shortly. 

The channel attention module is nearly indistinguishable from SE, but batch normalization is applied before ReLU and, as mentioned, sigmoid is left out. [Equation 5.1](#equation_5.1) describes how this unnormalized attention vector, \\(\bold a^′ \isin \reals^{c}\\), is calculated, with weight matrices \\(\bold W_{1} \isin \reals^{\frac {c} {r} \times c}\\) and \\(\bold W_{2} \isin \reals^{c \times \frac {c} {r}}\\), and [snippet 5.1](#snippet_5.1) implements it.

<div id="equation_5.1"></div>
$$
\bold a' := \bold W_{2} \delta (\textrm{BN}(\bold W_{1}\bold z))
$$
<center><sub>Equation 5.1.
</sub></center><br>
<div id="snippet_5.1"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	AdaptiveAvgPool2d,
	BatchNorm2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	)


class BAMChannelAttention(Module):
	"""
	BAM's channel attention
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		) -> None:
		"""
		Sets up the modules
		
		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor for the 
			bottleneck layer.
			Default is 16
		"""
		super().__init__()

		self.avg_pool = AdaptiveAvgPool2d(
			output_size=1,
			)

		bottleneck_dim = in_dim//reduction_factor
		self.mlp = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			BatchNorm2d(
				num_features=bottleneck_dim,
				),
			ReLU(),
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
			)
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		avg_pooled = self.avg_pool(input)
		attention = self.mlp(avg_pooled)
		return attention
{{< / highlight >}}
<center><sub>Snippet 5.1. Owing to the spatial dimensions being \(1\) x \(1\), a convolution of kernel size \(1\) x \(1\) and a fully-connected linear layer would be identical.</sub></center><br>
BAM's spatial attention is more nuanced than that of CBAM. The first distinction is that instead of a large-kernelled convolution, two dilated \(3\) x \(3\) convolutions capture spatial interactions. Also, in lieu of pooling for compressing the channels, \(1\) x \(1\) convolutions are utilized. Lastly, this compression happens in two stages; specifically, the channels are initially reduced by a factor of \(r\) (the same reduction factor as that of BAM's channel attention module), spatial relationships are modelled through the \(3\) x \(3\) convolutions, and the number of channels is shrunk a second time, this time to \(1\). Every convolution, other than the last one, is succeeded by batch normalization and ReLU. The mathematical expression for this module would be too convoluted, and the reader is directly referred to <a href="#snippet_5.2">snippet 5.2</a> instead. Again, an unnormalized attention matrix \(\bold A' \isin \reals^{h \times w}\) is outputted.<br><br>
<div id="snippet_5.2"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	BatchNorm2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	)


class BAMSpatialAttention(Module):
	"""
	BAM's spatial attention
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		dilation: int = 4,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor for the 
			bottleneck layer.
			Default is 16
			dilation (int): Dilation for the 3 X 3 convolutions.
			Default is 4
		"""
		super().__init__()

		bottleneck_dim = in_dim//reduction_factor
		self.reduce_1 = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			BatchNorm2d(
				num_features=bottleneck_dim,
				),
			ReLU(),
			)

		self.layers = Sequential(
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=bottleneck_dim,
				kernel_size=3,
				padding=dilation,
				dilation=dilation,
				),
			BatchNorm2d(
				num_features=bottleneck_dim,
				),
			ReLU(),
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=bottleneck_dim,
				kernel_size=3,
				padding=dilation,
				dilation=dilation,
				),
			BatchNorm2d(
				num_features=bottleneck_dim,
				),
			ReLU(),
			)
		
		self.reduce_2 = Conv2d(
			in_channels=bottleneck_dim,
			out_channels=1,
			kernel_size=1,
			)
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		reduced = self.reduce_1(input)
		attention = self.layers(reduced)
		attention = self.reduce_2(attention)
		return attention
{{< / highlight >}}
<center><sub>Snippet 5.2.</sub></center>
<br>
To transform the input using the attention values and get the final output, \(\bold a'\) and \(\bold A'\) are element-wise multiplied by one another and normalized to get a final attention tensor \(A \isin \reals^{c \times h \times w}\) (<a href="#equation_5.2">equation 5.2</a>). The output \(\tilde X \isin \reals^{c \times h \times w}\) is \(X\) element-wise multiplied by \(A + 1\) (<a href="#equation_5.3">equation 5.3</a>). <a href="#snippet_5.3">Snippet 5.3</a> implements the finalized BAM, and <a href="#figure_5.1">figure 5.1</a> visualizes it.
<div id="equation_5.2"></div>
$$
A_{c',h',w'} := \sigma (\bold A'_{h',w'} \bold a'_{c'})
$$
<center><sub>Equation 5.2. \(\forall c' \isin \{1, 2, ..., c\}\), \(h' \isin \{1, 2, ..., h\}\), \(w' \isin \{1, 2, ..., w\}\).
</sub></center><br>
<div id="equation_5.3"></div>
$$
\tilde X := X \odot (A + 1)
$$
<center><sub>Equation 5.3.
</sub></center><br>
<div id="snippet_5.3"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	Module,
	Sigmoid,
	)


class BAM(Module):
	"""
	Bottleneck attention module
	"""
	def __init__(
		self, 
		in_dim: int,
		reduction_factor: int = 16,
		dilation: int = 4,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor in the channel 
			and spatial attention modules.
			Default is 16
			dilation (int): Dilation in the spatial attention module.
			Default is 4
		"""
		super().__init__()

		self.channel_attention = BAMChannelAttention(
			in_dim=in_dim,
			reduction_factor=reduction_factor,
			)
		self.spatial_attention = BAMSpatialAttention(
			in_dim=in_dim,
			reduction_factor=reduction_factor,
			dilation=dilation,
			)
		self.sigmoid = Sigmoid()

	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		channel_attention = self.channel_attention(input)
		spatial_attention = self.spatial_attention(input)

		attention = channel_attention*spatial_attention
		attention = self.sigmoid(attention)
		attention = attention+1

		output = attention*input
		return output
{{< / highlight >}}
<center><sub>Snippet 5.3.</sub></center><br>
<div id="figure_5.1"></div>
<img src="figure_5.1.png">
<center><sub>Figure 5.1. An overview of BAM. The strategy for fusing the channel and spatial attention maps is slightly different in this diagram, as they are added and not multiplied. Image from figure 2 of <a href="#bam">[15]<a>.</sub></center><br>
<a href="#table_5.1">Table 5.1</a> summarizes the top-1 accuracy of BAM with a few architectures.
Unlike SE, ECA, and CBAM, BAM is appended only after each network stage and is not present in every residual block. This table might lead one to believe BAM is inferior to SE and CBAM. Yet, across other tasks, datasets, and architectures, BAM is on par with CBAM, and there is no clear winner between the two.<br><br>
<center>
<div id="table_5.1"></div>

| **Architecture**    | **Plain** | **With SE** | **With CBAM** | **With BAM** |
|---------------------|-----------|-------------|---------------|---------------|
| ResNet-50           | 75.44%    | 76.86%      | 77.34%        | 75.98%        |
| ResNet-101          | 76.62%    | 77.65%      | 78.49%        | 77.56%        |
| ResNeXt-50          | 77.15%    | 78.09%      | 78.08%        | 77.15%        |
| MobileNet | 68.61%    | 70.03%      | 70.99%        | 69.42%        | 

</center>
<center><sub>
Table 5.1. Top-1 accuracy of SE, CBAM, BAM, and vanilla networks on ImageNet.
</sub></center>

<div id="gather-excite"></div>

---

# 6. Gather-Excite

Theoretically, the receptive fields of convolutional neural networks are sufficiently extensive to cover the totality of input images. However, the effective receptive field (ERF) of a network, that is, the actual size of its receptive field in practice, gauged empirically, is much smaller and is not the same as the theoretical receptive field<sup><a href="#eff_rec">[13]</a></sup>. 

For instance, the ERFs of CNNs of different depths with \\(3\\) x \\(3\\) kernels occupy only a fraction of their theoretical receptive fields (<a href="#figure_6.1">figure 6.1</a>). The issue is exacerbated the deeper a network gets, or if there exist non-linearities.<br><br>
<div id="figure_6.1"></div>
<img src="figure_6.1.png">
<center><sub>Figure 6.1. Effective versus theoretical receptive fields. The grids are the same size as the theoretical receptive fields, and the ERFs are depicted by bright pixels. The weights of the kernels are all \(1\)s for uniform and random values for random. Uniform and random do not contain non-linearities. Image from figure 1 of <a href="#eff_rec">[13]</a>.
</sub></center><br>
Hence, spatially distant neurons do not communicate, thereby hindering neural networks’ performance on tasks where adequate long-range interactions are indispensable. Ergo, <a href="#ge">[7]</a> suggests gather-excite (GE), a module that aggregates data from large spatial neighbourhoods via a function \(\xi G\), also known as the gather module, and redistributes the information back to every activation with another function \(\xi E\), also known as the excite module. In other words, the data is first spatially downsampled through \(\xi G\) so the resulting neurons contain information from activations that were previously distant and would thus normally not interact with one another. Then, \(\xi E\) redistributes the information of these neurons to the original activations to essentially force interactions amongst far-off neurons. <a href="#figure_6.2">Figure 6.2</a> provides an overview of GE.<br><br>
<div id="figure_6.2"></div>
<img src="figure_6.2.png">
<center><sub>Figure 6.2. A sketch of gather-excite. \(\xi G\) gathers information from the whitish, transparent squares on the leftmost figure to obtain the middle figure, and the values of the two are mixed through \(\xi E\).
Image from figure 1 of <a href="#ge">[7]</a>.
</sub></center><br>
This might sound unduly abstract, and a concrete example for input \(X \isin \reals^{c \times h \times w}\) would be helpful. \(\xi G\) can be average pooling with a kernel size of \(2e - 1\) and a stride of \(e\). \(e\), known as the extent, is proportional to the size of the receptive fields of the neurons of the output of \(\xi G\), \(Z \isin \reals^{c \times \frac {h} {e} \times \frac {w} {e}}\) (<a href="#equation_6.1">equation 6.1</a>). \(\xi E\) could simply enlarge \(Z\) to the shape of the original input with nearest-neighbour interpolation and normalize it to get an attention tensor \(A \isin \reals^{c \times h \times w}\) (<a href="#equation_6.2">equation 6.2</a>). Akin to the other attention mechanisms, \(A\) is element-wise multiplied by the original input \(X\) to get \(\tilde X \isin \reals^{c \times h \times w}\), the final output (<a href="#equation_6.3">equation 6.3</a>). A special case of this is when \(\xi G\) is <i>global</i> average pooling, in which case it would be equivalent to SE-Var-1 from the ECA paper. By convention, \(e\) is set to \(0\) to signify global pooling.
<div id="equation_6.1"></div>
$$
Z := \xi G(X) := \textrm{AvgPool}(X; \textrm{kernel size}=2e-1, \textrm{stride}=e)
$$
<center><sub>Equation 6.1. Padding is also applied to ensure the spatial dimensions are downsampled exactly by a factor of \(e\).
</sub></center><br>
<div id="equation_6.2"></div>
$$
A := \sigma(\textrm{Interpolate}(Z))
$$
<center><sub>Equation 6.2. The spatial target size of the interpolation is \(h \times w\), the same spatial dimension as the original input.
</sub></center><br>
<div id="equation_6.3"></div>
$$
\tilde X := X \odot A
$$
<center><sub>Equation 6.3.
</sub></center><br>
This parameter-free rudimentary module, implemented in <a href="#snippet_6.1">snippet 6.1</a>, is called GE-θ- (θ- symbolizes the lack of parameters) and consistently improves the performance of ResNet-50 (<a href="#table_6.1">table 6.1</a>). Importantly, the larger the extent, the better the results, a trend that is maintained throughout the rest of this section and indicates the significance of long-range interactions.<br><br>
<div id="snippet_6.1"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	AdaptiveAvgPool2d,
	AvgPool2d,
	Module,
	Sigmoid,
	)
from torch.nn.functional import interpolate


class GENoParams(Module):
	"""
	Gather-excite with no parameters
	"""
	def __init__(
		self,
		extent: int,
		) -> None:
		"""
		Sets up the modules

		Args:
			extent (int): Extent. 0 for a global 
			extent
		"""
		super().__init__()

		if extent_ratio == 0:
			self.gather = AdaptiveAvgPool2d(
				output_size=1,
				)

		else:
			kernel_size = 2*extent - 1
			padding = kernel_size//2
			self.gather = AvgPool2d(
				kernel_size=kernel_size,
				stride=extent,
				padding=padding,
				count_include_pad=False,
				)

		self.sigmoid = Sigmoid()
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module
		
		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		batch_size, in_dim, height, width = input.shape

		gathered = self.gather(input)

		attention = interpolate(
			input=gathered,
			size=(height, width),
			mode='nearest',
			)
		attention = self.sigmoid(attention)

		output = attention*input
		return output
{{< / highlight >}}
<center><sub>Snippet 6.1.</sub></center><br>
<center>
<div id="table_6.1"></div>

| **Extent** | **Top-1 accuracy** |
|------------------|--------------------|
| Original         | 76.71%             |
| 2            | 76.89%             |
| 4            | 77.13%             |
| 8            | 77.60%             |
| Global            | 77.86%             |

</center>
<center><sub>
Table 6.1. Accuracy of GE-θ- with various extents, as well as a plain ResNet-50, on ImageNet.
</sub></center><br>
Naturally, parameterizing GE-θ- should further help, so the authors decide to supplant average pooling in \(\xi G\) with convolutions to get GE-θ. Specifically, \(3\) x \(3\) depthwise convolutions with strides of \(2\) are used to downsample the input, where the number of convolutions is \(\log_2(e)\) to ensure that the input is downsampled by a factor of \(e\). For a global extent, a single depthwise convolution with a kernel size of \(h \times w\) is used. Batch normalization is also appended after each convolution, and for non-global extents, the convolutions are interleaved with ReLU. GE-θ is implemented in <a href="#snippet_6.2">snippet 6.2</a> and outperforms GE-θ- (<a href="#table_6.2">table 6.2</a>).<br><br>
<div id="snippet_6.2"></div>
{{< highlight python "linenos=true">}}
from math import log2
from typing import (
	Optional,
	Tuple,
	Union,
	)

from torch import Tensor                                                          
from torch.nn import (
	BatchNorm2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	Sigmoid,
	)
from torch.nn.functional import interpolate


class GEParams(Module):
	"""
	Gather-excite with parameters
	"""
	def __init__(
		self,
		in_dim: int,
		extent: int,
		spatial_dim: Optional[Union[Tuple[int, int], int]] = None,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			extent (int): Extent. 0 for a global
			extent
			spatial_dim (Optional[Union[Tuple[int, int], int]]):
			Spatial dimension of the input, required for a global 
			extent.
			Default is None
		"""
		super().__init__()

		if extent_ratio == 0:
			self.gather = Sequential(
				Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=spatial_dim,
					groups=in_dim,
					bias=False,
					),
				BatchNorm2d(
					num_features=in_dim,
					),
				)

		else:
			n_layers = int(log2(extent))
			layers = n_layers * [
				Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=3,
					stride=2,
					padding=1,
					groups=in_dim,
					bias=False,
					),
				BatchNorm2d(
					num_features=in_dim,
					),
				ReLU(),
				]
			layers = layers[:-1]
			self.gather = Sequential(*layers)

		self.sigmoid = Sigmoid()
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		batch_size, in_dim, height, width = input.shape

		gathered = self.gather(input)

		attention = interpolate(
			input=gathered,
			size=(height, width),
			mode='nearest',
			)
		attention = self.sigmoid(attention)

		output = attention*input
		return output
{{< / highlight >}}
<center><sub>Snippet 6.2. ReLU should not be the last layer of \(\xi G\), and live 75 removes it.</sub></center><br>
<center>
<div id="table_6.2"></div>

| **Extent** | **GE-θ-** | **GE-θ** |
|------------------|--------------------|--------------------|
| 2            | 76.89%             | 77.29%             |
| 4            | 77.13%             | 77.81%             |
| 8            | 77.60%             | 77.87%             |
| Global            | 77.86%             | 78.00%             |

</center>
<center><sub>
Table 6.2. Top-1 accuracy of GE-θ- and GE-θ of various extents on ImageNet with ResNet-50.
</sub></center><br>
Inspired by the potency of a parameterized gather unit, \(\xi E\) is also parameterized. Particularly, it is superseded by the excitation module from SE, with  \(W_{1} \isin \reals^{\frac {c} {16} \times c \times 1 \times 1}\) and \(W_{2} \isin \reals^{c \times \frac {c} {16} \times 1 \times 1}\) as its parameters, followed by interpolation (<a href="#equation_6.4">equation 6.4</a>). Since the spatial shape of the input is not necessarily \(1\) x \(1\), \(W_1\) and \(W_2\) cannot always be linear layers and are instead \(1\) x \(1\) convolutions. 
<div id="equation_6.4"></div>
$$
\xi E(Z; \bold W_1, \bold W_2) :=  \sigma(\textrm{Interpolate}((W_{2} * \delta (W_{1} * Z))))
$$
<center><sub>Equation 6.4. Whether sigmoid or interpolation is applied first is unimportant.
</sub></center><br>
GE-θ+ is the name of this final iteration of GE, and it combines a parameterized \(\xi G\) with a parameterized \(\xi E\) (<a href="#snippet_6.3">snippet 6.3</a>). With ResNet-50 and ResNet-101, GE-θ+ surpasses GE-θ-, GE-θ, and SE (<a href="#table_6.3">table 6.3</a>).<br><br>
<div id="snippet_6.3"></div>
{{< highlight python "linenos=true">}}
from math import log2
from typing import (
	Optional,
	Tuple,
	Union,
	)

from torch import Tensor                                                          
from torch.nn import (
	BatchNorm2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	Sigmoid,
	)
from torch.nn.functional import interpolate


class GEParamsPlus(Module):
	"""
	Gather-excite with parameters, including for the excite unit
	"""
	def __init__(
		self,
		in_dim: int,
		extent: int,
		spatial_dim: Optional[Union[Tuple[int, int], int]] = None,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			extent (int): Extent. 0 for a global
			extent
			spatial_dim (Optional[Union[Tuple[int, int], int]]):
			Spatial dimension of the input, required for a global 
			extent.
			Default is None
		"""
		super().__init__()

		if extent == 0:
			self.gather = Sequential(
				Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=spatial_dim,
					groups=in_dim,
					bias=False,
					),
				BatchNorm2d(
					num_features=in_dim,
					),
				)

		else:
			n_layers = int(log2(extent))
			layers = n_layers * [
				Conv2d(
					in_channels=in_dim,
					out_channels=in_dim,
					kernel_size=3,
					stride=2,
					padding=1,
					groups=in_dim,
					bias=False,
					),
				BatchNorm2d(
					num_features=in_dim,
					),
				ReLU(),
				]
			layers = layers[:-1]
			self.gather = Sequential(*layers)
		
		bottleneck_dim = in_dim//16
		self.mlp = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=bottleneck_dim,
				kernel_size=1,
				),
			ReLU(),
			Conv2d(
				in_channels=bottleneck_dim,
				out_channels=in_dim,
				kernel_size=1,
				),
			)
		self.sigmoid = Sigmoid()
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		batch_size, in_dim, height, width = input.shape

		gathered = self.gather(input)

		attention = self.mlp(gathered)
		attention = interpolate(
			input=attention,
			size=(height, width),
			mode='nearest',
			)
		attention = self.sigmoid(attention)

		output = attention*input
		return output
{{< / highlight >}}
<center><sub>Snippet 6.3. ReLU should not be the last layer of \(\xi G\), and live 74 removes it.</sub></center><br>

<center>
<div id="table_6.3"></div>

| **Architecture** | **Plain** | **SE** | **GE-θ-** | **GE-θ** | **GE-θ+** |
|------------------|--------------------|--------------------|------------------|------------------|------------------|
| ResNet-50            | 76.70%             | 77.88%             | 77.86%             | 78.00%             | 78.12%             |
| ResNet-101            | 77.80%             | 79.06%             | 78.53%             | 78.54%             | 79.26%             |
| ShuffleNet            | 67.40%             | 68.76%             | N.A.             | 68.20%             | 69.88%             |

</center>
<center><sub>
Table 6.3. Top-1 accuracy of GE modules (with global extents) and SE with a few backbones on ImageNet. The accuracy of GE-θ- with ShuffleNet was not reported.
</sub></center>

<div id="selective-kernel"></div>

---

# 7. Selective Kernel
Convolutional neural networks, since their nascent days, have been inspired by biological neurons in the primary visual cortex (V1) and aim to mimic the vision processing of animals. Kunihiko Fukushima, for example, developed the neocognitron in 1980<sup><a href="#fuk">[5]</a></sup>, a primitive archetype of CNNs that was based on the research of David Hubel and Torsten Wiesel on simple and complex cells ubiquitous in the V1<sup><a href="#spatial">[8]</a></sup>. Simple cells detect bar-like shapes of particular orientations (e.g., edges), whereas complex cells are composed using the integration of several simple cells, thereby exhibiting properties such as spatial invariance and recognition of more complex patterns like polygons.

Similarly, the Inception family of architectures<sup><a href="#inception">[19]</a><a href="#inceptionv2">[20]</a><a href="#inceptionv4">[18]</a></sup> collects multi-scale information from images using convolutions of various kernel sizes to be faithful to the fact that the receptive field sizes of nearby neurons in the visual cortex might differ (<a href="#figure_7.1">figure 7.1</a>). Notwithstanding, it failed to address a key ingredient of biological receptive fields, namely, how the receptive field size is adjusted according to the stimulus and is not static<sup><a href="#ncrf">[17]</a></sup>.<br><br>
<div id="figure_7.1"></div>
<img src="figure_7.1.png">
<center><sub>Figure 7.1. A typical Inception block with diverse kernel sizes. Image from figure 2 of <a href="#inception">[19]</a>.
</sub></center><br>

The selective kernel (SK)<sup><a href="#sk">[12]</a></sup> module is designed to bridge this gap and is intended as a drop-in replacement for traditional convolutions with fixed kernel sizes. It contains multiple branches of different kernel sizes and uses an attention mechanism to adaptively aggregate their outputs so more weight is assigned to branches with kernel sizes relevant for the current input.

SK's first constituent is _split_, where the goal is to transform the data via multiple branches, each with a convolutional layer of a different kernel size (<a href="#figure_7.2">figure 7.2</a>). For input \\(X \isin \reals^{c \times h \times w}\\) and two as the number of branches, there would be two convolutional layers (kernel sizes \\(3\\) x \\(3\\) and \\(5\\) x \\(5\\)) with weight tensors \\(W_1 \isin \reals^{c \times c \times 3 \times 3}\\) and \\(W_2 \isin \reals^{c \times c \times 5 \times 5}\\). They convolve over \\(X\\), followed by batch normalization and ReLU, to obtain \\(U_1 \isin \reals^{c \times h \times w}\\) and \\(U_2 \isin \reals^{c \times h \times w}\\). In general, the output of branch \\(m'\\), with \\(m\\) as the number of branches, is calculated by convolving over \\(X\\) with \\(W_{m'}\\) and applying batch normalization and ReLU (<a href="#equation_7.1">equation 7.1</a>).<br><br>
<div id="figure_7.2"></div>
<img src="figure_7.2.jpeg">
<center><sub>Figure 7.2. The first step of SK, split (somewhat of a misnomer, as the input is not divided into several non-overlapping parts). \(\tilde U\) and \(\hat U\) correspond to \(U_1\) and \(U_2\). Image from figure 1 of <a href="#sk">[12]</a>.
</sub></center><br>

<div id="equation_7.1"></div>
$$
U_i := \delta (\textrm{BN}(W_{m'} * X))
$$
<center><sub>Equation 7.1.
</sub></center><br>
The implementation of this component of SK slightly differs from the diagram and description above, in that rather than enlarging the kernel size, the dilation value is increased, and the kernel size is kept at \(3\) x \(3\) for efficiency. For instance, in lieu of a \(5\) x \(5\) convolution, a \(3\) x \(3\) convolution with a dilation value of \(2\) is used. Also, since SK is branded as a replacement for regular convolutions, it must accept other arguments, namely, the stride and group size, although there are other ones, e.g., padding mode, that are not supported by this implementation (<a href="#snippet_7.1">snippet 7.1</a>).<br><br>
<div id="snippet_7.1"></div>
{{< highlight python "linenos=true">}}
from torch import (
	Tensor,                                                          
	stack,
	)
from torch.nn import (
	BatchNorm2d,
	Conv2d,
	Module,
	ModuleList,
	ReLU,
	)


class Branch(Module):
	"""
	A branch for SK's split module
	"""
	def __init__(
		self,
		in_dim: int,
		kernel_size: int = 3,
		stride: int = 1,
		groups: int = 32,
		dilation: int = 1,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			kernel_size (int): Kernel size.
			Default is 3
			stride (int): Stride.
			Default is 1.
			groups (int): Number of groups.
			Default is 32
			dilation (int): Dilation.
			Default is 1
		"""
		super().__init__()

		padding = kernel_size//2 + (dilation-1)
		self.conv = Conv2d(
			in_channels=in_dim,
			out_channels=in_dim,
			kernel_size=kernel_size,
			stride=stride,
			padding=padding,
			dilation=dilation,
			groups=groups,
			bias=False,
			)
		self.bn = BatchNorm2d(
			num_features=in_dim,
			)
		self.relu = ReLU()
	
	def forward(
		self,
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		output = self.conv(input)
		output = self.bn(output)
		output = self.relu(output)
		return output


class Split(Module):
	"""
	SK's split module
	"""
	def __init__(
		self,
		in_dim: int,
		n_branches: int = 2,
		stride: int = 1,
		groups: int = 32,
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			n_branches (int): Number of branches.
			Default is 2
			stride (int): Stride for each branch.
			Default is 1
			groups (int): Number of groups for each branch.
			Default is 32
		"""
		super().__init__()

		branches = []
		for dilation in range(1, n_branches+1):
			branch = Branch(
				in_dim=in_dim,
				out_dim=in_dim,
				kernel_size=3,
				stride=stride,
				groups=groups,
				dilation=dilation,
				)
			branches.append(branch)
		self.branches = ModuleList(
			modules=branches,
			)
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		outputs = []
		for branch in self.branches:
			output = branch(input)
			outputs.append(output)

		output = stack(
			tensors=outputs, 
			dim=1,
			)
		return output
{{< / highlight >}}
<center><sub>Snippet 7.1. The input and output dimensions are assumed to be equal. Padding is not an argument, and it is set so that the spatial dimensions remain unchanged. The output has a new axis of size \(2\) at dimension \(1\), representing the two branches.</sub></center><br>
Next, a vector representation of \(U_1\) and \(U_2\) needs to be computed, which shall later be used to calculate attention values. That is done so by first summing them to get \(U \isin \reals^{c \times h \times w}\) (<a href="#equation_7.2">equation 7.2</a>) and performing global average pooling over \(U\) to get \(\bold s \isin \reals^{c}\) (<a href="#equation_7.3">equation 7.3</a>). Then, a feature vector \(\bold z \isin \reals^{\frac {c} {r}}\) (dimension reduced by a factor of \(r\) for efficiency) is extracted through transforming \(\bold s\) via a linear layer with weight matrix \(\bold W \isin \reals^{\frac {c} {r} \times c}\), followed by batch normalization and ReLU (<a href="#equation_7.4">equation 7.4</a>).<br><br>
<div id="equation_7.2"></div>
$$
U = \sum_{m'=1} ^ {m} U_{m'}
$$
<center><sub>Equation 7.2.</sub></center><br>
<div id="equation_7.3"></div>
$$
\bold s_{c'} := \frac {1} {hw} \sum_{h' = 1} ^{h} \sum_{w' = 1} ^{w} U_{c',h',w'}
$$
<center><sub>Equation 7.3. \(\forall c' \isin {\{1, 2, ..., c\}}\).
</sub></center><br>
<div id="equation_7.4"></div>
$$
\bold z := \delta(\textrm{BN}(\bold W \bold s))
$$
<center><sub>Equation 7.4.
</sub></center><br>
This step is known as <i>fuse</i> and is depicted in <a href="#figure_7.3">figure 7.3</a> and implemented in <a href="#snippet_7.2">snippet 7.2</a>.<br><br>
<div id="figure_7.3"></div>
<img src="figure_7.3.jpeg">
<center><sub>Figure 7.3. Split and fuse from SK. Ibid.
</sub></center><br>
<div id="snippet_7.2"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	AdaptiveAvgPool2d,
	BatchNorm2d,
	Conv2d,
	Module,
	ReLU,
	Sequential,
	)


class Fuse(Module):
	"""
	SK's fuse module
	"""
	def __init__(
		self,
		in_dim: int,
		reduction_factor: int = 16
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			reduction_factor (int): Reduction factor for the 
			fully-connected layer.
			Default is 16
		"""
		super().__init__()
		
		reduced_dim = in_dim//reduction_factor
		self.avg_pool = AdaptiveAvgPool2d(
			output_size=1,
			)
		self.fc = Sequential(
			Conv2d(
				in_channels=in_dim,
				out_channels=reduced_dim,
				kernel_size=1,
				bias=False,
				),
			BatchNorm2d(
				num_features=reduced_dim,
				),
			ReLU(),
			)
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input (the output of the split module)
		
		Returns (Tensor): Result of the module
		"""
		summed = input.sum(1)
		avg_pooled = self.avg_pool(summed)
		output = self.fc(avg_pooled)
		return output
{{< / highlight >}}
<center><sub>Snippet 7.2.</sub></center><br>
Lastly, SK's <i>select</i> module calculates attention vectors for every branch using \(\bold z\). Particularly, for every branch with index \(m'\), there is a weight matrix \(\bold W_{m'} \isin \reals^{c \times \frac {c} {r}}\) that transforms  \(\bold z\) into an unnormalized attention vector \(\bold a'_{m'} \isin \reals^{c}\) (<a href="#equation_7.5">equation 7.5</a>). Subsequently, the softmax function is applied along the branch axis to get normalized attention vectors for every branch, namely, \(\bold a_{m'} \isin \reals^{c}\) (<a href="#equation_7.6">equation 7.6</a>). The output of each branch is multiplied by its attention vector (i.e., each channel is multiplied by its associated attention value), and the results across all branches are summed to get the final output \(\tilde X \isin \reals^{c \times h \times w}\) (<a href="#equation_7.7">equation 7.7</a>).
<div id="equation_7.5"></div>
$$
\bold a'_{m'} := \bold W_{m'} \bold z
$$
<center><sub>Equation 7.5.
</sub></center><br>
<div id="equation_7.6"></div>
$$
\bold a_{m'} := \frac {e^{a_{m'}}} {\sum_{m''=1} ^ m e^{\bold a_{m''}}}
$$
<center><sub>Equation 7.6. The exponential of each vector is taken element wise.
</sub></center><br>
<div id="equation_7.7"></div>
$$
\tilde X_{c',h',w'} := \sum_{m'=1} ^ {m} \bold a_{m'_{c'}}U_{m'_{c',h',w'}}
$$
<center><sub>Equation 7.7. \(\forall c' \isin {\{1, 2, ..., c\}}, \forall h' \isin {\{1, 2, ..., h\}}, \forall w' \isin {\{1, 2, ..., w\}}\).
</sub></center><br>
The select unit and the complete version of SK are depicted in <a href="#figure_7.4">figure 7.4</a> and implemented in <a href="#snippet_7.3">snippet 7.3</a>. Some readers may have observed how similar squeeze-and-excitation is to the fuse and select components of SK; in fact, SK with a single branch is simply SE.<br><br>
<div id="figure_7.4"></div>
<img src="figure_7.4.jpeg">
<center><sub>Figure 7.4. \(\bold a\) and \(\bold b\) are two normalized attention vectors, i.e., \(\bold a_1\) and \(\bold a_2\). Ibid.
</sub></center><br>
<div id="snippet_7.3"></div>
{{< highlight python "linenos=true">}}
from torch import Tensor                                                          
from torch.nn import (
	Conv2d,
	Module,
	Softmax,
	)


class Select(Module):
	"""
	SK's select module
	"""
	def __init__(
		self,
		in_dim: int,
		n_branches: int = 2,
		reduction_factor: int = 16
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels (channels
			of the original input)
			n_branches (int): Number of branches.
			Default is 2
			reduction_factor (int): Reduction factor for the 
			fully-connected layer of the fuse module.
			Default is 16
		"""
		super().__init__()

		reduced_dim = in_dim//reduction_factor
		attention_dim = n_branches*in_dim
		self.fc = Conv2d(
			in_channels=reduced_dim,
			out_channels=attention_dim,
			kernel_size=1,
			bias=False,
			)
		self.softmax = Softmax(
			dim=1,
			)
	
	def forward(
		self, 
		input: Tensor,
		z: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input (the output of the split module)
			z (Tensor): z (the output of the fuse module)
		
		Returns (Tensor): Result of the module
		"""
		batch_size, n_branches, in_dim, height, width = input.shape

		attention = self.fc(z)
		attention = attention.reshape(batch_size, n_branches, in_dim, 1, 1)
		attention = self.softmax(attention)

		output = attention*input
		output = output.sum(1)
		return output


class SK(Module):
	"""
	Selective kernel module
	"""
	def __init__(
		self,
		in_dim: int,
		n_branches: int = 2,
		stride: int = 1,
		groups: int = 32,
		reduction_factor: int = 16
		) -> None:
		"""
		Sets up the modules

		Args:
			in_dim (int): Number of input channels
			n_branches (int): Number of branches.
			Default is 2
			stride (int): Stride for each branch.
			Default is 1
			groups (int): Number of groups for each branch.
			Default is 32
			reduction_factor (int): Reduction factor for the 
			fully-connected layer for the fuse module.
			Default is 16
		"""
		super().__init__()

		self.split = Split(
			in_dim=in_dim,
			n_branches=n_branches,
			stride=stride,
			groups=groups,
			)
		self.fuse = Fuse(
			in_dim=in_dim,
			reduction_factor=reduction_factor,
			)
		self.select = Select(
			in_dim=in_dim,
			n_branches=n_branches,
			reduction_factor=reduction_factor,
			)
	
	def forward(
		self, 
		input: Tensor,
		) -> Tensor:
		"""
		Runs the input through the module

		Args:
			input (Tensor): Input
		
		Returns (Tensor): Result of the module
		"""
		branches = self.split(input)
		z = self.fuse(branches)
		output = self.select(branches, z)
		return output
{{< / highlight >}}
<center><sub>Snippet 7.3. In the select unit, <code>self.fc</code> is equivalent to \(m\) distinct linear layers. Line 62 reshapes the output of <code>self.fc</code> so that there is a separate axis for the branches.</sub></center><br>
For testing SK, \(3\) x \(3\) convolutions in regular CNNs are replaced with SK modules containing two branches, and the other hyperparameters, e.g., the width, group size, and stride, are the same. Experiments with ResNeXt and ShuffleNetV2 corroborate the power of SK and adaptive receptive field sizes (<a href="#table_7.1">table 7.1</a>).<br><br>
<center>
<div id="table_7.1"></div>

| **Architecture** | **Plain** | **With SE** | **With SK** |
|------------------|-----------|-------------|-------------|
| ResNeXt-50       | 77.77%    | 78.88%      | 79.21%      |
| ResNeXt-101      | 78.89%    | 79.42%      | 79.81%      |
| ShuffleNetV2     | 69.43%    | 70.53%      | 71.64%      |

</center>
<center><sub>
Table 7.1. Top-1 accuracy of plain networks, plus their SE and SK counterparts, on ImageNet. The authors reported the accuracies of CBAM- and BAM-based networks as well, which were subpar compared to SE. That can be attributed to the different training regimen used by the SK paper.
</sub></center>

---

# References
<div id="general_survey"></div>

[1] Gianni Brauwers, Flavius Frasincar. [A General Survey on Attention Mechanisms in Deep Learning](https://arxiv.org/abs/2203.14263).
In _TKDE_, 2021.

<div id="charac"></div>

[2] Andrew Brock, Soham De, Samuel Smith. [Characterizing signal propagation to close the performance gap in
  unnormalized ResNets](https://arxiv.org/abs/2101.08692). In _ICLR_, 2021.

<div id="high_perf"></div>

[3] Andrew Brock, Soham De, Samuel Smith, Karen Simonyan. [High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/abs/2102.06171). In _ICML_, 2021.

<div id="imagenet"></div>

[4] Jia Deng, Wei Dong, Richard Socher, Li-Jia Li, Kai Li, Li Fei-Fei. [ImageNet: A large-scale hierarchical image database](https://image-net.org/static_files/papers/imagenet_cvpr09.pdf). In _CVPR_, 2009.

<div id="fuk"></div>

[5] Kunihiko Fukushima. [Neocognitron: A Self-organizing Neural Network Model for a Mechanism of Pattern Recognition Unaffected by Shift in Position](https://www.rctn.org/bruno/public/papers/Fukushima1980.pdf). In _Biological Cybernetics_, 1980.

<div id="cv_survey"></div>

[6] Meng-Hao Guo, Tian-Xing Xu, Jiang-Jiang Liu, Zheng-Ning Liu, Peng-Tao Jiang, Tai-Jiang Mu, Song-Hai Zhang, Ralph Martin, Ming-Ming Cheng, Shi-Min Hu.
[Attention Mechanisms in Computer Vision: A Survey](https://arxiv.org/abs/2111.07624). In _Computational Visual Media_, 2022.

<div id="ge"></div>

[7] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Andrea Vedaldi. [Gather-Excite: Exploiting Feature Context in Convolutional Neural Networks](https://arxiv.org/abs/1810.12348). In _NeurIPS_, 2018.

<div id="spatial"></div>

[8] David Hubel, Torsten Wiesel. [Receptive fields, binocular interaction and functional architecture in the cat's visual cortex](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC1359523/). In _The Journal of Physiology_, 1962.

<div id="se"></div>

[9] Jie Hu, Li Shen, Samuel Albanie, Gang Sun, Enhua Wu. [Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507).
In _CVPR_, 2018.

<div id="kan"></div>

[10] Nancy Kanwisher. [The Human Brain course at MIT, Lecture 24](https://www.youtube.com/watch?v=B4a0WdGp52g). 2019.

<div id="ese"></div>

[11] Youngwan Lee, Jongyoul Park. [CenterMask: Real-Time Anchor-Free Instance Segmentation](https://arxiv.org/abs/1911.06667).
In _CVPR_, 2020.

<div id="sk"></div>

[12] Xiang Li, Wenhai Wang, Xiaolin Hu, Jian Yang. [Selective Kernel Networks](https://arxiv.org/abs/1903.06586). In _CVPR_, 2019.

<div id="eff_rec"></div>

[13] Wenjie Luo, Yujia Li, Raquel Urtasun, Richard Zemel. [Understanding the Effective Receptive Field in Deep Convolutional Neural Networks](https://arxiv.org/abs/1701.04128). In _NeurIPS_, 2016.

<div id="ng"></div>

[14] Andrew Ng. [The Deep Learning Specialization course, Course 5](https://www.youtube.com/watch?v=SysgYptB198). 2018.

<div id="bam"></div>

[15] Jongchan Park, Sanghyun Woo, Joon-Young Lee, In Kweon. [BAM: Bottleneck Attention Module](https://arxiv.org/abs/1807.06514). In _BMCV_, 2018.

<div id="reg"></div>

[16] Ilija Radosavovic, Raj Kosaraju, Ross Girshick, Kaiming He, Piotr Dollár. [Designing Network Design Spaces](https://arxiv.org/abs/2003.13678).
In _CVPR_, 2020.

<div id="ncrf"></div>

[17] Lothar Spillmann, Birgitta Dresp-Langley, Chia-Huei Tseng. [Beyond the classical receptive field: The effect of contextual stimuli](https://pubmed.ncbi.nlm.nih.gov/26200888/). In _Journal of Vision_, 2015.

<div id="inceptionv4"></div>

[18] Christian Szegedy, Sergey Ioffe, Vincent Vanhoucke, Alex Alemi. [Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning](https://arxiv.org/abs/1602.07261). In _CVPR_, 2017.

<div id="inception"></div>

[19] Christian Szegedy, Wei Liu, Yangqing Jia, Pierre Sermanet, Scott Reed, Dragomir Anguelov, Dumitru Erhan, Vincent Vanhoucke, Andrew Rabinovich. [Going Deeper with Convolutions](https://arxiv.org/abs/1409.4842). In _CVPR_, 2015.

<div id="inceptionv2"></div>

[20] Christian Szegedy, Vincent Vanhoucke, Sergey Ioffe, Jonathon Shlens, Zbigniew Wojna. [Rethinking the Inception Architecture for Computer Vision](https://arxiv.org/abs/1512.00567). In _CVPR_, 2016.

<div id="eff"></div>

[21] Mingxing Tan, Quoc Le. [EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks](https://arxiv.org/abs/1905.11946).
In _ICML_, 2019.

<div id="eff2"></div>

[22] Mingxing Tan, Quoc Le. [EfficientNetV2: Smaller Models and Faster Training](https://arxiv.org/abs/2104.00298).
In _ICML_, 2021.

<div id="eca"></div>

[23] Qilong Wang, Banggu Wu, Pengfei Zhu, Peihua Li, Wangmeng Zuo, Qinghua Hu. [ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks](https://arxiv.org/abs/1910.03151). In _CVPR_, 2020.

<div id="cbam"></div>

[24] Sanghyun Woo, Jongchan Park, Joon-Young Lee, In Kweon. [CBAM: Convolutional Block Attention Module](https://arxiv.org/abs/1807.06521).
In _ECCV_, 2018.

---
