---
{"dg-publish":true,"permalink":"/论文阅读笔记/DNN拆分调研/Auto-Split/"}
---

# Auto-Split: A General Framework of Collaborative Edge-Cloud AI
Doi：[Fetching Title#7qr4](https://doi.org/10.1145/3447548.3467078)
年份：2021
## Summary
本文描述了一个名为Auto-Split的框架，该框架是华为云的原型，它可以在边缘云协作环境中高效部署和推断大量耗费资源的机器学习模型。本文讨论了设计行业产品所面临的挑战，这些产品旨在支持深度模型部署和高效地进行模型推断，同时保持高模型精度和低端到端延迟。Auto-Split 是一项专利技术，已在选定的应用程序上进行了验证，并作为端到端云边缘协作智能部署的自动化管道服务可供公众使用。本文还讨论了 Auto-Split 背后的技术和工程实践，以及如何将其用于拆分物体检测模型。Auto-Split 的目标函数是通过将所有延迟部分相加来定义的，本文为 Auto-Split 提供了非线性整数优化问题公式。
## Research Objective
1. 华为云的边缘云协同原型Auto - Split背后的技术和工程实践，将训练好的DNN和样本分析数据作为输入，并应用多个环境约束来优化人工智能应用的端到端延迟。
	1. 边缘设备约束：片上和片外存储器、NN加速器引擎的数量和大小、设备带宽和位宽支持。
	2. 网络约束：基于网络类型(例如BLE、3G、5G、WiFi等)的上行带宽。
	3. 云设备约束：内存、带宽和计算能力。
	4. 用户提供的精度要求。
2. 为了进一步降低网络传输成本和模型大小，将edge-cloud splitting和posttraining quantization结合。
## Background and Challenges
大规模的深度模型通常驻留在云服务器中，拥有强大的计算能力。同时，数据往往分布在云的边缘，即各种网络的边缘。 然而，海量数据与大型深度学习模型之间的鸿沟成为人工智能应用面临的艰巨挑战。一方面，大型深度学习模型无法加载到这些低功耗设备中。另一方面将高分辨率、大量的输入数据全部传输到云服务器会产生较高的传输成本，导致较高的端到端延迟。而且，当原始数据被传输到云端时，可能会施加额外的隐私风险。 

现有的行业解决方案有两种：
- Edge - Only：将模型压缩后放置于边缘，这种方法容易造成严重的精度损失。
- 分布式方法：
	- Cloud - Only方法在云上进行推理。这可能会带来高昂的数据传输成本，特别是在高精度应用的高分辨率输入数据的情况下。
	- Edge-Cloud推理方法将一个任务划分为多个子任务，在边缘部署一些子任务，并将这些任务的输出传输到其他任务所在的云端。
	- multi-exit solution在边缘部署轻量级模型，处理较简单的案例，并将较难的案例传输到云服务器中较大的模型。可能存在精度低、延迟不确定等问题。

新的方法是edge-cloud collaborative approach云边协同方法，即DNN拆分。对此，主要的挑战是开发一个通用的框架来划分边缘和云之间的深度神经网络，以最小化端到端延迟，并保持边缘中的模型尺寸较小。此外，该框架应具有通用性和灵活性，以便能够应用于许多不同的任务和模型。
## Method
![Pasted image 20230728143339.png](/img/user/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/DNN%E6%8B%86%E5%88%86%E8%B0%83%E7%A0%94/_resource/Pasted%20image%2020230728143339.png)
### PROBLEM FORMULATION

#### Objective function
$$\mathcal{L}\left(\mathbf{b}^{w}, \mathbf{b}^{a}, n\right)=\sum_{i=1}^{n} \mathcal{L}_{i}^{\text {edge }}+\mathcal{L}_{n}^{\text {tr }}+\sum_{i=n+1}^{N} \mathcal{L}_{i}^{\text {cloud }}$$
其中n代表分裂的层数，$b^w$和$b^a$分别代表权重向量和激活向量的位宽，$\mathcal{L}_{i}^{\text {edge }}$和$\mathcal{L}_{i}^{\text {cloud }}$分别代表边缘和云端的运行延迟，$\mathcal{L}_{n}^{\text {tr }}$代表传输延迟。优化目标则是使得上式最小。

等同于最小化
$$\begin{aligned}
&\mathcal{L}(\mathbf{b}^{\mathbf{w}},\mathbf{b}^{a},n)-\mathcal{L}(\mathbf{b}^{\mathbf{w}},\mathbf{b}^{a},0) \\
&=\left(\sum_{i=1}^{n}\mathcal{L}_{i}^{edge}+\mathcal{L}_{n}^{tr}+\sum_{i=n+1}^{N}\mathcal{L}_{i}^{cloud}\right)-\left(\mathcal{L}_{0}^{tr}+\sum_{i=1}^{N}\mathcal{L}_{i}^{cloud}\right) \\
&\left(\sum_{i=1}^{n}\mathcal{L}_{i}^{edge}+\mathcal{L}_{n}^{tr}\right)-\left(\mathcal{L}_{0}^{tr}+\sum_{i=1}^{n}\mathcal{L}_{i}^{cloud}\right)=\sum_{i=1}^{n}\mathcal{L}_{i}^{edge}+\mathcal{L}_{n}^{tr}-\sum_{i=1}^{n}\mathcal{L}_{i}^{cloud}
\end{aligned}$$
#### Memory constraint
权重向量内存开销：$\mathcal{M}^{w}=\sum_{i=1}^{n}(s_{i}^{w}\times b_{i}^{w})$
激活向量内存开销：$\mathcal{M}^{a}=\max_{i=1,...,n}(s_{i}^{a}\times{b}_{i}^{a})$
$${\cal M}^{w}+{\cal M}^{a}\leq M$$
#### Error constraint
使用方误差函数进行衡量
权重误差：${\cal D}_{i}^{w}=MSE(w_{i}(16),w_{i}(\mathbf{b}_{i}^{w}))$
激活误差：${\cal D}_{i}^{a}=MSE(a_{i}(16),a_{i}(\mathbf{b}_{i}^{a}))$
$$\sum_{i=1}^{n}(\mathcal{D}_{i}^{w}+\mathcal{D}_{i}^{a})\leq E.$$
#### Formulation
$$\begin{gathered}\min_{\mathbf{b}^w,\mathbf{b}^a\in\mathbb{B}^n,n}\left(\sum_{i=1}^n\mathcal{L}_i^{edge}+\mathcal{L}_n^{tr}-\sum_{i=1}^n\mathcal{L}_i^{cloud}\right)\\\mathrm{s.t.}\mathcal{M}^w+\mathcal{M}^a\leq M,\\\sum_{i=1}^n(\mathcal{D}_i^w+\mathcal{D}_i^a)\leq E,\end{gathered}$$

### SOLUTION
对于上述NP -hard问题，提出了一种多步搜索方法来寻找满足内存约束的潜在解列表，然后选择一个最小化延迟并且满足误差约束的解。
1. Potential Split Identification
![Pasted image 20230728152128.png](/img/user/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/DNN%E6%8B%86%E5%88%86%E8%B0%83%E7%A0%94/_resource/Pasted%20image%2020230728152128.png)
首先，我们在原始图(a )上进行批量范数折叠和激活融合等图优化，得到优化图(b )。对于DNN，一个潜在的分裂点应该满足给定边缘服务器最低位宽分配的内存限制。然后创建了加权DAG（c），其中节点为层，边的权重为最低传输成本。最后，将加权DAG按照拓扑顺序进行排序，生成新的传输DAG（d）。即：
$$\mathbb{P}=\left\{n\in0,1,...,N|T_n\leq T_0,b_{min}(\sum_{i=1}^ns_i^w+\max_{i=1,...,n}s_i^a)\leq M\right\}.$$
其中$T_0$为将原始数据传输到云端的成本，是一个固定常数。

2. Bit-Width Assignment
![Pasted image 20230728152817.png|250](/img/user/%E8%AE%BA%E6%96%87%E9%98%85%E8%AF%BB%E7%AC%94%E8%AE%B0/DNN%E6%8B%86%E5%88%86%E8%B0%83%E7%A0%94/_resource/Pasted%20image%2020230728152817.png)

## Evaluation
### Setting
我们在基于ARM的SCALE - SIM 的周期精确模拟器上测量边缘和云设备延迟。对于边缘设备，模拟Eyeriss ，对于云设备，模拟Tensor Processing Units ( TPU )。Eyeriss和TPU的硬件配置均取自SCALE - SIM。在仿真中，较低的比特精度(低于8 - bit )并不能加快MAC操作本身的速度，因为现有的硬件都有固定的INT - 8 MAC单元。然而，较低的位精度加速了数据在片外和片上存储器之间的移动，进而导致了整体的加速比。
### Detail
1. Accuracy vs Latency Trade-off：精度与延迟的权衡，说明Auto - Split可以根据用户误差阈值选择若干个解。给出了ResNet - 50和Yolov3 DNNs的拆分可行解。
2. 对比实验：与Neurosurgeon，QDMP，U8 (uniform 8-bit quantization/Edge)和CLOUD16 (Cloud-Only)的延迟与精度进行对比。其中前两个为现有的拆分方法。
3. 消融实验：与QDMP+量化进行对比，因为QDMP是以浮点精度运行的，这里对QDMP生成解的边缘部分进行量化的，结果发现也是不足的。 
4. Case Study：实时车牌识别的案例
## Conclusion
本文研究了在边缘和云之间分配DNN推理的可行性，同时在DNN的边缘划分上应用混合精度量化。将该问题建模为一个优化问题，其目标是确定权重和激活的拆分和位宽分配，从而在不牺牲准确性的情况下降低总体延迟。与现有策略相比，该方法具有安全性、确定性和架构灵活性等优势。所提出的方法提供了在精度-延迟权衡中的一系列选项，这些选项可以根据目标应用需求进行选择。 
## Note


