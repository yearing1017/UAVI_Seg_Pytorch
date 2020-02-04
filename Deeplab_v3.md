### 1. 前言

- 本文原文链接：[Rethinking Atrous Convolution for Semantic Image Segmentation](https://arxiv.org/pdf/1706.05587v3.pdf)
- Deeplabv3的核心是**ASPP模块，且已经去除了之前的DenseCRF。**

### 2. Abstract

- 在本文中，我们重新回顾了空洞卷积在语义分割中的应用，这是一种**显式调整滤波器感受野和控制网络特征响应分辨率**的有力工具。
- 为了解决多尺度分割对象的问题，我们设计了**采用级联或并行的方式使用多个不同空洞率的空洞卷积模块**，以捕获多尺度上下文信息。
- 此外，扩充了先前提出的**空洞卷积空间金字塔池化模块，该模块在多尺度上探测卷积特征，可以编码图像级的全局上下文特征，并能进一步提高性能**。
- `Furthermore, we propose to augment our previously proposed Atrous Spatial Pyramid Pooling module, which probes convolutional features at multiple scales, with image-level features encoding global context and further boost performance.`

- 提出的DeepLab V3比我们以前的DeepLab有了很大的改进，没有经过Dense CRF的后处理，并且在Pascal VOC 2012语义图像分割基准上获得了state-of-art的性能。

### 3. Introduction

- 深层卷积神经网络(DCNNs)应用于语义分割的任务，我们考虑了面临的两个挑战：
- 第一个挑战：**连续池化操作或`convolution striding`导致的特征分辨率降低**。这使得DCNN能够学习更抽象的特征表示。然而，这种不变性可能会阻碍密集预测任务，因为不变性也导致了详细空间信息的不确定。**为了克服这个问题，我们提倡使用空洞卷积**。论文里有关空洞卷积的介绍如下：

- `Atrous convolution, also known as dilated convolution, allows us to repurpose ImageNet [72] pretrained networks to extract denser feature maps by removing the downsampling operations from the last few layers and upsampling the corresponding ﬁlter kernels, equivalent to inserting holes (‘trous’ in French) between ﬁlter weights.With atrous convolution, one is able to control the resolution at which feature responses are computed within DCNNs without requiring learning extra parameters.`

- 第二个挑战源自`the existence of objects at multiple scales.`。几种方法已经被提出来处理这个问题，在本文中我们主要考虑了这些工作中的四种类型，如图所示。

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/11-1.jpg)

- 第一种，**将DCNN应用到不同scale的image-input上，`where objects at different scales become prominent at different feature maps.`最后将两个结果融合得到输出。**

- 第二种，**encoder-decoder结构：encoder部分得来多尺度特征，decoder部分恢复空间分辨率。**

- 第三种：在原始模型的顶端叠加额外的模块，以捕捉像素间大范围信息。例如Dense CRF，或者叠加一些其他的卷积层。
- 第四种：**Spatial Pyramid Pooling空间金字塔池化，使用不同采样率和多种视野的卷积核或池化操作，以捕捉多尺度对象**。
- 在本工作中：
  - 我们重新讨论了**在级联模块和空间金字塔池化的框架下应用空洞卷积，这使得能够有效地扩大滤波器的感受野，将多尺度的上下文结合起来**。特别的，我们提出的模块由具有不同采样率的空洞卷积和BN层组成，对于训练十分重要。
  - 我们试验了**级联方式或并行方式部署模块，该模块具体来说就是`Atrous Spatial Pyramid Pooling (ASPP) method`。**
  - 讨论了一个重要问题：大采样率的3×3空洞卷积，图像边界处无法捕获远距离信息，会退化为1×1，我们建议**在ASPP模块中加入图像级特征**。
  - 此外，我们详细介绍了实现的细节，并分享了训练模型的经验，还包括一种简单而有效的引导方法，用于处理稀有和精细注释的对象。
- 最后，提出的模型DeepLab V3改进了我们以前的工作，并在Pascal VOC 2012上获得了85.7%的表现，并且我们没有使用CRF后处理。

### 4. Related Work

- 多个工作已经证明了全局特性或上下文的相互作用有助于语义分割，在本文中，我们讨论了四种利用上下文信息进行语义分割的全卷积网络(FCNs)，见下图：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/11-1.jpg)

- **Image pyramid:**
  - 同样的模型，通常使用共享权重，使用多尺度的输入。小尺寸的输入特征响对应长距离语义，大尺寸输入的相应修正细节。通过拉普拉斯金字塔对输入图像进行变换，将不同尺度的图片输入到DCNN，并将所有比例的特征图合并。有人将多尺度输入按顺序从粗到细依次应用，也有人直接将输入调整成不同的大小，并融合所有大小的特征。这类模型的主要缺点是由于GPU内存，较大/更深的DCNN不方便应用，因此通常在推理阶段应用。
- **Encoder-decoder：**
  - 该模型由两部分组成：**(a)编码器中，特征映射的空间维度逐渐减小，从而更容易捕获较长范围内的信息；(b)解码器中，目标细节和空间维度逐渐恢复。**
  - 例如，**使用反卷积来学习对低分辨率特征响应进行上采样。**SegNet复用编码器中的池化索引，学习额外的卷积层来平滑特征响应；U-net将编码器中的特征层通过跳跃连接添加到相应的解码器激活层中；LRR使用了一个拉普拉斯金字塔重建网络。最近，RefineNet等证明了基于编码-解码结构的有效性。这类模型也在对象检测的领域得到了应用。
- **Context module**：
  - 包含了额外的模块，采用级联的方式，用来编码远距离上下文信息。一种有效的方法是合并Dense CRF到DCNNs中，共同训练DCNN和CRF。
- **Spatial pyramid pooling**：
  - **空间金字塔池化可以在多个范围内捕捉上下文信息**。
  - ParseNet从不同图像等级的特征中获取上下文信息。DeepLabv V2提出了空洞卷积空间金字塔池化(ASPP)，使用不同采样率的并行空洞卷积层才捕获多尺度信息。PSPNet在不同网格尺度上执行空间池化，并在多个语义分割数据集上获得出色的性能。还有其他基于LSTM的方法聚合全局信息。

- 在本工作中，我们主要探讨**空洞卷积作为上下文模块和空间金字塔池化的工具。**
- 我们提出的框架是一般性的，可以适用于任何网络。具体而言，我们取ResNet最后一个block，复制多个级联起来，送入到包含多个平行空洞卷积的ASPP模块中。`Note that our cascaded modules are applied directly on the feature maps instead of belief maps.`

- 我们通过实验发现使用**BN层有利于模块的训练**。为了进一步捕获全局上下文，我们建议像PSPNet一样**在ASPP上融入图像级特征。**

### 5. Method

- 这里主要**回顾如何应用atrous convolution来提取紧凑的特征，以进行语义分割; 然后介绍在串行和并行中采用atrous convolution的模块。**

#### 5.1 Atrous Convolution for Dense Feature Extraction

- 全卷积的DCNNs已经证明了其在语义分割任务 上的有效。
- 经常独读到的为什么使用空洞卷积的原因：`However, the repeated combination of max-pooling and striding at consecutive layers of these networks signiﬁcantly reduces the spatial resolution of the resulting feature maps, typically by a factor of 32 across each direction in recent DCNNs [47, 78, 32]. Deconvolutional layers (or transposed convolution) [92, 60, 64, 3, 71, 68] have been employed to recover the spatial resolution. Instead, we advocate the use of ‘atrous convolution’, originally developed for the efﬁcient computation of the undecimated wavelet transform in the “algorithme a trous” scheme of [36] and used before in the DCNN context by [26, 74, 66].`
- **一个二维信号，针对每个位置$i$，输出 $y$ 和filter $w$ ，对于输入 feature map $x$进行 atrous convlution 计算：**

$$
\boldsymbol{y}[\boldsymbol{i}]=\sum_{\boldsymbol{k}} \boldsymbol{x}[\boldsymbol{i}+r \cdot \boldsymbol{k}] \boldsymbol{w}[\boldsymbol{k}]
$$

- **其中，atrous rate $r$表示对输入信号进行采样的步长( stride)，等价于将输入$x$和通过在两个连续的filters值间沿着各空间维度插入 $r−1$个零值得到的上采样filters进行卷积。**
- 标准卷积即是atrous convlution 的一种rate $r=1$的特殊形式。空洞卷积如下图所示：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/11-2.jpg" style="zoom:50%;" />

#### 5.2 Going Deeper with Atrous Convolution

- 用级联的方式设计了空洞卷积模块。也可说为串行的方式。
- 具体而言，我们**取ResNet中最后一个block(ResNet的block4)，并将他们级联到了一起，如图所示：**

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/11-3.jpg)

- 有三个3×3卷积在这些块中，除了最后一个块，其余的模块中最后的一个卷积步长为2，类似于原来的ResNet。
- 这种网络模型设计的动机，**引入的 stride 能更容易的捕获较深的blockes中的大范围信息，例如，整体图像feature可以融合到最后一个小分辨率的 feature map 中，如Figure3(a).**

- 然而，我们发现连续的stride对语义分割是有害的，会造成细节信息的丢失。
- 因此，这里**采用由期望 outpur_stride 值来确定 rates 的atrous convolution 进行模型设计**，如Figure3(b)。采用串行的ResNet， 级联block为block5、block6、block7，均为block4的复制，如果没有 atrous convolution， 其output_stride=256。

- 注：**output stride为是输入图像的空间分辨率和输出特征图的空间分辨率的比值。**

##### 5.2.1 Multi-grid Method

- 在上面的图3中可得知，block4-block7有不同的空洞卷积率rate。且block的结构都一致，3个3 x 3的卷积。
- 文中**为block中的3个卷积定义了一个multi-grid，也叫做unit-rate = (r1,r2,r3)，该方法适用于3个block，计算每个block中3个卷积的方法为：rate = corresponding rate * (r1,r2,r3)。**
- 以block4为例，在图三中，corresponding rate = 2，若unit-rate= (1,2,4)，那么block4中的三个卷积的真实rate = 2 * (1,2,4) = (2,4,8)。

#### 5.3 Atrous Spatial Pyramid Pooling

- 我们重新审视了DeepLab V2中提出的**ASPP，其在特征映射的顶层并行应用了四个具有不同采样率的空洞卷积。**ASPP的灵感来自于空间金字塔池化，它表明在不同尺度上采样特征是有效的。**DeepLab V3的ASPP中包括了BN。**
- **具有不同 atrous rates 的 ASPP 能够有效的捕获多尺度信息. 不过，论文发现，随着sampling rate的增加，有效filter特征权重(即有效特征区域，而不是补零区域的权重)的数量会变小。**如下图所示：

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/11-4.jpg" style="zoom:50%;" />

- 当在65 × 65大小的特征图上应用不同采样率的3 × 3卷积时。在比率值已经接近于特征映射大小的极端情况下，这时的3 × 3卷积核已经无法捕获整个图像上下文信息，而是退化为一个简单的1×1卷积核，因为此时只有中心点的权重才是有效的。
- 为了克服这个问题，并将全局上下文信息纳入模型，我们采用了图像级特征。
- **具体来说，我们在模型的最后一个特征图采用全局平均池化，将重新生成的图像级别的特征提供给带256个滤波器(和BN)的1×1卷积，然后双线性插值将特征提升到所需的空间维度。**

- 改进的ASPP如下所示：

![](https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/11-5.jpg)

- (a) 当output_stride=16时，包括一个 1×1 convolution 和三个3×3 convolutions，其中3×3 convolutions的 rates=(6,12,18)，(所有的filter个数为256，并加入batch normalization) 需要注意的是，当output_stride=8时，rates将加倍。
- (b) 图像级特征:（即之前说的全局平均池化+送入256个1x1的卷积中）
- 然后将所有分支的特征图通过一个1×1卷积(有256个滤波器和BN)concatenate起来，送入最后的1×1卷积以产生最终分数。

### 6. Experimental Evaluation

- 采用**Image-Net预训练的ResNet为基础层，配合使用空洞卷积来提取密集特征**。
- **output_stride定义为输入图像的分辨率与最终输出分辨率的比值。**

#### 6.1 Training Protocol

- **Learning rate policy：**
  - 采用**poly策略(预设规则学习率变化法的一种)**， 在初始学习率基础上乘以$\left(1-\frac{\text {iter}}{\max _{-i \operatorname{ter}}}\right)^{p o w e r}$，power=0.9。
- **Crop size：**
  - **为了大采样率的空洞卷积能够有效，需要较大的图片裁剪尺寸；**否则，大采样率的空洞卷积权值就会主要用于padding区域。
  - 在Pascal VOC 2012数据集的训练和测试中我们采用了513的裁剪尺寸。

- **Batch normalization:**
  - 我们在ResNet之上添加的模块都包括BN层。
  - 当output_stride=16时，采用batchsize=16，同时BN层的参数做参数衰减0.9997。
  - 在增强的数据集上，以初始学习率0.007训练30K后，冻结BN层参数，然后采用output_stride=8，再使用初始学习率0.001在PASCAL官方的数据集上训练30K。
  - 训练output_stride=16比output_stride=8要快很多，因为其中间的特征映射在空间上小四倍。但output_stride=16在特征映射上相对粗糙，快是因为牺牲了精度。
- **Upsampling logits:**
  - 在先前的工作上，我们是将output_stride=8的输出与Ground Truth下采样8倍做比较。
  - 现在我们发现保持Ground Truth无损更重要，故我们是**将最终的输出上采样8倍与完整的Ground Truth比较。**

- **Data augmentation:**
  - 在训练阶段，随机缩放输入图像(从0.5到2.0)和随机左-右翻转。

#### 6.2 Going Deeper with Atrous Convolution

- 我们首先实验级联更多的空洞卷积模块。
- **ResNet-50：**使用ResNet-50时，我们探究output_stride的影响。

<img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/11-6.jpg" style="zoom:50%;" />

- `As shown in the table, in the case of output stride = 256 (i.e., no atrous convolution at all), the performance is much worse than the others due to the severe signal decimation.`

- 当output_stride为256时，由于严重的信号抽取，性能相比其他output_stride大大的下降了。
- 当使用不同采样率的空洞卷积时，性能上升，这说明了语义分割中使用空洞卷积的必要性。

- **ResNet-50 vs. ResNet-101:**

  - 用更深的模型，并改变级联模块的数量。

  <img src="https://blog-1258986886.cos.ap-beijing.myqcloud.com/paper/11-7.jpg" style="zoom:50%;" />

  - 当block增加，性能也随之增加。
  - 随着添加更多的block，提升变得更小。
  - 值得注意的是，ResNet-50使用block7会稍微降低性能，同时ResNet-101使用后仍然可以提升性能。

- 剩余部分都是类似以上实验过程的对比结果，详见论文。

#### 6.3 Atrous Spatial Pyramid Pooling

- DeepLab V3的ASPP模块与DeepLab V2的主要区别在于，增加了BN层，增加了图像级别的特征。

### 7. Conclusion

- `Our proposed model “DeepLabv3” employs atrous convolution with upsampled ﬁlters to extract dense feature maps and to capture long range context. Speciﬁcally, to encode multi-scale information, our proposed cascaded module gradually doubles the atrous rates while our proposed atrous spatial pyramid pooling module augmented with image-level features probes the features with ﬁlters at multiple sampling rates and effective ﬁeld-of-views. Our experimental results show that the proposed model signiﬁcantly improves over previous DeepLab versions and achieves comparable performance with other state-of-art models on the PASCAL VOC 2012 semantic image segmentation benchmark.`

