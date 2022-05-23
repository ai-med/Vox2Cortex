---
permalink: /vox2cortex/
title: "Vox2Cortex"

---

The reconstruction of cortical surfaces from brain magnetic resonance imaging (MRI) scans is essential for quantitative analyses of cortical thickness and sulcal morphol-
ogy. Although traditional and deep learning-based algorithmic pipelines exist for this purpose, they have two major drawbacks: lengthy runtimes of multiple hours (traditional) or intricate post-processing, such as mesh extraction and topology correction (deep learning-based). In this work, we address both of these issues and propose Vox2Cortex, a deep learning-based algorithm that directly yields topologically correct, three-dimensional meshes of the boundaries of the cortex. Vox2Cortex leverages convolutional and graph convolutional neural networks to deform an initial template to the densely folded geometry of the cortex represented by an input MRI scan. We show in extensive experiments on three brain MRI datasets that our meshes are as accurate as the ones reconstructed by state-of-the-art methods in the field, without the need for time- and resource-intensive post-processing. To accurately reconstruct the tightly folded cortex, we work with meshes containing about
168,000 vertices at test time, scaling deep explicit reconstruction methods to a new level.


---



