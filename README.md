# Z-SASLM: Zero-Shot Style-Aligned SLI Blending Latent Manipulation

[![CVPR 2025 Workshop on AI for Creative Visual Content Generation, Editing and Understanding Accepted](https://img.shields.io/badge/CVPR%202025-Workshop-blue)](https://https://cveu.github.io//) 
[![Paper Accepted](https://img.shields.io/badge/Paper-Accepted-green)](https://github.com/alessioborgi/Z-SASLM)

---

## Overview

**Z-SASLM** is a novel Zero-Shot Style-Aligned SLI Blending Latent Manipulation pipeline that overcomes the limitations of current multi-style blending methods. By leveraging the non-linear geometry of the latent space through SLI Blending, our approach ensures high-fidelity, coherent style integration—all without any fine-tuning. 

This work has been **accepted to the CVPR 2025 Workshop on AI for Creative Visual Content Generation, Editing and Understanding**, where we showcase both the theoretical advancements and practical benefits of our method in multi-modal content fusion.

---

## Abstract

> We introduce **Z-SASLM**, a Zero-Shot Style-Aligned SLI (Spherical Linear Interpolation) Blending Latent Manipulation pipeline that overcomes the limitations of current multi-style blending methods. Conventional approaches rely on linear blending, assuming a flat latent space leading to suboptimal results when integrating multiple reference styles. In contrast, our framework leverages the non-linear geometry of the latent space by using SLI Blending to combine weighted style representations. By interpolating along the geodesic on the hypersphere, Z-SASLM preserves the intrinsic structure of the latent space, ensuring high-fidelity and coherent blending of diverse styles—all without the need for fine-tuning. We further propose a new metric, **Weighted Multi-Style DINO VIT-B/8**, designed to quantitatively evaluate the consistency of the blended styles. While our primary focus is on the theoretical and practical advantages of SLI Blending for style manipulation, we also demonstrate its effectiveness in a multi-modal content fusion setting through comprehensive experimental studies. Experimental results show that Z-SASLM achieves enhanced and robust style alignment.

---

## Features

- **Zero-Shot Capability:** No fine-tuning required for new styles.
- **SLI Blending:** Uses geodesic interpolation on the hypersphere for smooth, natural style transitions.
- **Non-Linear Latent Navigation:** Preserves the intrinsic structure of the latent space.
- **Innovative Evaluation Metric:** Introduces Weighted Multi-Style DINO VIT-B/8 for quantitative assessment.
- **Multi-Modal Fusion:** Demonstrates robust performance across diverse content fusion scenarios.

---

## Installation

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/alessioborgi/Z-SASLM.git
cd Z-SASLM
pip install -r requirements.txt
```

---

## Usage

Below is a quick example to get started with Z-SASLM:

```python
from z_saslm import ZSASLM

# Initialize the pipeline with default settings
pipeline = ZSASLM()

# Specify the source image and a list of reference style images
source_image = "path/to/source_image.jpg"
style_images = ["path/to/style1.jpg", "path/to/style2.jpg"]

# Blend the styles using the SLI Blending approach
result = pipeline.blend(source_image, style_images)

# Save the output image
result.save("path/to/output_image.jpg")
```

For more detailed examples, please check out the [notebooks](./notebooks) directory.

---

## Citation

If you use this work in your research, please cite our paper:

```bibtex
@inproceedings{borgi2025z-saslm,
  title={Z-SASLM: Zero-Shot Style-Aligned SLI Blending Latent Manipulation},
  author={Borgi, Alessandro and others},
  booktitle={CVPR 2025 Workshop on AI for Creative Visual Content Generation, Editing and Understanding},
  year={2025}
}
```

---

## Contributing

Contributions are highly welcome! Please see our [contributing guidelines](CONTRIBUTING.md) for more details on how to help improve the project.

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements

We would like to extend our gratitude to the CVPR 2025 Workshop organizers and the entire research community for their support and feedback.

---

*Stay creative and keep pushing the boundaries of style manipulation!*
