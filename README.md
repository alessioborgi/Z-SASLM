<div align="center">

# <b>Z-SASLM: Zero-Shot Style-Aligned SLI Blending Latent Manipulation

[Alessio Borgi](https://www.linkedin.com/in/alessio-borgi-a85b461a2/), [Luca Maiano](https://scholar.google.com/citations?user=FZyBVqkAAAAJ&hl=it&oi=ao), [Irene Amerini](https://scholar.google.com/citations?user=4ZDhr6UAAAAJ&hl=it&oi=ao)

Sapienza University of Rome

### <b>[CVPR 2025](https://cvpr.thecvf.com/) [Workshop on AI for Creative Visual Content Generation, Editing, and Understanding](https://cveu.github.io/)

</div>

<p align="center">
  <a href="https://cveu.github.io/">
    <img src="https://img.shields.io/badge/CVPR%202025-Workshop-blue" alt="CVPR 2025 Workshop Accepted" style="height: 30px; margin-right: 5px;">
  </a>
  <a href="https://arxiv.org/abs/XXXX">
    <img src="https://img.shields.io/badge/arXiv-XXXX-orange" alt="arXiv" style="height: 30px; margin-right: 5px;">
  </a>
  <a href="https://github.com/alessioborgi/Z-SASLM">
    <img src="https://img.shields.io/badge/Paper-Accepted-green" alt="Paper Accepted" style="height: 30px; margin-right: 5px;">
  </a>
  <a href="LICENSE">
    <img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT" style="height: 30px; margin-right: 5px;">
  </a>

</p>


![Header Image](./assets/cover_image.gif)

---

## Overview 🚀

This is an *Official Implementation* of the paper **Z-SASLM: Zero-Shot Style-Aligned SLI Blending Latent Manipulation**, accepted at **CVPR 2025, Workshop on AI for Creative Visual Content Generation, Editing, and Understanding**. Z-SASLM is a zero-shot framework for multi-style image synthesis that leverages Spherical Linear Interpolation (SLI) to achieve smooth, coherent blending—without any fine-tuning. 
---

## Abstract ✨

> We introduce **Z-SASLM**, a **Zero-Shot Style-Aligned SLI (Spherical Linear Interpolation) Blending Latent Manipulation** pipeline that overcomes the limitations of current multi-style blending methods. Conventional approaches rely on linear blending, assuming a flat latent space leading to suboptimal results when integrating multiple reference styles. In contrast, our framework leverages the non-linear geometry of the latent space by using SLI Blending to combine weighted style representations. By interpolating along the geodesic on the hypersphere, Z-SASLM preserves the intrinsic structure of the latent space, ensuring high-fidelity and coherent blending of diverse styles—all without the need for fine-tuning. We further propose a new metric, Weighted Multi-Style DINO VIT-B/8, designed to quantitatively evaluate the consistency of the blended styles. While our primary focus is on the theoretical and practical advantages of SLI Blending for style manipulation, we also demonstrate its effectiveness in a multi-modal content fusion setting through comprehensive experimental studies. Experimental results show that Z-SASLM achieves enhanced and robust style alignment. 

---

## Features 🔥

- **Zero-Shot Capability:** No fine-tuning needed for new styles.
- **SLI Blending:** Geodesic interpolation ensures natural and coherent style transitions.
- **Intrinsic Latent Navigation:** Respects the non-linear structure of the latent space.
- **Innovative Metric:** Introduces Weighted Multi-Style DINO VIT-B/8 for robust evaluation.
- **Multi-Modal Fusion:** Effectively blends styles across varied content scenarios.

---

## Installation 🛠️

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/alessioborgi/Z-SASLM.git
cd Z-SASLM
pip install -r requirements.txt
```

---

## Quick Start 🚀

Here's a simple example to get you up and running:

```python
from z_saslm import ZSASLM

# Initialize the pipeline
pipeline = ZSASLM()

# Define source and style images
source_image = "path/to/source_image.jpg"
style_images = ["path/to/style1.jpg", "path/to/style2.jpg"]

# Perform SLI Blending
result = pipeline.blend(source_image, style_images)

# Save the blended image
result.save("path/to/output_image.jpg")
```

For more details, check out our [notebooks](./notebooks) directory.

---

## Live Demo 🎥

Watch our demo video to see Z-SASLM in action:  
[![Watch the Demo](https://img.youtube.com/vi/your_video_id/0.jpg)](https://www.youtube.com/watch?v=your_video_id)

---

## Citation 📚

If you find our work useful, please cite our paper:

```bibtex
@inproceedings{borgi2025z-saslm,
  title={Z-SASLM: Zero-Shot Style-Aligned SLI Blending Latent Manipulation},
  author={Borgi, Alessandro and others},
  booktitle={CVPR 2025 Workshop on AI for Creative Visual Content Generation, Editing and Understanding},
  year={2025}
}
```

---

## Contributing 🤝

Contributions are highly welcome! Please see our [contributing guidelines](CONTRIBUTING.md) for details on how you can help.

---

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements 🙏

Special thanks to the CVPR 2025 Workshop organizers and the research community for their support and feedback.  
*Stay creative and push the boundaries of style manipulation!*
