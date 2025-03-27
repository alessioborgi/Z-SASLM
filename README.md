<div align="center">

# <b>Z-SASLM: Zero-Shot Style-Aligned SLI Blending Latent Manipulation

[Alessio Borgi](https://www.linkedin.com/in/alessio-borgi-a85b461a2)<sup>*</sup>, [Luca Maiano](https://scholar.google.com/citations?user=FZyBVqkAAAAJ&hl=it&oi=ao), [Irene Amerini](https://scholar.google.com/citations?user=4ZDhr6UAAAAJ&hl=it&oi=ao)

Sapienza University of Rome
<p>
  <a href="mailto:borgi.1952442@studenti.uniroma1.it">borgi.1952442@studenti.uniroma1.it</a>,
  <a href="mailto:maiano@diag.uniroma1.it">maiano@diag.uniroma1.it</a>,
  <a href="mailto:amerini@diag.uniroma1.it">amerini@diag.uniroma1.it</a>
</p>

<p align="center"><sup>*</sup>Corresponding author: <a href="mailto:borgi.1952442@studenti.uniroma1.it">borgi.1952442@studenti.uniroma1.it</a></p>

### <b>[CVPR 2025](https://cvpr.thecvf.com/) [Workshop on AI for Creative Visual Content Generation, Editing, and Understanding](https://cveu.github.io/)

</div>


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

- **Zero-Shot Versatility:** Unlock infinite style possibilities without any fine-tuning.
- **SLI Blending for Multi-Reference Style Conditioning:** Introduces a novel architecture that leverages spherical linear interpolation to seamlessly blend multiple reference styles without any fine-tuning.
- **Latent Space Mastery:** Capitalizes on the intrinsic non-linearity of the latent manifold for optimal style integration.
- **Innovative Evaluation Metric:** Proposes the Weighted Multi-Style DINO VIT-B/8 metric to rigorously quantify style consistency across generated images.
- **Multi-Modal Content Fusion:** Demonstrates the framework’s robustness by integrating diverse modalities—such as image, audio, and weather data—into a unified content fusion approach.


---

## Installation 🛠️

Clone the repository and install the necessary dependencies:

```bash
git clone https://github.com/alessioborgi/Z-SASLM.git
cd Z-SASLM
pip install -r requirements.txt
```

---

## Architecture Proposed 📐


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

## License 📄

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgements 🙏

Special thanks to the CVPR 2025 Workshop organizers and the research community for their support and feedback.  
*Stay creative and push the boundaries of style manipulation!*
