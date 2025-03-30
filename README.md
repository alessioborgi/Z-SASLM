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
    <img src="https://img.shields.io/badge/CVPR%202025-Workshop-blue" alt="CVPR 2025 Workshop Accepted" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://arxiv.org/abs/XXXX">
    <img src="https://img.shields.io/badge/arXiv-XXXX-orange" alt="arXiv" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://www.researchgate.net/publication/390303255_Z-SASLM_Zero-Shot_Style-Aligned_SLI_Blending_Latent_Manipulation">
    <img src="https://img.shields.io/badge/ResearchGate-Paper-00CCBB?logo=ResearchGate&logoColor=white" alt="ResearchGate" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://paperswithcode.com/paper/XXXXX">
    <img src="https://img.shields.io/badge/Papers%20with%20Code-Enabled-9cf?logo=paperswithcode&logoColor=white" alt="Papers with Code" style="height: 25px; margin-right: 5px;">
  </a>
  <a href="https://www.academia.edu/128519694/Z_SASLM_Zero_Shot_Style_Aligned_SLI_Blending_Latent_Manipulation">
    <img src="https://img.shields.io/badge/Academia-Visit-blue" alt="Academia.edu" style="height: 25px;">
  </a>
</p>


---

## Overview 🚀

This is an *Official Implementation* of the paper **Z-SASLM: Zero-Shot Style-Aligned SLI Blending Latent Manipulation**, accepted at **CVPR 2025, Workshop on AI for Creative Visual Content Generation, Editing, and Understanding**. Z-SASLM is a zero-shot framework for multi-style image synthesis that leverages Spherical Linear Interpolation (SLI) to achieve smooth, coherent blending—without any fine-tuning. 

<div>
  <img src="assets/blending/SLERP_2_Styles_Blending_MedCub.png" alt="Screenshot" width="500" style="display: block; margin: 0 auto;">
</div>


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

<div style="text-align: center">
  <img src="assets/method/architecture.png" alt="Screenshot" width="1200"/>
</div>

Our framework is built as a modular pipeline that efficiently combines diverse style references and multi-modal cues without fine-tuning. The architecture comprises four main components:

1. **Reference Image Encoding & Blending:**  
   - A Variational Autoencoder (VAE) extracts latent representations from each reference style image.  
   - Our novel Spherical Linear Interpolation (SLI) Blending module then fuses these latent codes along the geodesic of the hypersphere, ensuring smooth and coherent style transitions.

2. **Text Encoding:**  
   - Textual prompts are encoded using a CLIP-based module, capturing semantic cues and aligning them with visual features.  
   - This stage supports both simple captions and richer prompts derived from multiple modalities.

3. **Style-Aligned Image Generation:**  
   - The blended style representation is combined with the text embeddings to condition a diffusion-based generation process.  
   - A style-aligned attention mechanism reinforces consistent style propagation throughout the image generation.

4. **Optional Multi-Modal Content Fusion:**  
   - Additional inputs such as audio, music, or weather data are first transformed into text.  
   - These are fused into a single “Multi-Content Textual Prompt” via a T5-based rephrasing module, further enriching the conditioning signal for improved creative synthesis.


---

## Results & Examples 📊

<div style="text-align: center">
  <img src="assets/method/architecture.png" alt="Screenshot" width="1200"/>
</div>

Our experimental evaluation confirms the effectiveness of Z-SASLM across various style blending scenarios:

- **Style Consistency:**  
  Quantitative comparisons using our innovative Weighted Multi-Style DINO VIT-B/8 metric demonstrate that our SLI Blending significantly outperforms conventional linear interpolation—producing images with robust and coherent style alignment.

- **Visual Quality:**  
  As shown in the example figures below, Z-SASLM preserves fine stylistic details and avoids the abrupt transitions common in linear blending.  
  *[Insert side-by-side comparison figures or example images here]*

- **Multi-Modal Fusion:**  
  Our ablation studies reveal that incorporating diverse content (e.g., audio and weather data) further enhances the richness of the generated visuals, confirming the benefits of our multi-modal integration.

Overall, the results validate that Z-SASLM not only improves style consistency but also delivers high-fidelity images even under challenging multi-reference conditions. Explore the [notebooks](./notebooks) for interactive examples and detailed visual comparisons.

<div style="text-align: center">
  <img src="assets/blending/SLERP_3_Styles_Blending_EgyVanGoghMacro.png" width="1200">
</div>
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
