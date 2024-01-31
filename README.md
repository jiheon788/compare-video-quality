# Video Quality Comparative Analysis: MP4 vs. WebM

## Introduction

This study analyzes video quality degradation during MP4 to WebM format conversion, critical in frontend development and digital content management. Utilizing advanced metrics like PSNR, SSIM, and VIF, the research evaluates these formats comprehensively.

## Research Objectives

The goal is to assess WebM's fidelity compared to MP4 and understand the impact of video optimization in frontend development. This is key for selecting video formats and optimization strategies, balancing quality and efficiency on digital platforms.

## Methodological Framework

- **Tools**: Conducted using `Python`, with `OpenCV`, `NumPy`, and `skimage`.
- **Sampling Strategy**: Frame-by-frame analysis at one-second intervals, balancing thoroughness with efficiency.
- **Metrics:** PSNR for peak error measurement, SSIM for perceptual degradation, and VIF for visual quality assessment.
  - `PSNR`: A measure of the peak error between the original and compressed video.
  - `SSIM`: An index quantifying the perceptual degradation as perceived change in structural information.
  - `VIF`: A scale that assesses visual quality based on natural scene statistics and human visual system modeling.

## Analytical Insights

- **PSNR and SSIM**: High values indicate minimal quality deviation between formats.
- **VIF**: Suggests slight perceptual variances under certain conditions.
- **Anomaly Detection**: Noted in monochrome frames, highlighting the need for contextual metric interpretation.

## Conclusion

This project aids frontend developers in making informed decisions about video optimization in web applications. It offers insights into the impact of format conversion on video quality, aiding in achieving the ideal balance between quality and performance.
