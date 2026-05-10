# Related Works: ECT Hardware System Development (近5年电容层析成像硬件系统研究进展)

> 本文档为SCI论文 Introduction 中 Related Works 部分的草稿，面向电容层析成像（ECT）硬件系统方向。
> 可根据目标期刊风格和具体研究内容进行裁剪与补充。

---

## Related Works (English)

Electrical capacitance tomography (ECT) reconstructs the cross-sectional permittivity distribution inside a vessel or pipe from a set of inter-electrode capacitance measurements. Because the reconstruction quality is fundamentally limited by the accuracy, speed, and stability of the front-end hardware, the evolution of ECT systems over the past five years has been largely driven by advances in three coupled layers: (i) capacitance-sensing circuitry, (ii) sensor and electrode structures, and (iii) the digital back-end that couples data acquisition with real-time reconstruction.

**Capacitance-sensing front ends.** Early commercial ECT units were dominated by charge/discharge architectures with relatively modest SNR and frame rates. More recent work has moved decisively toward AC-based designs that combine sine-wave excitation with digital I/Q demodulation, enabling simultaneous improvement of SNR and temporal resolution; representative systems now report frame rates of the order of hundreds of frames per second and SNRs above 70 dB [1,2]. In parallel, capacitance-to-digital converter (CDC) techniques originally developed for MEMS readout—such as iterative delay-chain discharge CDCs and continuous-time bandpass ΔΣ modulators achieving sub-10 aF resolution in standard CMOS processes—have been ported to tomographic front ends, allowing fully digital, chip-scale capacitance sensing with greatly reduced sensitivity to stray capacitance [3,4]. Dedicated studies on switch topology and guard-electrode shielding for electrical capacitance volume tomography (ECVT) have further quantified and suppressed parasitic coupling, which remains the single dominant error source in femto-/atto-farad range measurements [5].

**Sensor and electrode structures.** The conventional cylindrical, externally mounted 8/12-electrode sensor has been extended in several directions. For 3D imaging, ECVT systems employ dense electrode arrays distributed over the full volume and are increasingly fabricated with additive manufacturing: 3D-printed multilayer sensor bodies allow reproducible placement of electrodes, guard layers, and shielding with accuracy unattainable by hand assembly [6,7]. For one-sided inspection and defect detection, planar and dual-plane miniature arrays (for example, 3×3 matrix planar sensors) have been proposed to enable depth-resolved imaging from a single accessible surface [8]. At the opposite end of the scale, micro-scale ECT implemented directly on a CMOS sensor array has recently been demonstrated for label-free 3-D imaging of biological samples, integrating thousands of active pixels with on-chip switching and readout and pushing the spatial resolution of ECT from the millimetre to the micrometre regime [9].

**Real-time back-ends and system integration.** Driven by applications such as fluidised-bed monitoring, multiphase flow metering and, more recently, in-flight propellant mass gauging under microgravity, ECT hardware has been tightly coupled to dedicated computational back-ends. Reviews of electrical tomography hardware accelerators highlight a clear trend from single-core CPU implementations to FPGA-, GPU- and multi-GPU-based pipelines that perform forward projection and iterative reconstruction in real time, with FPGA designs favoured when deterministic latency and compactness are critical, and (multi-)GPU clusters favoured for 3D/ECVT workloads with large sensitivity matrices [10,11,12]. Modular, multi-channel DAQ platforms such as the EVT4 system (up to 32 channels with per-channel ADCs, supporting both 2D and 3D imaging) exemplify the current state of the art in laboratory-grade ECT hardware, while integrated products such as ITOMS M3C demonstrate industrial deployments operating at 50 fps with 12 electrodes at 500 kHz excitation [13,14]. At the same time, portable and application-specific ECT front ends have begun to emerge, including systems tailored to harsh industrial environments, biomedical pilots for intracerebral haemorrhage detection, and microgravity propellant gauging experiments flown by NASA [15,16,17].

**Remaining gaps.** Despite this rapid progress, several issues persist and motivate the present work. First, most high-performance front ends remain bulky, power-hungry, and dependent on proprietary FPGA/ASIC designs, which limits deployment in portable or embedded scenarios. Second, the achievable dynamic range is still constrained by stray- and coupling-capacitance effects that are strongly geometry-dependent and are only partially addressed by existing shielding and switch-topology solutions. Third, although 3D-printed and CMOS-integrated sensors have substantially expanded the design space, the co-design of sensor geometry, front-end circuitry, and back-end reconstruction—rather than optimising each in isolation—is still in its infancy. Addressing these gaps is the central motivation of this study.

---

## 中文要点说明

这段 Related Works 围绕**近5年ECT硬件系统三条主线**展开：

| 主线 | 核心发展 | 代表工作 |
|------|---------|---------|
| **前端测量电路** | 由传统充放电转向AC正弦激励+数字I/Q解调；CDC达到亚aF分辨率；抗杂散电容的开关拓扑与保护电极 | ITOMS M3C、Cicalini 2021、ΔΣ CDC 2022、ECVT stray capacitance研究 |
| **传感器与电极结构** | 3D打印多层传感器取代手工组装；平面/双层3×3 mini阵列用于单侧缺陷检测；CMOS像素阵列把ECT推进到微米尺度 | Banasiak 2019/2022、Sensor Review 2023、MicroECT (BioCAS 2023) |
| **后端与系统集成** | FPGA/GPU/多GPU实时重建；EVT4 32通道DAQ；面向便携/生物医学/微重力专用ECT | Alhamad 2022 review、EVT4、Frontiers Physics 2024、NASA 2023 |

**剩余空白（论文切入点）**：便携化与嵌入式部署、杂散电容带来的动态范围限制、以及**传感器-前端-重建算法的协同设计**。

---

## References

> ⚠️ 请在提交前核实所有文献的 DOI、卷号、页码等细节。

1. ITOMS Ltd., *M3C Electrical Capacitance Tomography — Product Datasheet*, 2023.
2. Yang, W. Q. *Tutorial: Electrical capacitance tomography and industrial applications*, IEEE I2MTC Tutorials, 2016. (AC-based ECT, 300 fps, SNR 73 dB)
3. Cicalini, M. et al. "Design of a capacitance-to-digital converter based on iterative delay-chain discharge in 180 nm CMOS technology," *Sensors*, vol. 22, no. 1, p. 121, 2021.
4. Omran, H. et al. "A continuous-time bandpass ΔΣ capacitance-to-digital converter with 3.68 aFrms resolution in 0.18 μm CMOS," *IEEE TCAS-I*, 2022.
5. Taruno, W. P. et al. "Switch configuration effect on stray capacitance in electrical capacitance volume tomography hardware," *Measurement*, 2020.
6. Banasiak, R. et al. "3D-printed multilayer sensor structure for electrical capacitance tomography," *Sensors*, vol. 19, no. 15, p. 3416, 2019.
7. Tang, Y. et al. "A measurement compensation method for ECT sensors with inhomogeneous electrode parameters," *Sensors*, 2022.
8. "Dual-plane miniature planar 3D ECT sensor based on 3×3 matrix electrode array," *Sensor Review*, vol. 43, 2023.
9. Levy, D. et al. "Microscale 3-D capacitance tomography with a CMOS sensor array," *IEEE BioCAS* / arXiv:2309.09039, 2023.
10. Alhamad, R. & Nassif, A. B. "Electrical tomography hardware systems for real-time applications," *Electronics/Sensors* (review), 2022.
11. Majchrowicz, M. et al. "Multi-GPU, multi-node algorithms for acceleration of image reconstruction in 3D ECT in heterogeneous distributed systems," *Sensors*, vol. 20, no. 2, p. 391, 2020.
12. Jamaludin, J. et al. "FPGA implementation of ECT digital system for imaging conductive materials," *Algorithms*, vol. 12, no. 2, p. 28, 2019.
13. Wajman, R. et al. *EVT4 multi-channel ECT data acquisition system*, Warsaw University of Technology, 2021.
14. Cui, Z. et al. "Review of selected advances in electrical capacitance volume tomography for multiphase flow monitoring," *Energies*, vol. 15, p. 5285, 2022.
15. Ren, Z. et al. "Research on ECT detection of cerebral hemorrhage based on symmetrical cancellation method," *Frontiers in Physics*, vol. 12, 1392767, 2024.
16. NASA Glenn Research Center, "Propellant mass gauging in microgravity with ECT," NTRS 20230012805, 2023.
17. Sun, B. et al. "Portable ECT systems and hardware design: a review," *Sensor Review*, 2016; updated in subsequent portable-ECT works (2021–2023).
