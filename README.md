<div align="center">

# 🌟 What and Where to Adapt: Structure–Semantics Co-Tuning for Machine Vision Compression via Synergistic Adapters

**_Role-Guided Co-Tuning for Encoder-Decoder and Entropy Modeling!_**

[![arXiv](https://img.shields.io/badge/arXiv-2604.10017-b31b1b?style=flat-square&logo=arxiv)](https://arxiv.org/abs/2604.10017)
[![GitHub Stars](https://img.shields.io/github/stars/Brock-bit4/S2-CoT?style=flat-square&logo=github&color=ffca1c&cacheSeconds=300)](https://github.com/Brock-bit4/S2-CoT)
[![Visitors](https://visitor-badge.laobi.icu/badge?page_id=Brock-bit4.S2-CoT&style=flat-square)](https://github.com/Brock-bit4/S2-CoT)
[![License](https://img.shields.io/github/license/Brock-bit4/S2-CoT?style=flat-square&logo=opensourceinitiative&color=green&cacheSeconds=300)](https://github.com/Brock-bit4/S2-CoT/blob/main/LICENSE)

**Shaobo Liu**, Haobo Xiong, Kai Liu*, Yuna Lin

School of Computer Science and Technology, Xidian University

</div>

## 🏆 Awards
Awarded <span style="color:gold">***CVPR Compute Transparency Champion***</span> by CVPR 2026 Organizing Committee!


## 📌 Abstract
Parameter-efficient fine-tuning of pre-trained codecs is a promising direction in image compression for human and machine vision. While most existing works have primarily focused on tuning the feature structure within the encoder-decoder backbones, the adaptation of the statistical semantics within the entropy model has received limited attention despite its function of predicting the probability distribution of latent features. Our analysis reveals that naive adapter insertion into the entropy model can lead to suboptimal outcomes, underscoring that the effectiveness of adapter-based tuning depends critically on the coordination between adapter type and placement across the compression pipeline. Therefore, we introduce **S**tructure–**S**emantics **Co**-**T**uning (**S²-CoT**), a novel framework that achieves this coordination via two specialized, synergistic adapters: the Structural Fidelity Adapter (SFA) and the Semantic Context Adapter (SCA). SFA is integrated into the encoder-decoder to preserve high-fidelity representations by dynamically fusing spatial and frequency information; meanwhile, the SCA adapts the entropy model to align with SFA-tuned features by refining the channel context for more efficient statistical coding. Through joint optimization, S²-CoT turns potential performance degradation into synergistic gains, achieving state-of-the-art results across four diverse base codecs with only a small fraction of trainable parameters, closely matching full fine-tuning performance.


## 🎯 Highlights
✅ **Novel Co-Tuning Strategy**: Theoretically \& Experimentally Validate the Structure–Semantics Synergy<br>
✅ **What and Where to Adapt**: Adapter Effectiveness Depends on Matching Type to Its Placement<br>
✅ **Ultra-Lightweight Adapter**: Structural Fidelity Adapter (SFA) and Semantic Context Adapter (SCA)<br>
✅ **Strong Performance**: Outperforms SOTA methods on classification, detection, segmentation<br>
✅ **Comprehensive Analysis**: Full analysis in [arXiv](https://arxiv.org/abs/2604.10017) main text & appendix


## 🕒 Updates
[TODO] Release more codes.  
[2026/04/14] Initial release of this repo.<br>
[2026/07/06] Update error correction part.<br>
[2026/07/16] Release trained model checkpoints(ckpt) and inference logs.<br>


## 🚀 Overview
<div align="center">
<img src="./assets/overview.png" width="95%"/>
</div>


## 📊 Experimental Results
<div align="center">
<img src="./assets/results1.png" width="95%"/><br>
<img src="./assets/results2.png" width="95%"/>
</div>


## 📝 Updates & Errata
We will continuously update known typos, formula errors and description mistakes in the paper and repository.
- If you find any mistakes, typographical errors, or inconsistent experimental details, feel free to submit an Issue or pull request to notify us.
- All confirmed corrections will be recorded and updated in this section in real time.

### Record of Corrections
| Date       | Location                                                       |     Error Description     | Correction Content                                                          |
|------------|----------------------------------------------------------------|---------------------------|-----------------------------------------------------------------------------|
| 2026/07/06 | ①CvF->supp->Appendix.K->Tab.9<br>②arXiv.v1->Appendix.K->Tab.17 | Incorrect table value for trainable params of 'Ours' Method | Wrong:0.42 (0.35%)<br>***Right:0.67 (0.56%)***                              |
| 2026/07/16 | ①CvF->supp->Appendix.O<br>②arXiv.v1->Appendix.O                | Some codes sorted by SFA and SCA contain typographical inaccuracies | The complete version is ***in the SFA and SCA Class of this project code*** |


## 📚 Libraries & Dataset
All library versions are listed in `requirements.txt`. Please pay attention to the **COMMENTS** in this file.

**Datasets**

The following datasets are used and needed to be downloaded:  
- COCO2017 Train/Val for Detection and Segmentation
- Kodak for Detection and Segmentation
- ImageNet for Classification

## 📥 Inference: onlyTest
The model checkpoints used in this paper and inference logs are provided in [Google Drive](https://drive.google.com/drive/folders/1Taq24NirnRx_HnS5eVlbIHM3KOCIrsOl) as follows (notes: Retraining some models with missing weights yields performance metrics comparable to those reported in the original paper, with discrepancies very small and well within the acceptable tolerance.):
<table>
  <tr>
    <th style="text-align:center;">base codec</th>
    <th style="text-align:center;">task</th>
    <th style="text-align:center;">task_lmbda</th>
    <th style="text-align:center;">ckpt</th>
    <th style="text-align:center;">log(results)</th>
  </tr>
  <tr>
    <td rowspan="9">Lu2022-TIC</td>
    <td rowspan="4">det</td>
    <td style="text-align:center;">0.5</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1957YrNCfgXPUsih1QNjejjghpbs-KyTt/view?usp=drive_link">S2_CoT_det_1_checkpoint.pth.tar</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1Jk331x2hc4JjUt5QAQePUfvK_dola4di/view?usp=drive_link">S2_CoT_det_1.txt</a></td>
  </tr>
  <tr>
    <td style="text-align:center;">0.9</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1vMZGTkSR3Og-6QEu2OhN3S8_BOdXqSeW/view?usp=drive_link">S2_CoT_det_2_checkpoint.pth.tar</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1tzDidMqgJekwuVtfXgoHk60hG1Kkkxxg/view?usp=drive_link">S2_CoT_det_2.txt</a></td>
  </tr>
  <tr>
    <td style="text-align:center;">1.8</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1DEAo-r8SNN_0JhWWrTYkDShdny4bXn5o/view?usp=drive_link">S2_CoT_det_3_checkpoint.pth.tar</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/13SSRGR8gg4eDvEWOUy5mUlZua6HJw8-a/view?usp=drive_link">S2_CoT_det_3.txt</a></td>
  </tr>
  <tr>
    <td style="text-align:center;">3.2</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1WLX0sKVY1ik_XAIZXhyiFdRYZuK71Bch/view?usp=drive_link">S2_CoT_det_4_checkpoint.pth.tar</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1E8nkc46qX5WbxlwJHpbR5FuW1d33aZyM/view?usp=drive_link">S2_CoT_det_4.txt</a></td>
  </tr>
  <tr>
    <td rowspan="5">seg</td>
    <td style="text-align:center;">0.35</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/19VNFBWK5kU2myIf9O1ICvTmu5dCS8EmV/view?usp=drive_link">S2_CoT_seg_0_checkpoint.pth.tar</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1hOp5NuLhLAuV_WeQ8lfn5TaClWtqPsmZ/view?usp=drive_link">S2_CoT_seg_0.txt</a></td>
  </tr>
  <tr>
    <td style="text-align:center;">0.5</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1OV3hWkEBOMU3516dGHHyHPfUfk6DPVps/view?usp=drive_link">S2_CoT_seg_1_checkpoint.pth.tar</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1Y7btUdIcXPs73DywoboH-TY7xGjbYg1v/view?usp=drive_link">S2_CoT_seg_1.txt</a></td>
  </tr>
  <tr>
    <td style="text-align:center;">0.9</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1h0x3kse5vtushzCdBefMzVbkU6Yh6gsn/view?usp=drive_link">S2_CoT_seg_2_checkpoint.pth.tar</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1ohMuMUgTSyVacEA1ptF2YLrXPVntkDUO/view?usp=drive_link">S2_CoT_seg_2.txt</a></td>
  </tr>
  <tr>
    <td style="text-align:center;">1.8</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1lKN2OyrjpwwTX98yM8UkZtFc2maJ6fi5/view?usp=drive_link">S2_CoT_seg_3_checkpoint.pth.tar</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1d0_14OUaxoGyS_W-J_6PCTgw7VsMhGp2/view?usp=drive_link">S2_CoT_seg_3.txt</a></td>
  </tr>
  <tr>
    <td style="text-align:center;">3.2</td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1gD7Vidk90EBeWxPf3ZPKS95LTvLtDcLn/view?usp=drive_link">S2_CoT_seg_4_checkpoint.pth.tar</a></td>
    <td style="text-align:center;"><a href="https://drive.google.com/file/d/1GqBZVrTHdqd_dPnwlUDc3WRQZpYO5qpw/view?usp=drive_link">S2_CoT_seg_4.txt</a></td>
  </tr>
</table>

1. Download the trained checkpoints and place them at "path/to/best_ckpt.pth.tar";
2. Run the command below to conduct inference on a single/best trained model:

- Detection(onlyTest)
```shell
python examples/detection_base.py -c config/detection_base.yaml -ot -om "path/to/best_ckpt.pth.tar"
``` 
- Segmentation(onlyTest)
```shell
python examples/segmentation_base.py -c config/segmentation_base.yaml -ot -om "path/to/best_ckpt.pth.tar"
```

3. Other model checkpoints can be obtained by following the training procedure below.


## 🧩 Example: Train/Eval/Test
Different base codecs are sourced from their respective official libraries or latest research works. For instance, the Lu2022-TIC base codec (base_codec_1-4.pth.tar) is derived from [Adapt-ICMH](https://github.com/qingshi9974/ECCV2024-AdpatICMH).

### Detection(Train/Eval/Test)
```shell
python examples/detection_base.py -c config/detection_base.yaml -V -T
```
### Segmentation(Train/Eval/Test)
```shell
python examples/segmentation_base.py -c config/segmentation_base.yaml -V -T
```


## ⚡ Ackownledgment
Our work is based on the framework of [CompressAI](https://github.com/InterDigitalInc/CompressAI) and [TransTIC](https://github.com/NYCU-MAPL/TransTIC).
Also, thanks to ICMH for their open source work [Adapt-ICMH](https://github.com/qingshi9974/ECCV2024-AdpatICMH).


## 📬 Contact
If you have any questions, suggestions, or collaboration opportunities, feel free to contact us via email:
- **Shaobo Liu**: shaoboo.liu@stu.xidian.edu.cn  
- **Haobo Xiong**: hbxiong@stu.xidian.edu.cn  
- **Kai Liu\*** (Corresponding Author): kailiu@mail.xidian.edu.cn  
- **Yuna Lin**: ynling@stu.xidian.edu.cn  


## 📖 Citation
If you find our work useful in your research, please cite our paper:

> #### ✦ THIS WORK ✦

***1. Google Scholar*** :
```bibtex
@inproceedings{liu2026gs,
  title={What and Where to Adapt: Structure-Semantics Co-Tuning for Machine Vision Compression via Synergistic Adapters},
  author={Liu, Shaobo and Xiong, Haobo and Liu, Kai and Lin, Yuna},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2813--2822},
  year={2026}
}
```
***2. CVPR 2026 open access*** :
```bibtex
@InProceedings{Liu_2026_CVPR,
    author    = {Liu, Shaobo and Xiong, Haobo and Liu, Kai and Lin, Yuna},
    title     = {What and Where to Adapt: Structure-Semantics Co-Tuning for Machine Vision Compression via Synergistic Adapters},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Findings},
    month     = {June},
    year      = {2026},
    pages     = {2813-2822}
}
```
***3. arXiv*** :
```bibtex
@article{liu2026S2CoTarxiv,
  title={What and Where to Adapt: Structure–Semantics Co-Tuning for Machine Vision Compression via Synergistic Adapters},
  author={Liu, Shaobo and Xiong, Haobo and Liu, Kai and Lin, Yuna},
  journal={arXiv preprint arXiv:2604.10017},
  year={2026}
}
```

> #### ✦ OTHER WORKS ✦

***1. MDPI*** :
```bibtex
@Article{rs18060881,
 title={MambaLIC: State-Space Models for Efficient Remote Sensing Image Compression},
 author={Xiong, Haobo and Liu, Kai and Xiao, Huachao and Ding, Chongyang and Wang, Feiyang},
 journal={Remote Sensing},
 volume={18},
 year={2026},
 number={6},
 article-number={881},
 url={https://www.mdpi.com/2072-4292/18/6/881},
 issn={2072-4292},
 doi={10.3390/rs18060881}
}
```
