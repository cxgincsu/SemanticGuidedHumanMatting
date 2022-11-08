<h2 align="center">Semantic Guided Human Matting - SGHM</h2>

<div align="center"><i>Robust Human Matting via Semantic Guidance (ACCV 2022)</i></div>

<img src="figs/teaser.gif" width="100%">

<div align="center"><b>SGHM is a  robust and accurate method for automatic human matting  which requires no trimap input. Semantic guidance is well incorporated into our model to predict coarse mask and fine-grained alpha matte successively.</b></div>
<div align="center"><b>SGHM是一个鲁棒、高质量的自动人像抠图方法。我们将语义监督很好地融入到抠图模型中，能够依次输出粗糙分割Mask图和精细Alpha图。</b></div>

<p align="center">
  <a href="https://arxiv.org/abs/2210.05210">Paper</a>
</p>


## Highlights

- **Semantic Guided Network :** A segmentation sub-network is first employed for the task of segmentation, and then it is reused to guide the matting process to focus on the surrounding area of the segmentation mask. To improve the performance and reduce computation, we share semantic encoder in two tasks. Under the guidance of powerful semantic features, our matting module successfully handle many challenging cases.
- **Data Efficient :** With only about 200 matting images, our method is able to produce high quality alpha details.  We can efficiently improve matting performance by collecting more coarse human masks in an easy and fast way rather than paying for the high cost fine-detailed alpha annotating.
- **SOTA Result :** We conduct comparisons on 5 benchmarks  qualitatively and quantitatively. SGHM outperforms other methods across all benchmarks.

<img src="figs/network.png" width="100%">

<img src="figs/alpha.png" width="100%">

## Usage ##

#### Requirements

- [x] python
- [x] pytorch
- [x] torchvision
- [x] opencv
- [x] tqdm

#### Testing

1. You can download our pretraind weights from [link](https://drive.google.com/drive/folders/15mGzPJQFEchaZHt9vgbmyOy46XxWtEOZ?usp=sharing) (google drive)  or [link](https://pan.baidu.com/s/147xULbZ_FPNot0YWLZ-isQ) (百度云, 提取码: u8g4) and save it in the`./pretrained` folder.  

2. Test your own images

   ```python
   python test_image.py \
       --images-dir "PATH_TO_IMAGES_DIR" \
       --result-dir "PATH_TO_RESULT_DIR" \
       --pretrained-weight ./pretrained/SGHM-ResNet50.pth
   ```

3. **Evaluate** on benchmarks 

   ```python
   python test_image.py \
       --images-dir "PATH_TO_IMAGES_DIR" \
       --gt-dir "PATH_TO_GT_ALPHA_DIR" \
       --result-dir "PATH_TO_RESULT_DIR" \
       --pretrained-weight ./pretrained/SGHM-ResNet50.pth
   ```

4.  Test your video

   ```python
   python test_video.py \
       --video "PATH_TO_INPUT_VIDEO" \
       --output-video "PATH_TO_OUTPUT_VIDEO" \
       --pretrained-weight ./pretrained/SGHM-ResNet50.pth
   ```


## Bibtex

If you use this code for your research, please consider to star this repo and cite our paper.

 ```latex
@inproceedings{chen2022sghm,
  author = {Chen, Xiangguang and Zhu, Ye and Li, Yu and Fu, Bingtao and Sun, Lei and Shan, Ying and Liu, Shan},
  title = {Robust Human Matting via Semantic Guidance},
  booktitle={Proceedings of the Asian Conference on Computer Vision (ACCV)},
  year={2022}
}
 ```

## Acknowledgement
In this project, parts of the code are adapted from : [BMV2](https://github.com/PeterL1n/BackgroundMattingV2) and [MG](https://github.com/yucornetto/MGMatting) . We thank the authors for sharing codes for their great works.
