# StarVQA
StarVQA: Space-Time Attention for Video Quality Assessment

#  Installation

First, create a conda virtual environment and activate it:

```
conda create -n StarVQA python=3.7 -y
source activate StarVQA
```
Then, install the following packages:

- fvcore: pip install 'git+https://github.com/facebookresearch/fvcore'
- simplejson: pip install simplejson
- einops: pip install einops
- timm: pip install timm
- PyAV: conda install av -c conda-forge
- psutil: pip install psutil
- scikit-learn: pip install scikit-learn
- OpenCV: pip install opencv-python
- tensorboard: pip install tensorboard

Clone this repo.

```
git clone https://github.com/GZHU-DVL/StarVQA.git
cd StarVQA
python setup.py build develop
```

# Pretrain model
[**checkpoint-baidu**](https://pan.baidu.com/s/16z7erijruMTJNYyr2IWwfw) 提取码:87st


If you find StarVQA useful in your research, please use the following BibTeX entry for citation.
```
Citation:  @article{StarVQA2021,
   author={Fengchuang Xing, Yuan-Gen Wang, Hanpin Wang, Leida Li, and Guopu Zhu},
   title = {{StarVQA}: Space-Time Attention for Video Quality Assessment},
   booktitle = {arXiv preprint arXiv:2108.09635},
   pages = {1-5},
   year = {2021},
}
```
# Acknowledgements
StarVQA is built on top of [**TimeSformer**](https://github.com/facebookresearch/TimeSformer) and pytorch-image-models by [**Ross Wightman**](https://github.com/rwightman). We thank the authors for releasing their code. If you use our model, please consider citing these works as well:
```
@inproceedings{gberta_2021_ICML,
    author  = {Gedas Bertasius and Heng Wang and Lorenzo Torresani},
    title = {Is Space-Time Attention All You Need for Video Understanding?},
    booktitle   = {Proceedings of the International Conference on Machine Learning (ICML)}, 
    month = {July},
    year = {2021}
}
```
```
@misc{rw2019timm,
  author = {Ross Wightman},
  title = {PyTorch Image Models},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  doi = {10.5281/zenodo.4414861},
  howpublished = {\url{https://github.com/rwightman/pytorch-image-models}}
}
```
