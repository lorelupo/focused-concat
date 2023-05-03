This repository contains the official implementation of the experiments discussed in two papers:

- [Focused Concatenation for Context-Aware Neural Machine Translation](https://aclanthology.org/2022.wmt-1.77/), presented at WMT2022.
- [Encoding Sentence Position in Context-Aware Neural Machine Translation with Concatenation](https://arxiv.org/abs/2302.06459), presented at Insights2023.

# Requirements and Installation

The implementation is based on the [fairseq package v0.9.0](https://github.com/pytorch/fairseq/tree/v0.9.0).

[fairseq](https://github.com/pytorch/fairseq) requires the following:

* [PyTorch](http://pytorch.org/) version >= 1.4.0
* Python version >= 3.6
* For training models, you'll also need an NVIDIA GPU and [NCCL](https://github.com/NVIDIA/nccl)

To install and develop locally:
```bash
cd focused-concat
pip install --editable ./
```

# Relevant files

The most relevant files describing the approaches presented in the paper are:

1. [concat_transformer.py](fairseq/models/concat_transformer.py), where the architecture of the concatenation NMT model is implemented;
2. [doc2doc_translation.py](fairseq/tasks/doc2doc_translation.py), where the task of NMT with windows of concatenated sentences is implemented, as well as the metod for discounting the loss computed on context;
3. [sent2doc_dataset.py](fairseq/data/sent2doc_dataset.py), containing the class that organizes the data in batches of sequences formed by concatenated sentences.

## Replicate results

The steps to reproduce our s4to4 results on IWSLT En-De are enumerated below:

1. Prepare WMT17 dataset `bash sh/prepare/en-de/wmt17/sent_all.sh`
    - this is needed for the BPE encoding, which was trained on WMT17)

2. Download IWSLT7 [from WIT3](https://wit3.fbk.eu/2017-01) and unzip it in `./data/en-de/orig/iwslt17`

3. Prepare IWSLT17 `bash sh/prepare/en-de/iwslt17/doc_all.sh standard`

4. Prepare ContraPro `bash sh/prepare/en-de/wmt17/large_pronoun.sh --k=3 --shuffle=False`

5. Train and evaluate models:

    1. s4to4: `bash sh/slurm/en-de/naver-submit-s4to4.slurm`
    2. s4to4+CD: `bash sh/slurm/en-de/naver-submit-cd-s4to4.slurm`
    3. s4to4+shift+CD: `bash sh/slurm/en-de/naver-submit-cd-segshift-s4to4.slurm`


For the English-Russian language pair, you can find analogous scripts in this repo. The corpus used in the paper is extracted from OpenSubtitles2018 by [Voita et al. (2019)](https://www.aclweb.org/anthology/P19-1116/), and it is available [here](https://github.com/lena-voita/good-translation-wrong-in-context#training-data).


# Citation

The [Focused Concatenation](https://aclanthology.org/2022.wmt-1.77/) paper:
```bibtex
@inproceedings{lupo-etal-2022-focused,
    title = "Focused Concatenation for Context-Aware Neural Machine Translation",
    author = "Lupo, Lorenzo  and
      Dinarelli, Marco  and
      Besacier, Laurent",
    booktitle = "Proceedings of the Seventh Conference on Machine Translation (WMT)",
    month = dec,
    year = "2022",
    address = "Abu Dhabi, United Arab Emirates (Hybrid)",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.wmt-1.77",
    pages = "830--842",
}
```

The [Encoding sentence position](https://arxiv.org/abs/2302.06459) paper:
```bibtex
@article{lupo2023encoding,
      title={Encoding Sentence Position in Context-Aware Neural Machine Translation with Concatenation}, 
      author={Lorenzo Lupo and Marco Dinarelli and Laurent Besacier},
      year={2023},
      eprint={2302.06459},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```

For Fairseq, please cite:

```bibtex
@inproceedings{ott2019fairseq,
  title = {fairseq: A Fast, Extensible Toolkit for Sequence Modeling},
  author = {Myle Ott and Sergey Edunov and Alexei Baevski and Angela Fan and Sam Gross and Nathan Ng and David Grangier and Michael Auli},
  booktitle = {Proceedings of NAACL-HLT 2019: Demonstrations},
  year = {2019},
}
```