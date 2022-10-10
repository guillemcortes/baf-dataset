This repository contains the necessary scripts to reproduce the work: "BAF: An Audio Fingerprinting Dataset For Broadcast Monitoring".

[![paper](https://img.shields.io/badge/preprint-10230/54139-green?logo=arxiv)](https://repositori.upf.edu/handle/10230/54139)

# Dataset
Broadcast Audio Fingerprinting dataset is an open, available upon request, annotated dataset for the task of music monitoring in broadcast. It contains 2,000 tracks from Epidemic Sound's private catalogue as reference tracks that represent 74 hours. As queries, it contains over 57 hours of TV broadcast audio from 23 countries and 203 channels distributed with 3,425 one-min audio excerpts.

It has been annotated by six annotators in total and each query has been cross-annotated by three of them obtaining high inter-annotator agreement percentages, which validates the annotation methodology and ensures the reliability of the annotations.  

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.6868083.svg)](https://doi.org/10.5281/zenodo.6868083) 

## Downloading the data
The dataset is available for conducting non-commercial research related to audio analysis. It shall not be used for music generation or music synthesis. It is **available upon request** on [Zenodo](https://doi.org/10.5281/zenodo.6868083) alongside an extended **description** of the dataset contents, **motivation**, **license**, **ownership** of the data, and the dataset **datasheet**. 

# Algorithms
Configuration files are located at [baf-dataset/configs](https://github.com/guillemcortes/baf-dataset/tree/master/configs).

* **Audfprint** 
code repository: https://github.com/dpwe/audfprint

* **Panako / Olaf**
code repository: https://github.com/JorenSix/panako (at its [2.1 version release](https://github.com/JorenSix/Panako/commit/20d94a2444677c9e4917dbb1a881a2599b657ba0))

* **NeuralFP**
code repository: https://github.com/guillemcortes/neural-audio-fp (forked from https://github.com/mimbres/neural-audio-fp)

* **PeakFP**
code can be found in this repository at [baf-dataset/peakfp](https://github.com/guillemcortes/baf-dataset/tree/master/peakfp) directory

# Code
```
baf-dataset/
├── compute_statistics.py --> Script to generate metrics
├── configs --> Parameter configurations used
│   ├── audfprint.cfg
│   ├── …
│   └── panako.cfg
└── peakfp --> Fingerprinting baseline
    ├── README.md
    ├── constants.py
    ├── …
    └── utils.py
```

# Installation
The authors recommend the use of virtual environments.

**Requirements:**
* Python 3.6+
* Create virtual environment and install requirements
```bash
git clone https://github.com/guillemcortes/baf-dataset.git
cd baf-dataset
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

# License
* The code in this repository is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0)
* Dataset license is detailed in [Zenodo](https://zenodo.org/record/6868083)

# Citation
Please cite the following [publication]() when using the dataset:

> Guillem Cortès, Álex Ciurana, Emilio Molina, Marius Miron, Owen Meyers, Joren Six and Xavier Serra. BAF: An Audio Fingerprinting Dataset for Broadcast Monitoring. In Proc. of the 23rd Int. Soc. for Music Information Retrieval Conf. (ISMIR), pages XXX-XXX, Bangalore, India, 2022.

Bibtex version:

```
TODO
```

# Acknowledgements

This research is part of *NextCore – New generation of music monitoring technology (RTC2019-007248-7)*, funded by the Spanish Ministerio de Ciencia e Innovación and the Agencia Estatal de Investigación. Also, has received support from Industrial Doctorates plan of the Secretaria d’universitats i Recerca, Departament d’Empresa i Coneixement de la Generalitat de Catalunya, grant agreement No. DI46-2020.

<p align="center">
  <img src="https://user-images.githubusercontent.com/25322884/186637988-7bb1f775-2eac-4110-9961-ad7bbf8cb520.png" height="50" hspace="10" vspace="10" />
  <img src="https://user-images.githubusercontent.com/25322884/186637802-f9c8bbb9-bcf2-4b5e-9407-1fdd49c9aa9b.jpg" height="50" hspace="10" vspace="10"/>
</p>
<p align="center">
  <img src="https://user-images.githubusercontent.com/25322884/186637854-50e06004-9dc6-40ee-8ec9-701899136a6e.png" height="50" hspace="10" vspace="10"/>
  <img src="https://user-images.githubusercontent.com/25322884/186637746-e18c4517-250c-4474-b11e-58df1e1f0787.jpeg" height="50" hspace="10" vspace="10"/>
  <img src="https://user-images.githubusercontent.com/25322884/186637861-24a64957-f82b-4faa-be34-5b1221bbd05c.png" height="50" hspace="10" vspace="10"/>
</p>