# Music Genre Classification
![Music](static/music.jpg)

This repository contains some Exploratory Data Analysis and experiments which were held in order to create a model for music genre classification. Some shallow Machine Learning algorithms were used in order to perform classification on tabular data. Moreover, some neural network architectures were proposed to perform genre classification on tabular, sequence, or even image data.

## Data
[GTZA](https://www.kaggle.com/andradaolteanu/gtzan-dataset-music-genre-classification) dataset was used for Machine Learning training and evaluation. The data consists of 1000 tracks, each 30 seconds in length, and consisting of 10 genres. Each genre contains 100 tracks.

## Methods
In `EDA.ipynb`, data analysis and shallow model selection was performed for the data. The repository also contains a couple of models, which can be trained in order to perform classification. Models, such as `CWaveNet` (1D convolutions), `DenseNet` (Feed-forward network for tabular data), or common RNNs for wave data, or common 2D CNNs for visual frequency representations were used.


### Endnote
You may check some of my other projects [here](https://wprazuch.github.io/).
