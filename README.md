# MatchXML
This is the official repo of the paper [MatchXML: An Efficient Text-label Matching Framework for Extreme Multi-label Text
Classification](???)


## Install the environment
* Create a virtual environment
    ```bash
    # We recommend you to use Anaconda to create a conda environment 
    conda create --name matchxml python=3.8
    conda activate matchxml
    ```
* Install the required software:
    ```bash
    pip install -r requirements.txt
    ```
## Prepare Data
* Download six XMC datasets from [XR-Transformer](https://github.com/amzn/pecos/tree/mainline/examples/xr-transformer-neurips21)

* Download our trained label embeddings from [Google Drive](https://drive.google.com/drive/folders/1ehOU7mRpDdsCORVlVaSL7LidaOBonS5l?usp=sharing) and save them to `xmc-base/{dataset}`
* Download our static text features(__static sentence embeddings__ + TF-IDF features)  from [Google Drive](https://drive.google.com/drive/folders/1ehOU7mRpDdsCORVlVaSL7LidaOBonS5l?usp=sharing) and save them to `xmc-base/{dataset}`


## Training MatchXML and evaluation
 `# eurlex-4k, wiki10-31k, amazoncat-31k, wiki-500k, amazon-670k, amazon-3m`
 
 `bash run.sh {dataset}`


## Training label2vec
`# eurlex-4k, wiki10-31k, amazoncat-31k, wiki-500k, amazon-670k, amazon-3m`

`bash ./label2vec_run/{dataset}.sh`  

## Pre-trained models
* Our pre-trained models can be downloaded from [Google Drive](???)

## Citation
If you find this work useful in your research, please consider citing:

```
@article{???,
  title={MatchXML: An Efficient Text-label Matching Framework for Extreme Multi-label Text Classification},
  author={Ye, Hui and Sunderraman, Rajshekhar and Ji, Shihao},
  journal={???},
  year={2022}
}

```
[comment]: <> (## Acknowledgment)
[comment]: <> (Our work is based on the following work:)
[comment]: <> (- [Fast Multi-Resolution Transformer Fine-tuning for Extreme Multi-label Text Classification])

