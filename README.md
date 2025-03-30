###  RVDNET: A TWO-STAGE NETWORK FOR REAL-WORLD VIDEO DESNOWING WITH DOMAIN ADAPTATION (ICASSP 2024)

**ABSTRACT**   Video snow removal is an important task in computer vision, as the snowflakes in videos reduce visibility and negatively affect the performance of outdoor visual systems. 
However, due to the complexity of real snowy scenarios, it is difficult to apply existing supervised learning-based methods to process real-world snowy videos. 
In this paper, we propose a novel two-stage video desnow network for the real world, called RVDNet. 
The first stage of RVDNet utilizes Spatial Feature Extraction Modules (SFEM) to extract the spatial features of the input frames. 
In the second stage, we design Spatial-Temporal Desnowing Modules (STDM) to remove snowflakes via spatio-temporal learning. 
Furthermore, we introduce the unsupervised domain adaptation module, which is embedded for aligning the feature space of real and synthetic data in the spatial and spatio-temporal domains, 
respectively. Experiments on the proposed SnowScape dataset prove that our method has superior desnow performance not only on synthetic data, but also in the real world.

### Installation
To replicate the environment:

```bash
cd code
conda install --file requirements.txt
```

### Preparing dataset
Please prepare the data before testing and training by modifying and runing scripts/gen_json_ntu.py and scripts/gen_json_ntu_real.py.

### Training
**Please first modify bash files accordingly with your data folder path.**

```bash
cd code/run_scripts
```
Train on SnowScape:

```bash
cd code/RUN_SCRIPTS/
bash train_snow_dvd_noLSTM.sh
```

### Testing

```bash
cd code/RUN_SCRIPTS/
bash test_dvd.sh
```

### Citation
```bibtex
@INPROCEEDINGS{10448423,
  author={Xue, Tianhao and Zhou, Gang and He, Runlin and Wang, Zhong and Chen, Juan and Jia, Zhenhong},
  booktitle={ICASSP 2024 - 2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)}, 
  title={RVDNet: A Two-Stage Network for Real-World Video Desnowing with Domain Adaptation}, 
  year={2024},
  volume={},
  number={},
  pages={3305-3309},
  keywords={Learning systems;Computer vision;Snow;Visual systems;Signal processing;Feature extraction;Spatial databases;Video desnowing;unsupervised domain adaptation;spatio-temporal learning},
  doi={10.1109/ICASSP48485.2024.10448423}}
```

### Acknowledgement
We learned a lot from **ESTINet** and **FastDVDNet**. You can search and read them for further information.
