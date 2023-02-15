## Adaptive-intra-group-feature-aggregation
The official repo of the ICASSP 2022 paper [Adaptive Intra-Group Aggregation for Co-Saliency Detection ](file:///tmp/mozilla_guangyu0/Adaptive_Intra-Group_Aggregation_for_Co-Saliency_Detection.pdf).


## Data Format

  Put the [DUTS_class (training dataset from GICD)](https://drive.google.com/file/d/1Ej6FKifpRi1bx09I0r7D6MO-GI8SDu_M/view?usp=sharing), [CoCA](http://zhaozhang.net/coca.html), [CoSOD3k](http://dpfan.net/CoSOD3K/) and [Cosal2015]() datasets to `./data` as the following structure:
  ```
  AIGANet
     ├── other codes
     ├── ...
     │ 
     └── data
           ├──── images
           |       ├── DUTS_class (DUTS_class's image files)
           |       ├── CoCA (CoCA's image files)
           |       ├── CoSOD3k (CoSOD3k's image files)
           │       └── Cosal2015 (Cosal2015's image files)
           │ 
           └────── gts
                    ├── DUTS_class (DUTS_class's Groundtruth files)
                    ├── CoCA (CoCA's Groundtruth files)
                    ├── CoSOD3k (CoSOD3k's Groundtruth files)
                    └── Cosal2015 (Cosal2015's Groundtruth files)
  ```  
  
<!-- USAGE EXAMPLES -->
## Usage

Run `sh all.sh` for training (`train_GPU0.sh`) and testing (`test.sh`).

## Citation
  ```
@inproceedings{ren2022adaptive,
  title={Adaptive Intra-Group Aggregation for Co-Saliency Detection},
  author={Ren, Guangyu and Dai, Tianhong and Stathaki, Tania},
  booktitle={ICASSP 2022-2022 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={2520--2524},
  year={2022},
  organization={IEEE}
}

  ```
