This is the official implementation of the paper titled [AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism](https://arxiv.org/abs/2601.22442).

## Instructions

First install `requirements.txt` and run the bash script `run.bash`. This script assumes an instance with at least 8 GPUs and runs our method for the base model on the `4 x 2 mesh`. Tested on PyTorch 2.5.1, CUDA 12.6, and Python 3.12.

## Acknowledgements
- [AsyncPP](https://github.com/PluralisResearch/AsyncPP)
- [SPARTA](https://github.com/matttreed/diloco-sim)

## License
Copyright © Pluralis Research. All rights reserved.

This project is licensed under the MIT License. See the LICENSE file for details.

## Citation
```
@article{ajanthan2026asyncmesh,
  title={AsyncMesh: Fully Asynchronous Optimization for Data and Pipeline Parallelism},
  author={Ajanthan, Thalaiyasingam and Ramasinghe, Sameera and Avraham, Gil and Mohaghegh Dolatabadi, Hadi and P Hewa Koneputugodage, Chamin and Shevchenko, Violetta and Zuo, Yan and Long, Alexander},
  journal={arXiv:2601.22442},
  year={2026}
}
```

