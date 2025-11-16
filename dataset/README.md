# Camera & IR-UWB Radars Fusion Vital Estimation Dataset

Datasets (Available if required). Corresponding Email: [huy.leminh@phenikaa-uni.edu.vn](mailto:huy.leminh@phenikaa-uni.edu.vn)

* The dataset contains Heart Beat data from 17 volunteers. In this project, one camera and three radars are used for data acquisition.
* The Camera is a Depth Camera sensor [IntelRealSenseD435i](https://www.intel.com/content/www/us/en/products/sku/190004/intel-realsense-depth-camera-d435i/specifications.html), manufactured by Intel.
* The FMCW Radar is a mmWave Radar sensor [AWR1243BOOST](https://www.ti.com/tool/AWR1243BOOST) and [DCA1000EVM](https://www.ti.com/tool/DCA1000EVM), both manufactured by Texas Instruments (TI).
* One of the IR-UWB radar is [Novelda](https://novelda.com/products/).
* The other IR-UWB radar is [UMAIN](https://umain.co.kr/)
* The ground-truth heart rate monitoring device is a [PPG](http://laxtha.net/ubpulse-h3/?ckattempt=3)

## Dataset should follow this structure

1 directory, 17 files
```
├── dataset
│   ├── BaLong_0.5_ideal.hdf5
│   │
│   └── ...
│
└── README.md
```