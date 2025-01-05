
# LSKNet

![lsk_arch](docs/lsk.png)

## Introduction

LSKNet adalah "A Foundation Lightweight Backbone for Remote Sensing"

## Results and models

Imagenet 300-epoch pre-trained LSKNet-T backbone: [Download](https://download.openmmlab.com/mmrotate/v1.0/lsknet/backbones/lsk_t_backbone-2ef8a593.pth)

Imagenet 300-epoch pre-trained LSKNet-S backbone: [Download](https://download.openmmlab.com/mmrotate/v1.0/lsknet/backbones/lsk_s_backbone-e9d2e551.pth)

DOTA1.0

|                           Model                            |  mAP  | Angle | lr schd | Batch Size |                                   Configs                                    |                                                               Download                                                               |     note     |
| :--------------------------------------------------------: | :---: | :---: | :-----: | :--------: | :--------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :----------: |
| [RTMDet-l](https://arxiv.org/abs/2212.07784) (1024,1024,-) | 81.33 |   -   | 3x-ema  |     8      |                                      -                                       |                                                                  -                                                                   |  Prev. Best  |
|                  LSKNet_T (1024,1024,200) + ORCNN          | 81.37 | le90  |   1x    |    2\*8    |     [lsk_t_fpn_1x_dota_le90](./configs/lsknet/lsk_t_fpn_1x_dota_le90.py)     | [model](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_t_fpn_1x_dota_le90/lsk_t_fpn_1x_dota_le90_20230206-3ccee254.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_t_fpn_1x_dota_le90/lsk_t_fpn_1x_dota_le90_20230206.log) |              |
|                  LSKNet_S (1024,1024,200) + ORCNN          | 81.64 | le90  |   1x    |    1\*8    |   [lsk_s_fpn_1x_dota_le90](./configs/lsknet/lsk_s_fpn_1x_dota_le90.py)       | [model](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_fpn_1x_dota_le90/lsk_s_fpn_1x_dota_le90_20230116-99749191.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_fpn_1x_dota_le90/lsk_s_fpn_1x_dota_le90_20230116.log) |              |
|                 LSKNet_S\* (1024,1024,200) + ORCNN         | 81.85 | le90  |   1x    |    1\*8    | [lsk_s_ema_fpn_1x_dota_le90](./configs/lsknet/lsk_s_ema_fpn_1x_dota_le90.py) | [model](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_ema_fpn_1x_dota_le90/lsk_s_ema_fpn_1x_dota_le90_20230212-30ed4041.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_ema_fpn_1x_dota_le90/lsk_s_ema_fpn_1x_dota_le90_20230212.log) | EMA Finetune |
|                  LSKNet_S (1024,1024,200) + Roi_Trans      | 81.22 | le90  |   1x    |    2\*8    |   [lsk_s_roitrans_fpn_1x_dota](./configs/lsknet/lsk_s_roitrans_fpn_1x_dota.py)   | [model](https://pan.baidu.com/s/1OhK5juH__L9CeVKQoHFkDQ?pwd=lsks) \| [log](https://pan.baidu.com/s/1MQj0N9qcfPPWiZRlZ2Ad7A?pwd=lsks) |              |
|                  LSKNet_S (1024,1024,200) + R3Det          | 80.08 | oc    |   1x    |    2\*8    |   [lsk_s_r3det_fpn_1x_dota](./configs/lsknet/lsk_s_r3det_fpn_1x_dota.py)   | [model](https://pan.baidu.com/s/186A8Q_j4lNxCp3JcEWy2Bw?pwd=lsks) \| [log](https://pan.baidu.com/s/1xN1GOg1qV7pqhlgUCk0FTQ?pwd=lsks) |              |
|                  LSKNet_S (1024,1024,200) + S2ANet         | 81.32 | le135 |   1x    |    2\*8    |   [lsk_s_s2anet_fpn_1x_dota](./configs/lsknet/lsk_s_s2anet_fpn_1x_dota.py)   | [model](https://pan.baidu.com/s/1bQ41PBzK-OUQX_FYKDO32A?pwd=lsks) \| [log](https://pan.baidu.com/s/1Q4MtKVkyxmFrjW5SMEbTPQ?pwd=lsks) |              |

FAIR1M-1.0

|         Model         |  mAP  | Angle | lr schd | Batch Size |                                                    Configs                                                     |                                                                                                                                                                              Download     | note                                                                                                                                                                         |
| :----------------------: | :---: | :---: | :-----: | :------: | :------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------: | :--------: |
| [O-RCNN](https://arxiv.org/abs/2108.05699) (1024,1024,200) | 45.60 | le90  |   1x    |    1*8     |  [oriented_rcnn_r50_fpn_1x_fair_le90](./configs/oriented_rcnn/oriented_rcnn_r50_fpn_1x_fair_le90.py)  |      -   | Prev. Best |
| LSKNet_S (1024,1024,200) | 47.87 | le90  |   1x    |    1*8     |            [lsk_s_fpn_1x_dota_le90](./configs/lsknet/lsk_s_fpn_1x_dota_le90.py)             |         [model](https://pan.baidu.com/s/1sXyi23PhVwpuMRRdwsIJlQ?pwd=izs8) \| [log](https://pan.baidu.com/s/1idHq3--oyaWK3GWYqd8brQ?pwd=zznm)         | |

HRSC2016 

|                    Model                     | mAP(07) | mAP(12) | Angle | lr schd | Batch Size |                                      Configs                                      |                                                               Download                                                               |    note    |
| :------------------------------------------: | :-----: | :-----: | :---: | :-----: | :--------: | :-------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------: | :--------: |
| [RTMDet-l](https://arxiv.org/abs/2212.07784) |  90.60  |  97.10  | le90  |   3x    |     -      |                                         -                                         |                                                                  -                                                                   | Prev. Best |
|  [ReDet](https://arxiv.org/abs/2103.07733)   |  90.46  |  97.63  | le90  |   3x    |    2\*4    | [redet_re50_refpn_3x_hrsc_le90](./configs/redet/redet_re50_refpn_3x_hrsc_le90.py) |                                                                  -                                                                   | Prev. Best |
|                   LSKNet_S                   |  90.65  |  98.46  | le90  |   3x    |    1\*8    |       [lsk_s_fpn_3x_hrsc_le90](./configs/lsknet/lsk_s_fpn_3x_hrsc_le90.py)        | [model](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_fpn_3x_hrsc_le90/lsk_s_fpn_3x_hrsc_le90_20230205-4a4a39ce.pth) \| [log](https://download.openmmlab.com/mmrotate/v1.0/lsknet/lsk_s_fpn_3x_hrsc_le90/lsk_s_fpn_3x_hrsc_le90_20230205-4a4a39ce.pth) |            |

# MMRotate Installation Guide

MMRotate bergantung pada [PyTorch](https://pytorch.org/), [MMCV](https://github.com/open-mmlab/mmcv), dan [MMDetection](https://github.com/open-mmlab/mmdetection). Berikut adalah langkah-langkah cepat untuk instalasi. Untuk petunjuk yang lebih rinci, silakan merujuk ke [Panduan Instalasi](https://mmrotate.readthedocs.io/en/latest/install.html).

## 1. Instalasi Anaconda

### Unduh dan Instal Anaconda:
```shell
# Unduh Anaconda
curl -O https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh

# Instal Anaconda
bash Anaconda3-2024.10-1-Linux-x86_64.sh -b 

# Reload terminal
source ~/anaconda3/bin/activate
source ~/.bashrc
```

## 2. Instal MMRotate

### Buat dan Aktifkan Environment Conda:
```shell
# Buat environment Conda
conda create --name openmmlab python=3.8 -y
conda activate openmmlab

# Instal PyTorch dan dependensi lainnya
conda install pytorch==1.8.0 torchvision==0.9.0 cudatoolkit=10.2 -c pytorch
```

### Instal Modul MMRotate:
```shell
# Instal OpenMMLab
pip install -U openmim
mim install mmcv-full
mim install mmdet
```

### Clone Repository LSKNet:
```shell
git clone https://github.com/zcablii/Large-Selective-Kernel-Network.git
cd Large-Selective-Kernel-Network
pip install -v -e .
```

### Update Ubuntu dan Instal unzip:
```shell
apt update
apt install unzip -y
```

## 3. Menghubungkan Conda Environment dengan Kernel Jupyter
```shell
conda install ipykernel
pip install timm roboflow jupyter future tensorboard gdown
python -m ipykernel install --user --name=openmmlab --display-name "openmmlab"
```

## 4. Menjalankan Jupyter Notebook
```shell
jupyter notebook
```

## 5. Catatan Penting

### Penggunaan Single GPU
Jika hanya menggunakan satu GPU, ubah `SyncBN` menjadi `BN` di konfigurasi yang digunakan:
```shell
# Sebelum
norm_cfg=dict(type='SyncBN', requires_grad=True)

# Sesudah
norm_cfg=dict(type='BN', requires_grad=True)
```
Konfigurasi dapat ditemukan di:
`configs/lsknet/lsk_s_ema_fpn_1x_dota_le90.py`

### Menjalankan Jupyter Notebook di Server Non-Local
```shell
jupyter notebook --ip=0.0.0.0 --port=8889 --no-browser --allow-root
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.allow_origin='*' --NotebookApp.allow_remote_access=True
```

## 6. Isu yang Mungkin Terjadi
Jika menghadapi masalah selama instalasi atau penggunaan, periksa issue berikut:
- [Issue #1066](https://github.com/open-mmlab/mmrotate/issues/1066)
- [Issue #945](https://github.com/open-mmlab/mmrotate/issues/945)


## Get Started

Please see [get_started.md](docs/en/get_started.md) for the basic usage of MMRotate.
We provide [colab tutorial](demo/MMRotate_Tutorial.ipynb), and other tutorials for:

- [learn the basics](docs/en/intro.md)
- [learn the config](docs/en/tutorials/customize_config.md)
- [customize dataset](docs/en/tutorials/customize_dataset.md)
- [customize model](docs/en/tutorials/customize_models.md)
- [useful tools](docs/en/tutorials/useful_tools.md)


# Tutorial Menjalankan RunPod

## 1. Deploy Pod di RunPod

1. Akses [RunPod.io](https://www.runpod.io/) dan login.
2. Pergi ke menu **Pods** dan lakukan deploy.
3. Pilih GPU yang direkomendasikan: **A40 (VRAM 48GB)** atau **RTX4090 (VRAM 24GB)**.
4. Konfigurasi deployment:
   - Template: **RunPod PyTorch 2.4.0**
   - Edit konfigurasi:
     - Ekspose **port 6007** (untuk tensorboard)
     - Sesuaikan **storage** sesuai kebutuhan (**disarankan 30GB**).
5. Klik **Deploy On-Demand** dan tunggu hingga pod menyala.

## 2. Menghubungkan ke Pod

1. Setelah pod menyala, klik **Connect** → pilih **Connect to Jupyter Lab [Port 8888]**.
2. Setelah launcher terbuka, pilih **Terminal**.
3. Clone repository dengan perintah berikut:
   ```bash
   git clone https://github.com/AbiyaMakruf/TelU-Tubes-TugasAkhir-LSKNet.git
   ```
4. Verifikasi versi CUDA dan GCC:
   ```bash
   nvcc -V
   gcc --version
   ```
   Pastikan keduanya terdeteksi agar proses training berjalan dengan lancar.

## 3. Instalasi Dependencies

Setiap kali pod dinyalakan, jalankan perintah berikut:
```bash
pip install torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 --index-url https://download.pytorch.org/whl/cu118
pip install -U openmim
mim install mmcv-full
mim install mmdet
apt update
apt install unzip -y
pip install timm roboflow future tensorboard gdown
```

## 4. Instalasi MMRotate

```bash
cd TelU-Tubes-TugasAkhir-LSKNet/
pip install -v -e .
```

## 5. Modifikasi Kode (Hanya Sekali)

Buka file `mmrotate/core/post_processing.py`, lalu ubah baris **42**:

**Sebelum:**
```python
labels = torch.arange(num_classes, dtype=torch.long)
```

**Sesudah:**
```python
labels = torch.arange(num_classes, dtype=torch.long, device=scores.device)
```

Jika sudah pernah diubah, **tidak perlu diubah kembali!**

## 6. Menjalankan Training

1. Buka `notebook.ipynb` di Jupyter Lab.
2. Jalankan sel-sel di notebook untuk memulai training.

## 7. Akses tensorboard setelah training berjalan

1. Buka website runpod.
2. Buka menu Pods dan pilih pod yang digunakan.
3. klik **Connect** → pilih **Connect to HTTP Service [Port 6007]**.

---

Silakan ikuti langkah-langkah di atas untuk menjalankan training menggunakan RunPod. Jika mengalami masalah, cek dokumentasi resmi atau forum diskusi komunitas. 🚀