## USB-Net-ICS

- [Model](https://pan.baidu.com/s/1YtlxHySG3Qu4av1XvURQ0Q?pwd=ln2j)

Put the pth files in the folder "model".

- Test
```
python test.py --model=usbnet --cs_ratio=25
```
The results will be generated in the folder "./results/usbnet/{dataset}/{cs_ratio}/",
where results.csv will save the results in the format "{Image},{PSNR},{SSIM},{Time}".

- Train

1. Multi-GPU
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --model=usbnet --data_path="" --eval_data_path="" --cs_ratio=10 --blr=1e-4 --min_lr=1e-6 --epochs=400 --batch_size=16 --warmup_epochs=10 --input_size=96
```
2. Single GPU
```
python train.py --model=usbnet --data_path="" --eval_data_path="" --cs_ratio=10 --blr=1e-4 --min_lr=1e-6 --epochs=400 --batch_size=16 --warmup_epochs=10 --input_size=96
```

## USB-Net-CSMRI

- Model
1. [Cratesian](https://pan.baidu.com/s/1cBoOMFfGQyKPj3RpEuLqNw?pwd=d3qf)
2. [Radial](https://pan.baidu.com/s/1uIzmZ9y6H9TC2TEsfQz9XQ?pwd=1pi5)

Put the pth files in the folder "model-Cartesian" or "model-Radial".

- Test
```
python test.py --model=usbnet --cs_ratio=5 --input_size=256 --mask_type=Radial
```
The results will be generated in the folder "./results/usbnet/{dataset}/{cs_ratio}/",
where results.csv will save the results in the format "{Image},{PSNR},{SSIM},{Time}".

- Train

1. Multi-GPU
```
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.launch --nproc_per_node=2 train.py --model=usbnet --data_path="" --eval_data_path="" --cs_ratio=5 --blr=5e-5 --min_lr=1e-6 --epochs=100 --batch_size=1 --warmup_epochs=10 --input_size=256 --mask_type=Radial
```
2. Single GPU
```
python train.py --model=usbnet --data_path="" --eval_data_path="" --cs_ratio=5 --blr=5e-5 --min_lr=1e-6 --epochs=100 --batch_size=1 --warmup_epochs=10 --input_size=256 --mask_type=Radial
```

