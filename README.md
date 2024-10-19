# WSI Patch Loader

An efficient Whole Slide Image (WSI) patch loader designed for large-scale pathology image analysis.

## Project Description

WSI Patch Loader is a high-performance data loading tool designed to handle large volumes of pathological slide images (.ndpi files). It utilizes MONAI and cuCIM libraries to achieve fast patch cutting and loading, making it particularly suitable for data preprocessing in deep learning model training.

Key Features:
- Efficient processing of large-scale WSI datasets
- GPU-accelerated patch cutting
- Adapted for multi-node DGX H100 environments
- Implements distributed data loading and processing

## Tech Stack

- Python 3.8+
- MONAI
- cuCIM
- PyTorch
- NVIDIA CUDA

## Installation and Usage

1. Clone the repository:
   ```bash
   git clone https://github.com/y-h-Lin/wsi-patch-loader.git
   cd wsi-patch-loader
   ```

3. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

4. Configure data path:
Set the path for WSI files in the `config.yaml` file.

5. Run the loader:
   ```bash
   python main.py
   ```


## Example Code

```python
from wsi_loader import WSIDataset, get_dataloader

dataset = WSIDataset(wsi_files, patch_size=1024, stride_size=512)
dataloader = get_dataloader(dataset, batch_size=32, num_workers=4)

for batch in dataloader:
 # Perform model training
```   
