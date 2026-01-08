# Quick training script with correct path
$drivePath = Get-ChildItem "f:\PHD\AI in Med\lastchanse" | Where-Object {$_.Name -like "*DRIVE*training*" -and $_.PSIsContainer} | Select-Object -First 1 -ExpandProperty FullName
$fullPath = Join-Path $drivePath "training"

Write-Output "Using path: $fullPath"

.\venv\Scripts\python.exe train_diffusion.py --dataset drive --dataset_path $fullPath --output_dir ./outputs/drive --batch_size 4 --image_size 512 --num_epochs 100 --lr 1e-4 --device cuda
