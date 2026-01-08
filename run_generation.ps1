# Generate augmented data using trained model
$drivePath = Get-ChildItem "f:\PHD\AI in Med\lastchanse" | Where-Object {$_.Name -like "*DRIVE*training*" -and $_.PSIsContainer} | Select-Object -First 1 -ExpandProperty FullName
$fullPath = Join-Path $drivePath "training"

Write-Output "Using dataset path: $fullPath"
Write-Output "Generating augmented data..."

.\venv\Scripts\python.exe generate_augmented_data.py --checkpoint ./outputs/drive/checkpoints/final_model.pt --dataset drive --dataset_path $fullPath --output_dir ./augmented_data/drive --num_augmentations 5 --device cuda
