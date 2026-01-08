# Train vessel segmentation model with augmented data
$drivePath = Get-ChildItem "f:\PHD\AI in Med\lastchanse" | Where-Object {$_.Name -like "*DRIVE*training*" -and $_.PSIsContainer} | Select-Object -First 1 -ExpandProperty FullName
$fullPath = Join-Path $drivePath "training"

Write-Output "Training segmentation model with augmented data..."
Write-Output "Dataset path: $fullPath"

.\venv\Scripts\python.exe train_segmentation.py --dataset_path $fullPath --output_dir ./segmentation_outputs --batch_size 8 --num_epochs 50 --lr 1e-4 --device cuda
