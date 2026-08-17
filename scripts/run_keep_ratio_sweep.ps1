param(
    [string]$PythonExecutable = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$configs = @(
    "configs/fanin_dense_2048_inflection_pruned_0875.yaml",
    "configs/fanin_dense_2048_inflection_pruned_0750.yaml",
    "configs/fanin_dense_2048_inflection_pruned_0625.yaml",
    "configs/fanin_dense_2048_inflection_pruned_0500.yaml",
    "configs/fanin_dense_2560_inflection_pruned_0875.yaml",
    "configs/fanin_dense_2560_inflection_pruned_0750.yaml",
    "configs/fanin_dense_2560_inflection_pruned_0625.yaml",
    "configs/fanin_dense_2560_inflection_pruned_0500.yaml"
)

foreach ($config in $configs) {
    & $PythonExecutable -m prometheus.cli train --config $config
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}