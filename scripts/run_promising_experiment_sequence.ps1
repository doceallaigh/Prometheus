param(
    [string]$PythonExecutable = ".\.venv\Scripts\python.exe"
)

$ErrorActionPreference = "Stop"
$env:PYTHONPATH = "src"

$configs = @(
    "configs/fanin_dense_2048.yaml",
    "configs/fanin_dense_2560.yaml",
    "configs/variant_modular_cluster_graph_matched.yaml",
    "configs/variant_modular_cluster_graph_large_2048.yaml",
    "configs/variant_modular_cluster_graph_matched_topk2.yaml",
    "configs/variant_modular_cluster_graph_matched_topk1.yaml",
    "configs/variant_modular_cluster_graph_large_2048_topk2.yaml",
    "configs/variant_modular_cluster_graph_large_2048_topk1.yaml"
)

foreach ($config in $configs) {
    & $PythonExecutable -m prometheus.cli train --config $config
    if ($LASTEXITCODE -ne 0) {
        exit $LASTEXITCODE
    }
}