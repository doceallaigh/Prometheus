param(
    [int]$MaxAttempts = 4
)

$ErrorActionPreference = "Continue"
$env:PYTHONPATH = "src"

$runs = @(
    @{ cfg = "configs/rrs_j_cfc_s2_chains12.yaml"; base = "outputs/rrs-base-cot-c212-20260715-004843"; pattern = "rrs-j-cfc-s2-chains12-*"; report = "reports/20260714-s2-chains12-local-eval.md" },
    @{ cfg = "configs/rrs_j_cfc_s2_width512.yaml"; base = "outputs/rrs-base-cot-w512-20260714-234122"; pattern = "rrs-j-cfc-s2-width512-*"; report = "reports/20260714-s2-width512-local-eval.md" },
    @{ cfg = "configs/rrs_j_cfc_s2_seed2.yaml"; base = "outputs/rrs-base-cot-20260710-035418"; pattern = "rrs-j-cfc-s2-seed2-*"; report = "reports/20260714-s2-seed2-local-eval.md" }
)

foreach ($r in $runs) {
    $ok = $false
    for ($i = 1; $i -le $MaxAttempts -and -not $ok; $i++) {
        Write-Host "=== train $($r.cfg) attempt $i ==="
        & .\.venv\Scripts\python.exe -u -m prometheus.cli train --config $r.cfg 2>&1 | Select-Object -Last 3
        if ($LASTEXITCODE -eq 0) { $ok = $true }
    }
    if (-not $ok) {
        Write-Host "TRAIN FAILED: $($r.cfg)"
        continue
    }

    $latent = Get-ChildItem outputs -Directory -Filter $r.pattern | Sort-Object Name -Descending | Select-Object -First 1
    if (-not $latent) {
        Write-Host "NO RUN DIR for $($r.pattern)"
        continue
    }

    $ok = $false
    for ($i = 1; $i -le $MaxAttempts -and -not $ok; $i++) {
        Write-Host "=== eval $($latent.Name) attempt $i ==="
        & .\.venv\Scripts\python.exe -u -m prometheus.cli evaluate-reasoning `
            --base-run $r.base `
            --latent-run $latent.FullName `
            --num-problems 300 `
            --device cuda `
            --output $r.report 2>&1 | Select-Object -Last 3
        if ($LASTEXITCODE -eq 0) { $ok = $true }
    }
    if (-not $ok) {
        Write-Host "EVAL FAILED: $($latent.Name)"
    }
}

Write-Host "=== local stage 2 complete ==="
