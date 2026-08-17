param(
    [string]$ProjectId = "project-c78041ad-9ad3-4697-86e",
    [string[]]$Zones = @("us-central1-a", "us-central1-b", "us-central1-c", "us-central1-f", "us-east1-c", "us-east1-d", "us-west1-b", "us-west1-c"),
    [string]$BucketUri = "gs://prometheus-rrs-stage-artifacts",
    [int]$Stage1BudgetUsd = 250,
    [int]$PerRunMaxMinutes = 150,
    [switch]$RunStage1,
    [switch]$RunStage2,
    [int]$Stage2BudgetUsd = 500,
    [int]$Stage2PerRunMaxMinutes = 360,
    [switch]$AllowCpuFallback = $true,
    [string[]]$Stage1RunKeys = @("main", "embed", "gru"),
    [string[]]$Stage2RunKeys = @("chains12", "width512", "seed2")
)

$ErrorActionPreference = "Stop"
if (Get-Variable -Name PSNativeCommandUseErrorActionPreference -ErrorAction SilentlyContinue) {
    $PSNativeCommandUseErrorActionPreference = $false
}

function Write-Info {
    param([string]$Message)
    Write-Host "[$(Get-Date -Format s)] $Message"
}

function New-SourceArchive {
    param([string]$ArchivePath)
    if (Test-Path $ArchivePath) {
        Remove-Item $ArchivePath -Force
    }
    git archive --format=zip --output=$ArchivePath HEAD
}

function Resolve-PytorchImageFamily {
    $families = gcloud.cmd compute images list --project deeplearning-platform-release --no-standard-images --filter="family~pytorch" --format="value(family)" 2>$null
    if (-not $families) {
        throw "No PyTorch image families found in deeplearning-platform-release"
    }
    $selected = ($families | Sort-Object -Unique | Select-Object -Last 1)
    if (-not $selected) {
        throw "Unable to select a PyTorch image family"
    }
    return $selected
}

function Wait-InstanceDone {
    param(
        [string]$Name,
        [string]$Project,
        [string]$ZoneName,
        [int]$MaxMinutes
    )

    $appeared = $false
    $appearDeadline = (Get-Date).AddMinutes(5)
    while ((Get-Date) -lt $appearDeadline) {
        $state = gcloud.cmd compute instances list --project $Project --zones $ZoneName --filter="name=($Name)" --format="value(status)"
        if ($state) {
            $appeared = $true
            break
        }
        Start-Sleep -Seconds 10
    }

    if (-not $appeared) {
        throw "Instance $Name did not appear in zone $ZoneName after creation"
    }

    $deadline = (Get-Date).AddMinutes($MaxMinutes)
    while ((Get-Date) -lt $deadline) {
        $state = gcloud.cmd compute instances list --project $Project --zones $ZoneName --filter="name=($Name)" --format="value(status)"
        if (-not $state) {
            return "deleted"
        }
        if ($state -in @("TERMINATED", "STOPPED")) {
            return "stopped"
        }
        Start-Sleep -Seconds 30
    }

    Write-Info "Timeout reached, deleting instance $Name"
    $previousPreference = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    gcloud.cmd compute instances delete $Name --project $Project --zone $ZoneName --quiet 2>$null | Out-Null
    $ErrorActionPreference = $previousPreference
    return "timeout"
}

function Invoke-StageRun {
    param(
        [string]$CfgName,
        [string]$RunName,
        [string]$StageName,
        [string]$SrcObject,
        [string]$BaseRunObject,
        [string]$ImageFamily,
        [int]$MaxMinutes = $PerRunMaxMinutes,
        [switch]$NoWait
    )

    $vmName = ("rrs-" + $RunName).ToLower()
    if ($vmName.Length -gt 63) {
        $vmName = $vmName.Substring(0, 63)
    }

    Write-Info "Launching $RunName with config $CfgName"
    $hardwareOptions = @(
        @{ machine = "n1-standard-8"; accelerator = "nvidia-tesla-t4"; evalDevice = "cuda" }
    )
    if ($AllowCpuFallback) {
        $hardwareOptions += @{ machine = "e2-standard-4"; evalDevice = "cpu" }
    }

    $launchedZone = $null
    $launchedHardware = $null
    foreach ($candidateZone in $Zones) {
        foreach ($hw in $hardwareOptions) {
            $isGpu = $hw.ContainsKey("accelerator")
            $target = if ($isGpu) { $hw.accelerator } else { "cpu-only" }
            Write-Info "Trying zone $candidateZone with $target"
            if ($isGpu) {
                gcloud.cmd compute instances create $vmName `
                    --project $ProjectId `
                    --zone $candidateZone `
                    --machine-type $($hw.machine) `
                    --accelerator type=$($hw.accelerator),count=1 `
                    --maintenance-policy TERMINATE `
                    --provisioning-model STANDARD `
                    --boot-disk-size 200GB `
                    --image-family $ImageFamily `
                    --image-project deeplearning-platform-release `
                    --scopes https://www.googleapis.com/auth/cloud-platform `
                    --metadata-from-file startup-script=scripts/gcp_stage_startup.sh `
                    --metadata "CFG_NAME=$CfgName,RUN_NAME=$RunName,PROJECT_ID=$ProjectId,BUCKET_URI=$BucketUri,SOURCE_OBJECT=$SrcObject,BASE_RUN_OBJECT=$BaseRunObject,MAX_MINUTES=$MaxMinutes,STAGE_NAME=$StageName,EVAL_DEVICE=$($hw.evalDevice)" | Out-Null
            } else {
                gcloud.cmd compute instances create $vmName `
                    --project $ProjectId `
                    --zone $candidateZone `
                    --machine-type $($hw.machine) `
                    --maintenance-policy MIGRATE `
                    --provisioning-model STANDARD `
                    --boot-disk-size 100GB `
                    --image-family $ImageFamily `
                    --image-project deeplearning-platform-release `
                    --scopes https://www.googleapis.com/auth/cloud-platform `
                    --metadata-from-file startup-script=scripts/gcp_stage_startup.sh `
                    --metadata "CFG_NAME=$CfgName,RUN_NAME=$RunName,PROJECT_ID=$ProjectId,BUCKET_URI=$BucketUri,SOURCE_OBJECT=$SrcObject,BASE_RUN_OBJECT=$BaseRunObject,MAX_MINUTES=$MaxMinutes,STAGE_NAME=$StageName,EVAL_DEVICE=$($hw.evalDevice)" | Out-Null
            }

            if ($LASTEXITCODE -eq 0) {
                $launchedZone = $candidateZone
                $launchedHardware = $hw
                break
            }
        }
        if ($launchedZone) {
            break
        }
    }

    if (-not $launchedZone) {
        throw "Failed to launch $RunName in all candidate zones/hardware options"
    }

    $launchedTarget = if ($launchedHardware.ContainsKey("accelerator")) { $launchedHardware.accelerator } else { "cpu-only" }
    Write-Info "Launched in $launchedZone with $launchedTarget"

    if ($NoWait) {
        return @{ Name = $vmName; Zone = $launchedZone; RunName = $RunName; MaxMinutes = $MaxMinutes }
    }

    $waitResult = Wait-InstanceDone -Name $vmName -Project $ProjectId -ZoneName $launchedZone -MaxMinutes ($MaxMinutes + 45)
    Write-Info "Run $RunName finished with instance state: $waitResult"
}

$tag = Get-Date -Format "yyyyMMdd-HHmmss"
$localArchive = "stage-src-$tag.zip"
$stageName = if ($RunStage2) { "stage2-$tag" } else { "stage1-$tag" }
$bucketName = $BucketUri -replace "^gs://", ""

Write-Info "Validating clean git state"
$gitStatus = git status --porcelain
if ($gitStatus) {
    throw "Working tree is not clean. Commit or stash before launching staged cloud jobs."
}

Write-Info "Enabling required APIs"
gcloud.cmd services enable compute.googleapis.com storage.googleapis.com --project $ProjectId | Out-Null

Write-Info "Ensuring bucket exists"
$bucketExists = gcloud.cmd storage buckets list --project $ProjectId --filter="name:$bucketName" --format="value(name)"
if (-not $bucketExists) {
    gcloud.cmd storage buckets create "gs://$bucketName" --project $ProjectId --location us-central1 --uniform-bucket-level-access | Out-Null
}

$imageFamily = Resolve-PytorchImageFamily
Write-Info "Using image family $imageFamily"

Write-Info "Creating source snapshot"
New-SourceArchive -ArchivePath $localArchive
$sourceObject = "$BucketUri/staging/$stageName/source.zip"
gcloud.cmd storage cp $localArchive $sourceObject | Out-Null
Remove-Item $localArchive -Force

Write-Info "Uploading base run artifacts"
$baseRunLocal = "outputs/rrs-base-cot-20260710-035418"
if (-not (Test-Path $baseRunLocal)) {
    throw "Missing local base run directory: $baseRunLocal"
}
$baseRunObject = "$BucketUri/base/rrs-base-cot-20260710-035418"
gcloud.cmd storage cp -r $baseRunLocal $BucketUri/base/ | Out-Null

if ($RunStage1) {
    Write-Info "Launching Stage 1 (budget cap $Stage1BudgetUsd USD)"
    $runs = @(
        @{ key = "main"; cfg = "rrs_j_cfc_distill.yaml"; run = "s1-main-$tag" },
        @{ key = "embed"; cfg = "rrs_j_cfc_ablate_embed.yaml"; run = "s1-embed-$tag" },
        @{ key = "gru"; cfg = "rrs_j_cfc_ablate_gru.yaml"; run = "s1-gru-$tag" }
    )

    $selectedKeys = @(
        $Stage1RunKeys |
            ForEach-Object { $_ -split ',' } |
            ForEach-Object { $_.Trim().ToLowerInvariant() } |
            Where-Object { $_ }
    )

    foreach ($r in $runs) {
        if ($selectedKeys -contains $r.key) {
            Invoke-StageRun -CfgName $r.cfg -RunName $r.run -StageName $stageName -SrcObject $sourceObject -BaseRunObject $baseRunObject -ImageFamily $imageFamily
        }
    }

    Write-Info "Stage 1 complete. Artifacts are under $BucketUri/$stageName"
}

if ($RunStage2) {
    Write-Info "Launching Stage 2 (budget cap $Stage2BudgetUsd USD)"

    function Resolve-LatestLocalRun {
        param([string]$Pattern)
        $dir = Get-ChildItem -Path "outputs" -Directory -Filter $Pattern | Sort-Object Name -Descending | Select-Object -First 1
        if (-not $dir) {
            throw "No local run directory matching outputs/$Pattern"
        }
        return $dir.Name
    }

    $baseC212 = Resolve-LatestLocalRun -Pattern "rrs-base-cot-c212-*"
    $baseW512 = Resolve-LatestLocalRun -Pattern "rrs-base-cot-w512-*"

    foreach ($baseDir in @($baseC212, $baseW512)) {
        Write-Info "Uploading base run $baseDir"
        gcloud.cmd storage rsync -r "outputs/$baseDir" "$BucketUri/base/$baseDir" | Out-Null
    }

    $runs2 = @(
        @{ key = "chains12"; cfg = "rrs_j_cfc_s2_chains12.yaml"; run = "s2-ch12-$tag"; base = "$BucketUri/base/$baseC212" },
        @{ key = "width512"; cfg = "rrs_j_cfc_s2_width512.yaml"; run = "s2-w512-$tag"; base = "$BucketUri/base/$baseW512" },
        @{ key = "seed2"; cfg = "rrs_j_cfc_s2_seed2.yaml"; run = "s2-seed2-$tag"; base = $baseRunObject }
    )

    $selectedKeys2 = @(
        $Stage2RunKeys |
            ForEach-Object { $_ -split ',' } |
            ForEach-Object { $_.Trim().ToLowerInvariant() } |
            Where-Object { $_ }
    )

    $launched = @()
    foreach ($r in $runs2) {
        if ($selectedKeys2 -contains $r.key) {
            $launched += Invoke-StageRun -CfgName $r.cfg -RunName $r.run -StageName $stageName -SrcObject $sourceObject -BaseRunObject $r.base -ImageFamily $imageFamily -MaxMinutes $Stage2PerRunMaxMinutes -NoWait
        }
    }

    foreach ($vm in $launched) {
        $waitResult = Wait-InstanceDone -Name $vm.Name -Project $ProjectId -ZoneName $vm.Zone -MaxMinutes ($vm.MaxMinutes + 45)
        Write-Info "Run $($vm.RunName) finished with instance state: $waitResult"
    }

    Write-Info "Stage 2 complete. Artifacts are under $BucketUri/$stageName"
}
