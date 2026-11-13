$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$bestEffortMount = "type=bind,source=$PSScriptRoot,target=/work"
docker version --format '{{json .Server.Version}}'
if ($LASTEXITCODE -ne 0) { throw 'Start Docker Desktop to use the preserved historical environment' }
docker image inspect hft-ganesh-rllib-2019:local --format '{{.Id}}'
if ($LASTEXITCODE -ne 0) { throw 'Build the historical image using the v3 instructions first' }
docker run --rm --network none --mount $bestEffortMount hft-ganesh-rllib-2019:local python v4/test_experiment.py
if ($LASTEXITCODE -ne 0) { throw 'Preprocessing and stability checks failed' }
docker run --rm --network none --mount $bestEffortMount hft-ganesh-rllib-2019:local python v4/test_variance_extension.py
if ($LASTEXITCODE -ne 0) { throw 'Conditional reward checks failed' }
docker run --rm --network none --mount $bestEffortMount hft-ganesh-rllib-2019:local python v4/study.py --phase pilot --workers 6
if ($LASTEXITCODE -ne 0) { throw 'Pilot comparison failed' }
docker run --rm --network none --mount $bestEffortMount hft-ganesh-rllib-2019:local python v4/study.py --phase confirm --workers 16
if ($LASTEXITCODE -ne 0) { throw 'Main experiment suite failed' }
docker run --rm --network none --mount $bestEffortMount hft-ganesh-rllib-2019:local python v4/study.py --phase diagnostic --workers 4
if ($LASTEXITCODE -ne 0) { throw 'Pricing diagnostic failed' }
docker run --rm --network none --mount $bestEffortMount hft-ganesh-rllib-2019:local python v4/variance_extension.py --workers 4
if ($LASTEXITCODE -ne 0) { throw 'Variance-reduction extension failed' }
docker run --rm --network none --mount $bestEffortMount hft-ganesh-rllib-2019:local python v4/risk_sensitivity.py --workers 15
if ($LASTEXITCODE -ne 0) { throw 'Risk-strength sensitivity failed' }
python v4/report.py
if ($LASTEXITCODE -ne 0) { throw 'Report verification failed' }
