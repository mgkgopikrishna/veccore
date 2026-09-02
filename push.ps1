# Run in PowerShell:   .\push.ps1
# Cleans up sandbox leftovers, then pushes VecCore to GitHub.
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$Remote = "https://mgkgopikrishna@github.com/mgkgopikrishna/veccore.git"

Write-Host "`n[1/4] Clearing sandbox leftovers..." -ForegroundColor Cyan
Get-ChildItem .git -Recurse -Filter *.lock -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Get-ChildItem .git\objects -Recurse -Filter "tmp_obj_*" -ErrorAction SilentlyContinue | Remove-Item -Force -ErrorAction SilentlyContinue
Remove-Item -Recurse -Force .pytest_cache, .ruff_cache -ErrorAction SilentlyContinue
Get-ChildItem -Recurse -Directory -Filter __pycache__ -ErrorAction SilentlyContinue | Remove-Item -Recurse -Force -ErrorAction SilentlyContinue

Write-Host "[2/4] Verifying before it goes public..." -ForegroundColor Cyan
python -m pip install -e ".[dev,server]" --quiet
python -m pytest -q
python -m ruff check veccore tests examples
python examples/migration_demo.py | Select-Object -Last 3

Write-Host "[3/4] Committing..." -ForegroundColor Cyan
git branch -M main
git add -A
git commit -m "Add push helper" 2>$null
if ($LASTEXITCODE -ne 0) { Write-Host "      (nothing new to commit - fine)" -ForegroundColor DarkGray }

Write-Host "[4/4] Pushing..." -ForegroundColor Cyan
# git writes to stderr when the remote is absent, which strict mode treats as fatal.
# Check first instead of relying on the removal succeeding.
$remotes = git remote
if ($remotes -contains "origin") { git remote set-url origin $Remote }
else { git remote add origin $Remote }
git push -u origin main

Write-Host "`nDone. -> https://github.com/mgkgopikrishna/veccore" -ForegroundColor Green
Write-Host "If 'Repository not found': create an EMPTY repo named veccore at" -ForegroundColor Yellow
Write-Host "https://github.com/new (untick all three init options), then rerun." -ForegroundColor Yellow
