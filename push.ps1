param([string]$m = "$(Get-Date -Format 'yyyy-MM-dd HH:mm') auto-commit")
Set-Location $PSScriptRoot
git rev-parse --is-inside-work-tree 2>$null | Out-Null
if ($LASTEXITCODE -ne 0) { Write-Error "Not a git repo here."; exit 1 }

git add -A
git diff --cached --quiet
if ($LASTEXITCODE -ne 0) {
  git commit -m "$m"
} else {
  Write-Host "No changes."
}

git pull --rebase --autostash origin main
if ($LASTEXITCODE -eq 0) {
  git push origin main
} else {
  Write-Warning "Resolve conflicts, then: git rebase --continue"
}
