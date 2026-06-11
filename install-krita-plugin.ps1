$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$dest = Join-Path $env:APPDATA "krita\pykrita"
New-Item -ItemType Directory -Force -Path $dest | Out-Null

Copy-Item -Recurse -Force "krita-plugin\ai_originality" (Join-Path $dest "ai_originality")
Copy-Item -Force "krita-plugin\ai_originality.desktop" (Join-Path $dest "ai_originality.desktop")

Write-Host ""
Write-Host "Installed to: $dest"
Write-Host ""
Write-Host "Next steps:"
Write-Host "  1. Restart Krita"
Write-Host "  2. Settings -> Configure Krita -> Python Plugin Manager"
Write-Host "  3. Enable 'Originality Check' and restart Krita again"
Write-Host "  4. Settings -> Dockers -> Originality Check"
Write-Host "  5. Keep .\run.ps1 running in another terminal"
Write-Host ""
