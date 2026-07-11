$ErrorActionPreference = "Stop"

$paperDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $paperDir
try {
    latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
}
finally {
    Pop-Location
}
