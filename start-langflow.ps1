# Starts Langflow with the Test Design nodes and the API keys from .env
#
#   powershell -ExecutionPolicy Bypass -File .\start-langflow.ps1
#
# Put your keys in .env (copy .env.example), e.g.  GEMINI_API_KEY=your-key
# .env is git-ignored, so the key never ends up in the repository.

$ErrorActionPreference = "Stop"
Set-Location -Path $PSScriptRoot

if (Test-Path ".env") {
    Get-Content ".env" | ForEach-Object {
        $line = $_.Trim()
        if ($line -and -not $line.StartsWith("#") -and $line.Contains("=")) {
            $name, $value = $line.Split("=", 2)
            if ($value.Trim()) {
                Set-Item -Path "Env:$($name.Trim())" -Value $value.Trim()
                Write-Host "Loaded $($name.Trim()) from .env"
            }
        }
    }
} else {
    Write-Host "No .env file found - nodes will only work in Offline mode or with a key pasted into the node."
}

$env:LANGFLOW_COMPONENTS_PATH = Join-Path $PSScriptRoot "langflow_flow\components"
Write-Host "Custom nodes: $env:LANGFLOW_COMPONENTS_PATH"
Write-Host "Opening Langflow at http://127.0.0.1:7860 ..."
& "$PSScriptRoot\.venv\Scripts\langflow.exe" run
