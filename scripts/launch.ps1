# LLM_Viz local launcher — what the Desktop shortcut runs.
#
# Starts the dev server (Vite + the content-editor's local backend, via
# `npm run dev`) and opens the app in the default browser once something is
# actually listening. Vite falls back to the next free port if 5173 is taken
# (another project's dev server, a previous LLM_Viz instance you forgot to
# close), so this polls a small range rather than assuming one port.
#
# Closing this window stops both servers — `concurrently -k` in package.json
# kills the content-server the moment Vite exits.

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot   # scripts/ -> project root
Set-Location $root

Write-Host "Starting LLM_Viz…" -ForegroundColor Cyan
Write-Host "(closing this window stops the server)" -ForegroundColor DarkGray
Write-Host ""

# Poll for the dev server and open the browser as soon as it answers, rather
# than opening immediately and racing Vite's startup.
Start-Job -ScriptBlock {
    for ($i = 0; $i -lt 60; $i++) {
        foreach ($port in 5173..5180) {
            try {
                $url = "http://localhost:$port/LLM_Viz/"
                $r = Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 1
                if ($r.StatusCode -eq 200) {
                    Start-Process $url
                    return
                }
            } catch {
                # Not up yet, or this port is something else — keep polling.
            }
        }
        Start-Sleep -Milliseconds 500
    }
} | Out-Null

npm run dev
