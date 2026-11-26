# Phat hanh ban cai dat len GitHub Releases.
# Chi upload file Setup .exe trong dist/ (khong upload source zip).
#
# Cach dung:
#   powershell -ExecutionPolicy Bypass -File scripts\Phat_Hanh_Release.ps1 -Version 0.1.0
#
# Token: lay tu bien moi truong GITHUB_TOKEN, neu khong co thi lay tu
# Git Credential Manager (thong tin dang nhap da dung de git push).

param(
    [Parameter(Mandatory = $true)][string]$Version,
    [string]$Repo = "QuocCuong66/elderly-ai-installer",
    [string]$AppShortName = "Fall_and_Stroke_Warning_System"
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

$root = Split-Path -Parent $PSScriptRoot
$tag = "v$Version"
$setupFile = Join-Path $root "dist\${AppShortName}_Setup_v$Version.exe"
# Ten co dinh de link https://github.com/<repo>/releases/latest/download/<ten> luon tro dung ban moi nhat
$assetName = "${AppShortName}_Setup.exe"

if (-not (Test-Path $setupFile)) { throw "Khong tim thay $setupFile. Hay build installer truoc." }

$token = $env:GITHUB_TOKEN
if (-not $token) {
    $cred = "protocol=https`nhost=github.com`n`n" | git credential fill
    $token = ($cred | Where-Object { $_ -like "password=*" }) -replace "^password=", ""
}
if (-not $token) { throw "Khong co token GitHub. Dat bien GITHUB_TOKEN hoac dang nhap git truoc." }

$headers = @{
    Authorization          = "Bearer $token"
    Accept                 = "application/vnd.github+json"
    "X-GitHub-Api-Version" = "2022-11-28"
}
$api = "https://api.github.com/repos/$Repo"

# Tag phai ton tai tren remote
git -C $root ls-remote --exit-code --tags origin "refs/tags/$tag" | Out-Null
if ($LASTEXITCODE -ne 0) { throw "Tag $tag chua co tren origin. Chay: git tag $tag; git push origin $tag" }

# Lay release hien co cua tag, neu chua co thi tao moi
try {
    $release = Invoke-RestMethod -Headers $headers -Uri "$api/releases/tags/$tag"
    Write-Host "Da co release $tag (id $($release.id))."
} catch {
    if ($_.Exception.Response.StatusCode.value__ -ne 404) { throw }
    $body = @{
        tag_name   = $tag
        name       = "$AppShortName $tag"
        body       = "Tai bo cai dat Windows: **$assetName** ben duoi.`n`nDownload the Windows installer: **$assetName** below."
        make_latest = "true"
    } | ConvertTo-Json
    $release = Invoke-RestMethod -Method Post -Headers $headers -Uri "$api/releases" -Body $body -ContentType "application/json"
    Write-Host "Da tao release $tag (id $($release.id))."
}

# Xoa asset trung ten (khi upload lai)
foreach ($asset in $release.assets) {
    if ($asset.name -eq $assetName) {
        Invoke-RestMethod -Method Delete -Headers $headers -Uri "$api/releases/assets/$($asset.id)" | Out-Null
        Write-Host "Da xoa asset cu $assetName."
    }
}

$uploadUrl = "https://uploads.github.com/repos/$Repo/releases/$($release.id)/assets?name=$assetName"
$sizeMb = [math]::Round((Get-Item $setupFile).Length / 1MB, 1)
Write-Host "Dang upload $assetName ($sizeMb MB)..."
$asset = Invoke-RestMethod -Method Post -Headers $headers -Uri $uploadUrl -InFile $setupFile -ContentType "application/octet-stream" -TimeoutSec 3600

Write-Host ""
Write-Host "Xong. Link tai truc tiep:"
Write-Host "  $($asset.browser_download_url)"
Write-Host "  https://github.com/$Repo/releases/latest/download/$assetName"
