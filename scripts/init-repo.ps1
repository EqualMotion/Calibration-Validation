#!/usr/bin/env pwsh

gh auth status 2>$null

if ($LASTEXITCODE -ne 0) {    
    gh auth login
    
    if ($LASTEXITCODE -ne 0) {
        Write-Error "Authentication failed, aborting."
        exit 1
    }
}

$repo = gh repo view --json nameWithOwner -q .nameWithOwner

gh api repos/$repo `
  --method PATCH `
  --field allow_merge_commit=false `
  --field allow_rebase_merge=false `
  --field allow_squash_merge=true `
  --field squash_merge_commit_title=PR_TITLE `
  --field squash_merge_commit_message=BLANK `
  --field delete_branch_on_merge=true `
  --field has_wiki=false `
  --field has_issues=false `
  --field has_projects=false

Write-Host "$repo configured"
