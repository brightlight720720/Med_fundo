# "command not found" when typing `claude rc`

Checked against Claude Code 2.1.296 on 2026-10-12.

## What `claude rc` is

`claude rc` is the shorthand for `claude remote-control`. It starts an interactive
Claude Code session on your own computer and makes it show up in the Claude Code app
(web or mobile) so you can drive it from there. It must be run on your own machine,
in the project folder, not inside a cloud session.

Equivalent forms on a current install:

```
claude rc
claude remote-control
claude --remote-control
```

## Why you see "command not found"

Read the exact error text in your terminal. There are two different cases.

### Case 1: `claude: command not found` (or `zsh: command not found: claude`)

The shell cannot find the `claude` binary at all. Claude Code is not installed,
or it is installed but not on your PATH.

Fix (macOS / Linux):

```
curl -fsSL https://claude.ai/install.sh | bash
```

Fix (Windows PowerShell):

```
irm https://claude.ai/install.ps1 | iex
```

Then close and reopen the terminal, and confirm with:

```
claude --version
```

If `claude --version` works in a new terminal but not the old one, the old shell
just had a stale PATH.

### Case 2: `error: unknown command 'rc'`

`claude` runs, but the installed version is older than the one that added the `rc`
alias. Update it:

```
claude update
```

Then retry `claude rc`. Until you update, the long form works on older versions:

```
claude remote-control
```

## After it starts

Run it from the folder you want to work in. The session appears in the Claude Code
app under the name it prints (default prefix is your hostname). Keep that terminal
open; closing it ends the Remote Control session.
