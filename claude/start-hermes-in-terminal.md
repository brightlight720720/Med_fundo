# How to start Hermes Agent in the terminal

Hermes Agent is Nous Research's open-source terminal agent. Checked against the
official README and quickstart on 2026-10-12.

## 1. Install (one time)

Linux, macOS, WSL2:

```
curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash
```

Windows (native PowerShell):

```
iex (irm https://hermes-agent.nousresearch.com/install.ps1)
```

The installer brings its own Python, Node.js, ripgrep and ffmpeg. Git must be
present on Linux and macOS (`git --version` to check).

Then reload the shell so the `hermes` command is on PATH:

```
source ~/.bashrc     # or: source ~/.zshrc
```

## 2. Configure (one time)

```
hermes setup          # full wizard: provider, model, API key, tools
hermes model          # just pick provider and model
hermes tools          # just enable or disable tools
hermes setup --portal # use Nous Portal instead of separate API keys
```

## 3. Start it

```
hermes                # interactive chat in the current folder
hermes --tui          # newer full-screen terminal UI, same sessions and config
```

## 4. If it does not start

```
hermes doctor         # checks Python, package, config, symlink, tools, credentials
```

If the shell says `hermes: command not found`, the install finished but the
shell was not reloaded. Open a new terminal or run the `source` command above.

Docs: https://hermes-agent.nousresearch.com/docs
