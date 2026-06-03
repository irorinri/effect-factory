# Security Policy

EffectFactory is a local desktop tool that reads local preset/effect files and
writes generated media outputs. The main security focus is avoiding unsafe file
handling, unexpected code execution paths, and bundled secrets.

## Supported Versions

The latest public release and the `main` branch are the supported versions.

## Reporting A Vulnerability

Please do not post sensitive proof-of-concept details in a public issue. If the
repository has GitHub private vulnerability reporting enabled, use that flow.
Otherwise, open a minimal public issue that says a security report is available,
or contact the maintainer through the GitHub profile.

Useful details include:

- platform and Python version
- affected effect or export path
- whether ffmpeg is involved
- minimal reproduction steps

## Scope

In scope:

- unsafe file path handling
- accidental leakage of local paths or generated private media
- crashes or hangs triggered by malformed presets/effect metadata
- unsafe plugin loading behavior

Out of scope:

- issues caused by arbitrary third-party plugins intentionally added by a user
- vulnerabilities in a system ffmpeg installation
- generated media content quality issues
