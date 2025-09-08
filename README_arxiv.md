
# Folie à Deux — arXiv Package

This folder contains an arXiv-ready LaTeX preprint for the **Folie à Deux** project.

## Files
- `main.tex` — the paper (uses only arXiv-safe packages)
- `metadata.tex` — edit title/authors/affiliations/emails here
- `refs.bib` — placeholder bib entries; update before submission
- `figures/tradeoff_template.png` — plot placeholder to replace with real results

## Build
```bash
# macOS (MacTeX) or Linux (TeX Live)
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# or plain:
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
```
If you don't have LaTeX installed:
- macOS: `brew install --cask mactex-no-gui` (then `export PATH=/Library/TeX/texbin:$PATH`)
- Ubuntu: `sudo apt-get install texlive-full`
- Minimal: `sudo apt-get install texlive-latex-recommended texlive-latex-extra texlive-fonts-recommended`

## Notes
- Avoid `minted` (requires shell-escape; not allowed on arXiv). We use `listings`.
- Replace placeholder references in `refs.bib` with proper entries.
- Insert real plots by overwriting `figures/tradeoff_template.png` and referencing them in `main.tex`.
- Keep all figure files and style files local in the submission zip.
