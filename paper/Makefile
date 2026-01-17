# Makefile for AerialWorld paper

MAIN = main
LATEX = pdflatex
BIBTEX = bibtex

.PHONY: all clean view

all: $(MAIN).pdf

$(MAIN).pdf: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.lot *.lof *.synctex.gz *.fdb_latexmk *.fls

view: $(MAIN).pdf
	open $(MAIN).pdf || xdg-open $(MAIN).pdf || evince $(MAIN).pdf

# For bibliography (if using .bib file)
bib: $(MAIN).tex
	$(LATEX) $(MAIN)
	$(BIBTEX) $(MAIN)
	$(LATEX) $(MAIN)
	$(LATEX) $(MAIN)
