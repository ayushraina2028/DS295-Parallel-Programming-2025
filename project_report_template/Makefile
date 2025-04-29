all: template.pdf

template.pdf: template.dvi
	dvipdf template.dvi

template.dvi: template.tex
	latex template.tex 
	bibtex template
	latex template.tex
	latex template.tex

clean:
	rm -f *.aux *.bbl *.blg *.dvi *.log
