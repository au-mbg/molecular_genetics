
render:
	quarto render

clean:
	rm -rf lessons/*html lessons/*.ipynb lessons/*_files lessons/.ipynb_checkpoints
	rm -rf _output/

move:
	cp _output/lessons/*.ipynb ../notebooks/
	cp _output/lessons/*.pdf ../notebooks/

all: clean render move