.PHONY: book clean serve-jb serve-mkdocs serve-jb-local serve-mkdocs-local build-mkdocs-local

book:
	poetry run jb build book

clean:
	poetry run jb clean book

serve-jb:
	poetry run ghp-import -n -p -f book/_build/html

serve-mkdocs:
	poetry run mkdocs gh-deploy

serve-jb-local:
	poetry run python -m http.server 8000 -d book/_build/html

serve-mkdocs-local:
	poetry run mkdocs serve

build-mkdocs-local:
	poetry run mkdocs build