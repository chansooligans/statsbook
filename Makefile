.PHONY: book clean serve serve-local

book:
	poetry run jb build book

clean:
	poetry run jb clean book

serve:
	poetry run ghp-import -n -p -f book/_build/html

serve-local:
	poetry run python -m http.server 8000 book/_build/html