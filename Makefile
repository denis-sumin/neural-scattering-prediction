current_dir = $(shell pwd)

.PHONY: style-fix
style-fix:
	isort fabnn setup.py
	autoflake --remove-all-unused-imports --recursive --in-place --ignore-init-module-imports \
		--exclude migrations,settings.py \
		fabnn setup.py
	black fabnn setup.py

.PHONY: style-check
style-check:
	isort --check-only fabnn setup.py
	flake8 fabnn setup.py
	black --fast --check fabnn setup.py
