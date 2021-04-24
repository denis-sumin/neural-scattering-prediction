current_dir = $(shell pwd)

.PHONY: style-fix
style-fix:
	isort fabnn
	autoflake --remove-all-unused-imports --recursive --in-place --ignore-init-module-imports \
		--exclude migrations,settings.py \
		fabnn
	black fabnn

.PHONY: style-check
style-check:
	isort --check-only fabnn
	flake8 fabnn
	black --fast --check fabnn
