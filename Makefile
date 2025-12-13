JULIA = julia
PROJECT = .

.PHONY: test

test:
	$(JULIA) --project=$(PROJECT) -e 'using Pkg; Pkg.test()'