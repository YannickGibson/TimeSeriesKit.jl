JULIA = julia
PROJECT = .

.PHONY: test coverage clean-cov

test:
	$(JULIA) --project=$(PROJECT) -e 'using Pkg; Pkg.test()'

coverage:
	@echo "Running tests with coverage..."
	@$(JULIA) --project=$(PROJECT) -e 'using Pkg; Pkg.test(; coverage=true)' || true
	@echo "\nProcessing coverage..."
	@$(JULIA) -e '\
		import Pkg; \
		Pkg.activate(; temp=true); \
		Pkg.develop(path="$(PWD)"); \
		Pkg.add("Coverage"); \
		using Coverage; \
		coverage = process_folder("$(PWD)/src"); \
		covered, total = get_summary(coverage); \
		percentage = round(100 * covered / total, digits=2); \
		println("\n========================================"); \
		println("Coverage: $$covered / $$total lines"); \
		println("Percentage: $$percentage%"); \
		println("========================================"); \
	'
	@$(MAKE) clean-cov

clean-cov:
	@echo "Cleaning up coverage files..."
	@find . -name "*.cov" -type f -delete
	@rm -f coverage.info coverage.lcov