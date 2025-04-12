pytest --cov-report=xml --cov=torchsmith tests
genbadge coverage -i coverage.xml -o .badges/coverage-badge.svg
