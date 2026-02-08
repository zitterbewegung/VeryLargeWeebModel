# Prevent pytest from collecting standalone scripts as test modules.
# scripts/integration_test.py is a standalone script (run via __main__),
# not a pytest test — its functions take explicit parameters, not fixtures.
collect_ignore_glob = ["scripts/*"]
