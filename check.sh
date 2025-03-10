return_code=0

echo "flake8"
flake8 ml_validation
return_code=$(($return_code | $?))

echo "mypy"
mypy ml_validation
return_code=$(($return_code | $?))

echo "isort"
isort ml_validation --diff --check-only
return_code=$(($return_code | $?))

echo "nbqa flake8"
nbqa flake8 *.ipynb
return_code=$(($return_code | $?))

echo "nbqa mypy"
nbqa mypy *.ipynb
return_code=$(($return_code | $?))

echo "nbqa isort"
nbqa isort *.ipynb --diff --check-only
return_code=$(($return_code | $?))

exit $return_code
