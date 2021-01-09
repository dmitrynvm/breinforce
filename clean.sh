find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
rm -rf .pytest_cache breinforce.egg-info
rm -rf results
rm -rf log.txt
