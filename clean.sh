find . | grep -E "(__pycache__|\.pyc|\.pyo$)" | xargs rm -rf
rm -rf .pytest_cache breinforce.egg-info
rm -rf log.txt
rm -rf *.dat
rm -rf checkpoints
rm -rf results
