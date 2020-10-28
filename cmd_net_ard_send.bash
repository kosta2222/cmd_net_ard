cd ../home/msys_u/code/python/cmd_net_ard
pwd
comm=$1
git add --all
git commit -m $comm
git push origin HEAD
