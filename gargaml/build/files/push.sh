#! /bin/bash

msg=$1

# # html 
# jupyter nbconvert --to html notebooks/*.ipynb && mv ./notebooks/*.html ./html/


# clean notebooks root
for f in *.ipynb 
do
  echo "File -> $f"

  # clear output
  jupyter nbconvert --clear-output --inplace $f

  # .py
  jupytext --to py:percent $f
done


# clean notebooks /notebooks/
for f in ./notebooks/*.ipynb 
do
  echo "File -> $f"


  # # html 
  # jupyter nbconvert --to html $f

  # clear output
  jupyter nbconvert --clear-output --inplace $f

  # .py
  jupytext --to py:percent $f
done


# clean sandbox /sandbox/
for f in ./sandbox/*.ipynb;
do
  echo "File -> $f"

  # # html 
  # jupyter nbconvert --to html $f
  
  # clear output
  jupyter nbconvert --clear-output --inplace $f

  # .py
  jupytext --to py:percent $f
done


# cp src
# clean sandbox /sandbox/
for f in ./notebooks/*.py 
do
  echo "File -> $f"

  # # html 
  # jupyter nbconvert --to html $f

  fn=$(basename $f)
  echo "FN => $fn"

  new="./src/"$fn
  echo "new => $new"

  mv $f $new
done


# black
python -m black ./


# git
git add .
git commit -m "update $msg"
git push
