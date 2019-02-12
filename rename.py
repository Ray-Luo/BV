find -name "*.xml" -type f | rename 's/ /_/g'
find -name "*.JPG" -type f | rename 's/\(//g' *
find -name "*.JPG" -type f | rename 's/\)//g' *
