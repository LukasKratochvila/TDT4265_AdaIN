#!/bin/bash
# MOVE TO THE FOLDER WHERE THE DATASETS RESIDE!


read -p "Did you move this script to the location where you store your datasets? (y/n)?" choice
case "$choice" in 
  n|N ) echo "Then do it :)" && exit 1;;
  y|Y ) echo "Good for you :)";;
  * ) echo "invalid";;
esac


# Check if folders already exist. Abort if they do.
if [ \( ! -d ./test_style \) -a \( ! -d ./test_content \) ]
then
    mkdir ./test_style
    mkdir ./test_content

    style=(./wikiart/*)
    content=(./train2014/*)

    # 10 images from the style set
    declare -a arr=(64870 24722 9488 73066 2976 1110 30292 28856 12956 6677)
    for i in "${arr[@]}"
    do
        echo "${style[i]} --> ./test_style/"
        mv "${style[i]}" ./test_style/
    done

    # 50 images from the content set
    declare -a arr=(71297 67088 52856 57306 35032 1171 57301 73496 19964 2198 43775 9762 78629 45524 55752 17224 43583 41680 70331 17801 77540 33152 19038 56092 42885 71988 18338 2342 17184 58120 180 10066 10934 70172 69237 66184 15087 62820 47692 13253 13185 31409 60154 50815 54753 3029 17613 66572 63232 51472)
    for i in "${arr[@]}"
    do
        echo "${content[i]} --> ./test_content/" 
        mv "${content[i]}" ./test_content
    done
else echo "Test directories already exist. Can't execute this twice."
fi
