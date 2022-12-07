# Computational-Photography

# CS445FinalProject

The project contains baby generator pipelines implemented using two different kinds of alogrithms:   
a. Triangulation Mesh Warping  
b. Feature-based morphing using `The Baier The Baier-Neely morphing algorithm`

1. open the scene_complete.ipynb file

2. modify variable `datadir` to your working directory

3. modify variables `base_img_fn` and `base_img_mask_fn` with your base image and the mask.

4. add image format for the variable `ext` to iterate custom folder and search for the image format. (optional, default JPG)

5. modify the variable `search_dir` to your custom image search base.

6. modify the parameter `expand` of function `find_similar_image` to have custom background extension on the mask. (optional, default 50)

7. all the matching images computed from function `find_similar_image` is stored in variable `bestfits`. Search and modify the parameter `matching_img` passed to function `seam_find` to create the graph cut of the image of your choice.

8. In the last part, remember to search and modify the variable `foreground` to match your image selection in step 7.

9. The final result is stored as output.jpg
