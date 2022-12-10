# Computational-Photography

# CS445 Final Project

The project contains baby generator pipelines implemented using two different kinds of alogrithms:   
a. Triangulation Mesh Warping  
b. Feature-based morphing using `The Baier-Neely morphing algorithm`

### How to run Triangulation Mesh Warping  
1. run auto alignment with two images `female pic` and `male pic`  
   a. go to dir `baby_generator_tri/auto_align`  
   b. change the new image name to `unaligned.png`  
   c. run `python image_aligner.py`  
   d. the aligned image is under `output` as `aligned1.png`   
2. run `python baby_gen.py -h ` to query proper flags for generate baby face    
3. the baby face will be shown in a pop-out window  

### How to run Feature-based morphing 
1.Run `python "feature-based morphing.py" --input1path --input2path  --output path  --image size  --predictor path --alpha --whether bokeh --a --b --P` this method can accept any two input image and morph according to you request. Run -h for more details.  
2. The morphing result will appear at your assigned output path.
