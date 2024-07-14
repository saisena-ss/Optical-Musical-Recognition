This is done as a part of the Computer Vision course.

## Optical Music Recognition

- ###  Problem Understanding:
    
    Building on the first problem, the task here is to build an OCR model to detect music notes from music sheet along with other symbols such as treble, bass and eigth and quarter rests so as to a music notes can be detected from a music sheet. Music sheet consists of several objects inclding notes C, D, E, F, G, A, B along with treble, bass, quarter and eight rest symbol. Overall the problem can be subdivided into several parts: 
    1. Detecting the staff line (hough transform from part1)
    2. Detecting every treble and bass symbol 
    3. Detecting the notes and annotating them

    
- ###  Approach:
    The approach here is using the techniques used to solve the part 1 problem use the same to detect the staffs. In addition to detecting the all the staffs in the image we need to find all the notes with respect to the staffs and annotate them accordingly. Inorder to find the staffs we will use our implementation of hough tranform to detect all the parallel lines. For finding the treble and bass symbol we will use the technique of template matching using cross-correlation. This can be formed by using templates of these symbols and applying those template on the image to find their locations on the image. Similar concept can be applied to find all the notes on the image. After finding the staffs, treble, bass and note locations now in order to annotate the notes we need to find their relative positions from the staff lines.

- ###  Implementation:
    - The overall code is mainly divided into two parts: Houghtransform and DetectNotes.
    - The hough tranform code remains same as implemented in part1 and is explained above.
    - We define two utility functions:
        1. Cross correlation function: Takes an image, template and a threshold value. The output of the function is the cross correlation result and the locations of detected template on the image. This function is used to apple the template over all the parts of the image and find the similarity score for that region. The threshold determines the degree to which the template must match the image.
        2. Non-maximum supression: Takes the location of template and parallel distances between the line and returns a new location array with supressed repeated locations. 
    - The templates for treble, bass, quarter and eight rest are taken from the sample images and the template images provided with the problem statement.
    - Using these templates, apply the cross correlation to detect the symbols in the image.
    - After the templates have been identified we apply non-maximum supression to avoid repeating the same note in the results.
    - For finding the notes we start with a sequential approach finding the location of treble and bass and then realtively finding the location of notes on the staffs and finding their labels.
    - Using conditions and distance measurement we try and find the label of the detected note. For example, on the bass clef all the notes that lie on the first staff of the base clef will be "A", similarly all notes on second staff will be annotated "F" and going on. We first find all the notes belonging to the treble staffs and then move to the bass staff. This will be repeated for all the pairs of treble and bass staffs.

- ###  Challenges Faced:
    - One of the important challenges faced here was to get a good metric for template matching. As all images are different with these symbols present at different locations in the image. A simple distance similarity metric was giving a poor result and detecting several noise in the image. One way to tacxkle this was to use a robust technique for template matching by normalising both the matrices and computing their dot product.
    - Another challenge was removing the repeated note detection. By running the algorithm repeatedly over the image to find different symbols, many notes and symbols were detected multiple time making the algorithm slower, inefficient and inaccurate. We solved this by the technique of non-maximum supression.
    - One of the final challenge was finding the location of notes with respect to its corresponding satff inorder to find its labels. To tackle this we used relative distance of the notes from the first staff and the treble/bass symbol. We created our own template for bass and treble symbols to identify treble/bass.

- ### Improvements over time:
    - Initially, our hough transform was able to detect only a single staff as we were extracting location with maximum peak.Later on, we modified it to find many peaks by applying some threshold of maximum peak and after doing non maximal suppression, all staves were identified properly. Here, we took the assumption that all lines are horizontal and evenly spaced.
    
<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21694/files/e80383b8-32d3-4bf4-b5a3-50925c78ec81">
 </p>
 
 <p align ="center">
Fig. 13
</p>
    
And while detecting notes, each note was being detected multiple time.

<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21694/files/2b91b306-31d9-4840-8ddc-a3f0f10d28ba">
 </p>
 
 <p align ="center">
Fig. 14
</p>
    
<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21694/files/89c5228a-21af-4577-9ea6-4b43d83e707b">
</p>

<p align ="center">
Fig. 15
</p>

We used non maximal suppression to overcome with this.
Our final results on music1.png 
    
<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21694/files/2a1fefac-ea69-4091-8ef5-ae309b1e23cf">
</p>

<p align ="center">
Fig. 16
</p>


- ### Next steps:

- Our current algorithm successfully detcted all symbols and note lable for a clear music sheet with less noisy elements. With increasing noise in the sheet can lead to incorrect classification and false positives. Further enhacement can be done to handle this situation.
- Music sheet contins msuic notes in several place above the below the fifth staff lines and have a repeating behaviour along with some special symbols which are not part of the annotation.
- The algorithm can be made more robust and efficient by carefully going over all the iterations and replacing with advanced functions available in numpy and pillow to reduce the overhead tasks which could potentially improve speed and accuracy of the results.

- ### References:
1. https://docs.scipy.org/doc/scipy/tutorial/fft.html#and-n-d-discrete-fourier-transforms 
2. https://docs.adaptive-vision.com/4.7/studio/machine_vision_guide/TemplateMatching.html
3. https://www.youtube.com/watch?v=m_11ntjkn4k
4. https://www.cse.psu.edu/~rtc12/CSE486/lecture07.pdf

Sai and Prithviraj have majorly contributed to the ideation and development of this part.
