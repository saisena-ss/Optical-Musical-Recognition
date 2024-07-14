# pmaurya-ketvpate-sakrathi-saischin-a1
Computer vision Assignment 1


## Part0: Image processing in spatial and frequency spaces

- ###  Problem Understanding:
    - The problem at hand involves performing an analysis of an image of a bird in order to identify the origin of the wavy patterns that are present in the processed image. The objective is to develop a program called "remove_noise.py" that utilizes the Fourier transform algorithm to remove the noise from the image. The program should leverage the existing code from the "fourier.py" and "hough_lines.py" programs in the repository, and should be capable of detecting and eliminating the noise by means of Fourier analysis. The focus of this task is on the development of a method to remove the noise from the image   
    
- ### Implementation/ Approach:
    -  For this particular part the whole team got on online meet and went through the initial code provided. After a couple of days the team implemented their own version and completed the part0 by applying a mean filter. In the process we also implemented some of the other filters like gaussian, contour filter from PIL, contras and mean filter and random combination of various filters. After the results we finalized on just applying the mean filter to the image as it was giving decent enough results.
    -  The other approach can be to apply low pass filter after applying the FFT but the min filter gave the best result overall in comparision. This can be done by applying a block filter of k = 20 after to the fft of the image and all the frequencies above a certain threshold are set to zero (here, we set 99th percentile were set to zero). We then retrieve the de noised image by applying the inverse fft. A median filter and a edge enhance filter was then applied to the image, giving a denoised output.
  
    - The Final approach that we Implemented was to set a fine tune hyperparametrs which remove this high frequency noise manually looping through the image. The clean image by the final approach can be seen in the Fig. 1 below. The fft of the intermediate image and the final image can be seen below in the Fig.2 and Fig. 3 respectively.

<p align ="center">
<img width="239" alt="image" src= "part0/output_pichu.png">
</p>

<p align ="center">
Fig. 1
 </p>

<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21503/files/5924ab6d-ec2a-40f2-ba80-f5874f619363">
</p>

<p align ="center">
Fig. 2
 </p>

<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21503/files/7ac79373-c2b6-41b2-b144-2532f0ea47f0" >
</p>

<p align ="center">
Fig. 3
 </p>
    
 - ### Improvements over time:
 - These are some experimentations of applying low pass filter of different thresholds. 
 
<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21503/files/2da8db26-5f88-4de7-8198-cf9df1a2310d">
</p>

<p align ="center">
Fig. 4
 </p>

<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21503/files/3eeaa762-7c8b-4180-ad8c-9670aaf7da59">
 </p>

<p align ="center">
Fig. 5
 </p>
 
<p align = "center"> 
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21503/files/95edd632-a81b-4926-b7a9-53c452bf6254">
</p>

<p align ="center">
Fig. 6
 </p>

- ### Noise Source and removal: 
    -  There are different types of noises like salt and pepper noise which is background noise, Gaussian noise, Poisson noise, speckle noise. There are different techniques to identify the type of noise like either visually, histogram analysis, statistical analysis, etc. 
    
    -  Usually, if the noise is periodic, periodic patterns are shown in the frequency domain, after applying fft to the image. As we can see in the figure below the, in the frequency domain, we can see the leets HI in the 2nd quadrant and similarly IH in the 4th quandrant. 
    
<p align="center">
<img width="239" alt="image" src="https://media.github.iu.edu/user/21540/files/81623a49-1921-4a96-8a7f-069c649c6855">
</p>

<p align ="center">
Fig. 7
 </p>
    
    -  After applying low pass filter, the fft is shown below. The higher frequency parts are then removed.

<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21503/files/561ad828-b095-4bcf-8388-312b8d469e32">
</p>

<p align ="center">
Fig. 8
 </p>

## Part1:: Hough Transform

- ### Problem Understanding: 
     - Using the technique of hough transform, task is to find an object of 5 lines that are approximately parallel and approximately evenly spaced. Finding parallel lines is typical in hogh transform by using the polar coordinates. Each point maps to a sine wave in the hough space and the intersections of all those curves gets the maximum votes. This coordinate in the hough space now translates to a line coordinate in the coordinate system. But to implement this concept to find an object of 5 parallel and equi distant lines is not trivial. One of the greedy approaches is to use hough transform multiple times to get parallel lines but that is not generic and irrelevant to the problem of finding staffs in a music sheet. In order to solve this, we eed to come up with an approach where each point votes for 5 parallel line object that it belongs to and the set of lines that get the maximum votes would be our output. Sample output provided to us is below
        
    
- ###  Approach:

    With incremental changes to hough space by trying different parameters such as finding the intersection point of all sine curves in hough space and then with that the point can then vote for other parallel line which is equidistant from this line. But this approach has drawbacks such as this approach will be only useful to find 2 sets of parallel lines but not an entire object of 5 parallel equi-distant lines. Solving the problem in 2D hough space was challenging so we increase the dmiension. By using a 3rd dimension to the hough space, we can now define this 5 parallel line object using 3 paramemters. The 2D hough space requires two parameters rho - distance of line from origin and theta - angle of the line from x-axis. For this approach introducing a third parameter D - distance between lines is needed. In conclusion, now in this new 3D hough space, all points will not only vote for a line but will also vote for set of 5 lines which are at D distnace from its line that it belongs to.

- ###  Implementation:
    Implementation is performed in following steps:
    1. Apply gaussian filter to remove any noise in the image.
    2. Invert the image to convert black lines to white
    3. Apply min filter with radius 3 to remove all objects that are discontinuing objects, in our case the background text in the image will be removed.
    4. Zero out pixels where value is less than 100 - this is based on experimentation.
    5. Define rho, theta and spacing ranges - rhos goes from 0 diagonal distance to positive diagonal distance of the edge pixel, theta goes from 0 to 180 degrees, spacing can again go from diagonal/4 in increments of 2. (4 distances between parallel lines)
    6. Loop through all range of rho, theta and spacing to create a 3D matrix storing the votes from all pixels to this object, this matrix is called the accumulator.
    7. For counting, the difference between the current line rho and all possible rhos if this difference is exactly equal to the distance between 5 parallel lines then the voting is incremented. Store this vote in the accumulator for this particular pixel rho, theta and spacing position.
    8. After the counting has been completed by all pixles we then flatten the accumulator and find the position in the matrix where the max vote is accumulated. This gives us the rho, theta and spacing where the rho and theta are relelvant for finding the first line in the object and the spacing will help find all the other 4 lines parallel to it.
    9. Finally plot the highest voted lines on the image.

- ###  Challenges Faced:
    The main challenge was to figure out how to modify the 2D hough technique to find the object. There were multiple ways to acheieve this by finding parallel lines and then filtering them out. Second, was to try and find an arbitrary shape which is defined by rho and theta. But the most generic and working approach is to introduce a 3rd parameter spacing between the lines. Another challenge was in implementation, along with this robust technique the image had noisy background which was detected as a continous line by our algorithm at first. We had to perform some preprocessing to remove this noise so as to only detect the required object.
    
- Code replication:
    
    Command to run the file: python3 staff_finder.py input_image.png
    
- Output:
    
    Output will be stored in the same folder with filename detected.png which contains the original inage with parallel lines highlighted in red.
<p align = " center">
<img width="239" alt="image" src="https://media.github.iu.edu/user/21540/files/1d85d663-1929-4f4c-9ff6-0f2099a96006">
 </p>
    
<p align ="center">
Fig. 9
</p>


- ### Improvements over the time:
<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21503/files/5a0e2157-aff4-4e45-9bd3-1cda09d82e95">
</p>

<p align ="center">
Fig. 10
</p>

<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21503/files/102d2bb7-549c-4f6c-8e31-a068d4045c92">
</p>

<p align ="center">
Fig. 11
</p>

<p align = " center">
<img width="239" alt="image" src= "https://media.github.iu.edu/user/21503/files/cb65abfc-9cdf-4594-a424-a609dd464706">
</p>

<p align ="center">
Fig. 12
</p>
    
<p align = " center">
<img width="239" alt="image" src="https://media.github.iu.edu/user/21503/files/4e4da83f-65b1-4bd6-b285-1f4e066201e1">
</p>

<p align ="center">
Fig. 13
</p>

## Part2: Optical Music Recognition

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
- Part 0:
1. https://docs.scipy.org/doc/scipy/tutorial/fft.html#and-n-d-discrete-fourier-transforms

- Part 1:
1. https://en.wikipedia.org/wiki/Hough_transform
- Part 2:
1. https://docs.scipy.org/doc/scipy/tutorial/fft.html#and-n-d-discrete-fourier-transforms 
2. https://docs.adaptive-vision.com/4.7/studio/machine_vision_guide/TemplateMatching.html
3. https://www.youtube.com/watch?v=m_11ntjkn4k
4. https://www.cse.psu.edu/~rtc12/CSE486/lecture07.pdf

- ### Contributions:
Part 0: For part 0, Ketul, Prithviraj, Sakshi experimented on different methods to denoise the noise pichu image. Ketul and Sakshi have contributed towards the report.
Part 1: Ketul, Sai ,Sakshi tried their own approaches. Mainly Sai was able to ideate his code with appropriate result. Ketul has drafted the report.
Part 2: Sai and Prithviraj have majorly contributed towards the ideation and development of this part. Prithiviraj and Sai have described the part2 in detail in the report.
