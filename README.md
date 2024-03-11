# computer-vision
I love you homies <3

Some library functions aren't allowed, but we could use them temporarily to quickly get a working solution then retroactively replace them with our own implementations.
This would also give us some explar results to regress on later down the line. 

Task1 TODO:
1. store error as a float and see how that changes our total error, I suspect it will lower it significantly
2. Check our canny implementation is correct. from inspecting images it seems the edges are a lot thicker than the OpenCV impl. We think it could be a problem with the non-max suppression.
3. (optional) vectorise canny functions
4. Once canny is definitely working, try finding a good set of parameters for canny. Vectorisation will help this to be plausible. Possible tricks are running it on a reduced dataset and looking up if there exist some standard values used by other people.
5. One paramter we need to better understand is the magnitude interpolation coefficient used in non-max-suppression
6. Check if our strategy of picking lines from the hough space is sound. There will generally be more than one line in the hough space corresponding to a line in the image. At teh moment we pick the largest and smallest values of theta, but does a better strategy exist? Some of this will be addressed by having a better canny edge detection algorithm.
7. The report

## Running locally
1. Install python etc
2. (Optional) setup venv in '.venv' folder so that jupyter knows it exists
```shell
python3 -m venv .venv
source bin/activate
```

3. Install dependencies
```shell
pip install numpy pandas opencv-python
```

To use the jupyter notbook you'll need some more
```shell
pip install numpy pandas opencv-python ipykernel matplotlib
```

4. Run the specified task by providive the correct dataset. E.g. to run task 1
```shell
python main.py --Task1Dataset ./Task1Dataset
```

## Task 1 
Calculate the (smaller) angle between two lines in a black and white image.

For example, for the image below it should return 40.

![Two lines at a 40 degree angle](Task1Dataset/image1.png "Two lines at a 40 degree angle")